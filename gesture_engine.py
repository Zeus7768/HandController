import json
import logging
import os

import numpy as np

from config import atomic_json_write, backup_file
from feature_extractor import features_from_sample, knn_features
from localization import t

FEATURE_SIZE = 63
DB_VERSION = 8

# L'identite de la main choisit la banque de reconnaissance.
SIDES = ("LEFT", "RIGHT")
HAND_SLOTS = SIDES  # nom conserve pour les appelants existants
GESTURE_NAMES = [f"G{i:02d}" for i in range(1, 31)]
UNKNOWN = "INCONNU"
MAX_GESTURE_NUMBER = 10000
MAX_GESTURE_NAME = 32

# Latence : nombre de frames identiques exigees avant de valider un geste.
STABLE_FRAMES = 2
# Au-dessus de ce seuil de confiance, une seule frame suffit.
HIGH_CONFIDENCE_THRESHOLD = 0.85
# Frames INCONNU consecutives avant de relacher un geste.
UNKNOWN_FRAMES = 2

REPLACE = "replace"
APPEND = "append"


def normalize_side(value):
    """'Left', 'left', 'LEFT' -> 'LEFT'. Tout le reste -> None."""
    if not isinstance(value, str):
        return None
    upper = value.strip().upper()
    return upper if upper in SIDES else None


def gesture_id(number):
    """G01 … G99 sur deux chiffres, puis G100, G101, sans plafond à 30."""
    number = int(number)
    if number < 1 or number > MAX_GESTURE_NUMBER:
        return None
    return f"G{number:02d}" if number < 100 else f"G{number}"


def gesture_number(name):
    if not isinstance(name, str) or len(name) < 2 or name[0] != "G" or not name[1:].isdigit():
        return None
    number = int(name[1:])
    if number < 1 or number > MAX_GESTURE_NUMBER or gesture_id(number) != name:
        return None
    return number


def clean_gesture_name(value):
    """Nom affiché. Vide refusé. Accents et espaces conservés. L'ID ne change pas."""
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text or len(text) > MAX_GESTURE_NAME:
        return None
    return text


class GestureSamples(list):
    """Liste de samples compatible avec l'ancien format, plus un nom affiché."""

    def __init__(self, samples=None, name=""):
        super().__init__(samples or [])
        self.name = name or ""


def samples_of(entry):
    if isinstance(entry, GestureSamples):
        return entry
    if isinstance(entry, dict):
        return entry.get("samples") or []
    if isinstance(entry, list):
        return entry
    return []


def name_of(entry):
    if isinstance(entry, GestureSamples):
        return entry.name or ""
    if isinstance(entry, dict):
        return entry.get("name") or ""
    return ""


class GestureEngine:
    """Reconnaissance de 30 poses statiques par main (k-NN).

    Deux banques strictement separees : LEFT et RIGHT. Une main gauche n'est
    JAMAIS comparee aux samples RIGHT, et inversement. Aucune commande
    clavier/souris ici.
    """

    def __init__(
        self,
        gestures_file="gestures.json",
        k_neighbors=5,
        recognition_threshold=0.55,
        class_margin=0.12,
        stable_frames=STABLE_FRAMES,
        samples_per_gesture=30,
        swap_hands=True,
        high_confidence=HIGH_CONFIDENCE_THRESHOLD,
        unknown_frames=UNKNOWN_FRAMES,
        distance_margin=0.04,
    ):
        self.gestures_file = gestures_file
        self.k_neighbors = k_neighbors
        self.recognition_threshold = recognition_threshold
        self.class_margin = class_margin
        self.stable_frames = max(1, stable_frames)
        self.samples_per_gesture = samples_per_gesture
        # swap_hands sert UNIQUEMENT a corriger le miroir de cv2.flip pour
        # determiner LEFT ou RIGHT. Il ne melange jamais les banques.
        self.swap_hands = swap_hands
        self.high_confidence = high_confidence
        self.unknown_frames = max(1, unknown_frames)
        self.distance_margin = max(0.0, float(distance_margin))
        self.gesture_names = list(GESTURE_NAMES)

        self.database = self.load_database()
        self._index = {side: None for side in SIDES}
        self._index_sq = {side: None for side in SIDES}
        self._labels = {side: [] for side in SIDES}
        self.rebuild_index()

        self.histories = {side: [] for side in SIDES}
        self.stable_gestures = {side: UNKNOWN for side in SIDES}
        self.cancel_calibration()

    # ---- Base de donnees ---------------------------------------------------
    def create_empty_database(self):
        return {
            "version": DB_VERSION,
            "gestures": {side: {g: GestureSamples() for g in self.gesture_names} for side in SIDES},
            "retired": {side: [] for side in SIDES},
        }

    def _clean_entry(self, value):
        if isinstance(value, dict) and "samples" in value:
            return GestureSamples(self._clean_samples(value.get("samples")), clean_gesture_name(value.get("name") or "") or "")
        if isinstance(value, dict):
            return GestureSamples()
        return GestureSamples(self._clean_samples(value), "")

    def _empty_side(self):
        return {name: GestureSamples() for name in self.gesture_names}

    def _clean_side(self, raw):
        side = self._empty_side()
        if isinstance(raw, dict):
            for name, value in raw.items():
                if gesture_number(name) is None:
                    continue
                side[name] = self._clean_entry(value)
        return side

    @staticmethod
    def _is_valid_sample(sample):
        try:
            array = np.asarray(sample, dtype=np.float32).flatten()
        except (TypeError, ValueError):
            return False
        return array.size == FEATURE_SIZE and np.isfinite(array).all()

    def _clean_samples(self, value):
        if not isinstance(value, list):
            return []
        return [sample for sample in value if self._is_valid_sample(sample)]

    def load_database(self):
        log = logging.getLogger("handcontroller")
        try:
            with open(self.gestures_file, "r", encoding="utf-8-sig") as file:
                data = json.load(file)
        except FileNotFoundError:
            log.info("gestures.json not found — starting with an empty gesture bank.")
            return self.create_empty_database()
        except json.JSONDecodeError as error:
            print(t("engine.invalid"))
            log.error("gestures.json is invalid: %s", error)
            backup_file(self.gestures_file)
            database = self.create_empty_database()
            self.database = database
            self.save_database()
            return database
        except OSError as error:
            print(t("engine.unread"))
            log.error("Unable to read gestures.json: %s", error)
            backup_file(self.gestures_file)
            return self.create_empty_database()

        if not isinstance(data, dict):
            return self.create_empty_database()

        gestures = data.get("gestures")
        migrated = False
        banks = {side: self._empty_side() for side in SIDES}

        if isinstance(gestures, dict) and any(normalize_side(key) for key in gestures):
            # V8 : deja separe par main.
            for key, value in gestures.items():
                side = normalize_side(key)
                if side:
                    banks[side] = self._clean_side(value)
            if int(data.get("version", 0)) != DB_VERSION:
                migrated = True
            else:
                print(t("engine.v8"))
        elif isinstance(gestures, dict):
            sample = next(iter(gestures.values()), None)
            if isinstance(sample, dict):
                # V6 : {"G01": {"Left": [...], "Right": [...]}} -> l'info de main existe.
                print(t("engine.v6"))
                for name in self.gesture_names:
                    entry = gestures.get(name) or {}
                    if isinstance(entry, dict):
                        for key, value in entry.items():
                            side = normalize_side(key)
                            if side:
                                banks[side][name] = self._clean_samples(value)
            else:
                # V7 : banque unique, aucune information de main.
                print(t("engine.v7"))
                print(t("engine.v7_copy"))
                print(t("engine.v7_hint"))
                for name in self.gesture_names:
                    samples = self._clean_samples(gestures.get(name, []))
                    banks["LEFT"][name] = [list(s) for s in samples]
                    banks["RIGHT"][name] = [list(s) for s in samples]
            migrated = True
        elif "LEFT" in data or "RIGHT" in data:
            # V5 : {"LEFT": {...}, "RIGHT": {...}} -> l'info de main existe.
            print(t("engine.v5"))
            for key in ("LEFT", "RIGHT"):
                banks[key] = self._clean_side(data.get(key))
            migrated = True
        else:
            return self.create_empty_database()

        database = {"version": DB_VERSION, "gestures": banks, "retired": {side: [] for side in SIDES}}
        raw_retired = data.get("retired") if isinstance(data, dict) else None
        if isinstance(raw_retired, dict):
            for side in SIDES:
                values = raw_retired.get(side) or []
                database["retired"][side] = [name for name in values if gesture_number(name)]
        if migrated:
            self.database = database
            self.save_database()
            print(t("engine.migrated"))
        return database

    def save_database(self):
        log = logging.getLogger("handcontroller")
        backup_file(self.gestures_file)
        try:
            directory = os.path.dirname(os.path.abspath(self.gestures_file))
            if directory:
                os.makedirs(directory, exist_ok=True)
            payload = {
                "version": self.database.get("version", DB_VERSION),
                "gestures": {
                    side: {
                        name: {"name": name_of(entry), "samples": [list(sample) if not isinstance(sample, list) else sample for sample in samples_of(entry)]}
                        for name, entry in bank.items()
                    }
                    for side, bank in self.database["gestures"].items()
                },
                "retired": self.database.get("retired", {side: [] for side in SIDES}),
            }
            atomic_json_write(self.gestures_file, payload)
            return True
        except OSError as error:
            print(t("engine.unwritable"))
            log.error("Unable to save gestures.json: %s", error)
            return False

    def rebuild_index(self):
        """Un index numpy par main. Aucune conversion liste -> numpy par frame."""
        for side in SIDES:
            vectors = []
            labels = []
            for name, entry in self.database["gestures"][side].items():
                if gesture_number(name) is None:
                    continue
                for sample in samples_of(entry):
                    converted = GestureEngine.canonicalize_features(sample)
                    if converted is not None:
                        vectors.append(converted)
                        labels.append(name)
            if vectors:
                index = np.ascontiguousarray(np.stack(vectors, axis=0))
                self._index[side] = index
                self._index_sq[side] = np.einsum("ij,ij->i", index, index)
                self._labels[side] = labels
            else:
                self._index[side] = None
                self._index_sq[side] = None
                self._labels[side] = []

    def sample_count(self, side, gesture):
        side = normalize_side(side)
        if side is None or gesture_number(gesture) is None:
            return 0
        return len(list(samples_of(self.database["gestures"][side].get(gesture, []))))

    def calibrated_count(self, side):
        side = normalize_side(side)
        if side is None:
            return 0
        return sum(1 for name, entry in self.database["gestures"][side].items() if samples_of(entry))

    # ---- Features ----------------------------------------------------------
    @staticmethod
    def canonicalize_features(sample):
        """Palm-frame 63-vector. Old wrist-relative samples are converted in place of use."""
        return features_from_sample(sample)

    @staticmethod
    def normalize_points(points):
        """k-NN features: wrist origin, palm orthonormal frame, size-normalized.

        Geometry Engine must keep the original filtered landmarks. This frame
        is only for recorded-gesture comparison, so a pose stays close when
        the hand tilts or moves in the image.
        """
        array = np.asarray(points, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] < 10 or array.shape[1] < 3:
            return np.zeros(FEATURE_SIZE, dtype=np.float32)
        return knn_features(array)

    @staticmethod
    def normalize_landmarks(hand):
        count = len(hand)
        if count < 10:
            return np.zeros(FEATURE_SIZE, dtype=np.float32)
        # fromiter : pas de liste de listes Python intermediaire a chaque frame.
        flat = np.fromiter((value for lm in hand for value in (lm.x, lm.y, lm.z)),
                           dtype=np.float32, count=count * 3)
        return GestureEngine.normalize_points(flat.reshape(-1, 3))

    def get_hand_side(self, result, index):
        """Retourne LEFT ou RIGHT. swap_hands corrige le miroir de cv2.flip."""
        if result.handedness is None or index >= len(result.handedness):
            return None
        side = result.handedness[index][0].category_name
        if self.swap_hands:
            side = "Right" if side == "Left" else "Left" if side == "Right" else side
        return normalize_side(side)

    @staticmethod
    def handedness_score(result, index):
        if result.handedness is None or index >= len(result.handedness):
            return 0.0
        return float(result.handedness[index][0].score)

    # ---- Reconnaissance ----------------------------------------------------
    def recognize(self, features, hand_side):
        """Cherche UNIQUEMENT dans la banque de la main donnee."""
        side = normalize_side(hand_side)
        if side is None:
            return UNKNOWN, 0.0
        index = self._index[side]
        feat = GestureEngine.canonicalize_features(features)
        if feat is None or index is None:
            return UNKNOWN, 0.0

        # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2 : aucun tableau (N, 63) alloue.
        squared = self._index_sq[side] - 2.0 * index.dot(feat) + float(feat.dot(feat))
        np.maximum(squared, 0.0, out=squared)
        k = min(self.k_neighbors, squared.size)
        nearest_idx = np.argpartition(squared, k - 1)[:k]
        nearest_idx = nearest_idx[np.argsort(squared[nearest_idx])]
        # Racine carree uniquement sur les k voisins retenus.
        nearest_dists = np.sqrt(squared[nearest_idx])
        labels = self._labels[side]
        neighbours = [(labels[int(i)], float(d)) for i, d in zip(nearest_idx, nearest_dists)]

        votes = {}
        for gesture, distance in neighbours:
            votes[gesture] = votes.get(gesture, 0.0) + (1.0 / (distance + 0.0001))

        ranked = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        best_gesture, best_votes = ranked[0]
        best_distances = [d for gesture, d in neighbours if gesture == best_gesture]
        nearest = min(best_distances)
        confidence = 1.0 / (1.0 + nearest)
        if len(ranked) > 1 and best_votes > 0:
            gap = (best_votes - ranked[1][1]) / best_votes
            if gap < 1.0:
                confidence *= 0.72 + 0.28 * max(0.0, gap)

        if len(ranked) > 1:
            second_votes = ranked[1][1]
            if best_votes <= 0 or (best_votes - second_votes) / best_votes < self.class_margin:
                return UNKNOWN, confidence
            if self.distance_margin > 0:
                second_name = ranked[1][0]
                second_distances = [d for gesture, d in neighbours if gesture == second_name]
                if second_distances:
                    best_mean = sum(best_distances) / len(best_distances)
                    second_mean = sum(second_distances) / len(second_distances)
                    if second_mean - best_mean < self.distance_margin:
                        return UNKNOWN, confidence
        if confidence < self.recognition_threshold:
            return UNKNOWN, confidence
        return best_gesture, confidence

    # ---- Stabilisation (une histoire par main) ------------------------------
    def stabilize(self, side, gesture, confidence=1.0):
        """Confiance haute -> 1 frame. Sinon stable_frames. INCONNU -> unknown_frames."""
        side = normalize_side(side)
        if side is None:
            return UNKNOWN
        history = self.histories[side]
        history.append(gesture)
        window = max(self.stable_frames, self.unknown_frames)
        if len(history) > window:
            del history[: len(history) - window]

        current = self.stable_gestures[side]

        if gesture == UNKNOWN:
            recent = history[-self.unknown_frames:]
            if len(recent) >= self.unknown_frames and all(item == UNKNOWN for item in recent):
                self.stable_gestures[side] = UNKNOWN
            return self.stable_gestures[side]

        if gesture == current:
            return current

        # Entrée depuis INCONNU : une frame suffit si la confiance est haute.
        # Passage d'un geste connu à un autre : toujours stable_frames, pour
        # qu'une frame parasite ne déclenche pas un second PRESS.
        if current == UNKNOWN and confidence >= self.high_confidence:
            needed = 1
        else:
            needed = self.stable_frames
        recent = history[-needed:]
        if len(recent) >= needed and all(item == gesture for item in recent):
            self.stable_gestures[side] = gesture
        return self.stable_gestures[side]

    def reset_hand(self, side):
        side = normalize_side(side)
        if side is not None:
            self.histories[side].clear()
            self.stable_gestures[side] = UNKNOWN

    def reset_all(self):
        for side in SIDES:
            self.reset_hand(side)

    # ---- Calibration -------------------------------------------------------
    def start_calibration(self, gesture_name, hand_side, mode=REPLACE):
        side = normalize_side(hand_side)
        if side is None or gesture_number(gesture_name) is None:
            return False
        self.calibration_side = side
        self.calibration_gesture = gesture_name
        self.calibration_mode = APPEND if mode == APPEND else REPLACE
        self.calibration_samples = []
        return True

    def add_calibration_sample(self, features):
        if self.calibration_gesture is None:
            return False
        feat = np.asarray(features, dtype=np.float32).flatten()
        if feat.size != FEATURE_SIZE or not np.isfinite(feat).all():
            return False
        if len(self.calibration_samples) < self.samples_per_gesture:
            self.calibration_samples.append(feat.tolist())
            return True
        return False

    def calibration_complete(self):
        return len(self.calibration_samples) >= self.samples_per_gesture

    def calibration_progress(self):
        return len(self.calibration_samples), self.samples_per_gesture

    def finish_calibration(self):
        side = self.calibration_side
        gesture = self.calibration_gesture
        if side is None or gesture_number(gesture) is None:
            return False
        if len(self.calibration_samples) < self.samples_per_gesture:
            print(t("calib.few"))
            return False
        bank = self.database["gestures"][side]
        current = bank.get(gesture, GestureSamples())
        kept_name = name_of(current)
        merged = list(samples_of(current)) + list(self.calibration_samples) if self.calibration_mode == APPEND else list(self.calibration_samples)
        bank[gesture] = GestureSamples(merged, kept_name)
        self.save_database()
        self.rebuild_index()
        print(t("calib.done", side=side, gesture=gesture, count=len(merged)))
        self.cancel_calibration()
        return True

    def editable_ids(self, side):
        side = normalize_side(side)
        if side is None:
            return []
        extras = [name for name in self.database["gestures"][side] if gesture_number(name) and name not in GESTURE_NAMES]
        extras.sort(key=gesture_number)
        return list(GESTURE_NAMES) + extras

    def next_gesture_id(self, side):
        """Plus petit Gxx libre. Un ID retiré n'est pas réutilisé."""
        side = normalize_side(side)
        if side is None:
            return None
        bank = self.database["gestures"][side]
        retired = set(self.database.setdefault("retired", {}).setdefault(side, []))
        for number in range(1, MAX_GESTURE_NUMBER + 1):
            gid = gesture_id(number)
            entry = bank.get(gid)
            occupied = bool(samples_of(entry)) or gid in retired or (gid not in GESTURE_NAMES and gid in bank)
            if not occupied:
                return gid
        return None

    def allocate_gesture(self, side, name=""):
        gid = self.next_gesture_id(side)
        side = normalize_side(side)
        if gid is None or side is None:
            return None
        label = clean_gesture_name(name) or ""
        self.database["gestures"][side][gid] = GestureSamples([], label)
        if gid not in GESTURE_NAMES:
            # Réserve l'ID même sans sample, pour ne pas le reproposer.
            self.database["gestures"][side][gid].name = label
        self.save_database()
        return gid

    def rename_gesture(self, side, gesture, name):
        side = normalize_side(side)
        label = clean_gesture_name(name)
        if side is None or gesture_number(gesture) is None or label is None:
            return False
        bank = self.database["gestures"][side]
        current = bank.get(gesture, GestureSamples())
        bank[gesture] = GestureSamples(list(samples_of(current)), label)
        self.save_database()
        return True

    def delete_gesture(self, side, gesture):
        """Oublie les samples et retire l'ID. Le mapping ne doit pas le réutiliser."""
        side = normalize_side(side)
        if side is None or gesture_number(gesture) is None:
            return False
        bank = self.database["gestures"][side]
        if gesture in GESTURE_NAMES:
            bank[gesture] = GestureSamples()
        else:
            bank.pop(gesture, None)
        retired = self.database.setdefault("retired", {}).setdefault(side, [])
        if gesture not in retired:
            retired.append(gesture)
        self.save_database()
        self.rebuild_index()
        return True

    def display_name(self, side, gesture):
        side = normalize_side(side)
        if side is None:
            return ""
        return name_of(self.database["gestures"][side].get(gesture))

    def pose_spread(self, side, gesture):
        """Écart moyen des samples à leur centre. None s'il y en a moins de deux."""
        side = normalize_side(side)
        if side is None:
            return None
        rows = []
        for sample in samples_of(self.database["gestures"][side].get(gesture, [])):
            array = np.asarray(sample, dtype=np.float32).ravel()
            if array.size == FEATURE_SIZE and np.isfinite(array).all():
                rows.append(array)
        if len(rows) < 2:
            return None
        stack = np.stack(rows)
        center = stack.mean(axis=0)
        return float(np.mean(np.linalg.norm(stack - center, axis=1)))

    def cancel_calibration(self):
        self.calibration_side = None
        self.calibration_gesture = None
        self.calibration_mode = REPLACE
        self.calibration_samples = []

    def reset_database(self):
        """Replace the gesture banks with an empty valid database. Backs up first."""
        self.cancel_calibration()
        self.reset_all()
        self.database = self.create_empty_database()
        saved = self.save_database()
        self.rebuild_index()
        return saved
