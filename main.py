import os
import sys
import cv2
import time
import numpy as np
import mediapipe as mp
from time import perf_counter

import config as config_mod
import input_controller as input_mod
import geometry_engine as geo
import landmark_replay as replay
import gesture_editor as editor
import system_gestures as sysg
from window_focus import interface_keys_enabled, own_window_is_foreground
from landmark_filter import OneEuroFilter, points_to_landmarks
from landmark_quality import ACCEPT, HOLD as QUALITY_HOLD, REJECT as QUALITY_REJECT, LandmarkHold, LandmarkQualityChecker
from config import (
    APP_NAME, APP_VERSION, CAMERA_RESOLUTIONS, GESTURES_FILE, MAPPING_FILE, MODEL_FILE,
    accept_camera_size, app_dir, camera_resolution_label,
    camera_size_matches, camera_try_order, default_settings,
    get_logger, load_settings, nearest_camera_preset, resource_path, save_settings,
    setup_logging, user_data_path,
)
from preview_window import (
    apply_window_icon, bind_preview_mouse, fit_preview_frame, hide_preview,
    preview_was_closed, show_preview_frame,
)
from hud_renderer import draw_debug_info, draw_hand_landmarks, draw_preview, mapping_row_parts, preset_display_name
from overlay_labels import hand_overlay_state, preset_catalog
from gesture_engine import APPEND, GestureEngine, REPLACE, SIDES, UNKNOWN, clean_gesture_name, gesture_number, name_of
from recognition_engine import RecognitionEngine
from ml_engine import format_ml_startup, get_ml_backend_status
from input_controller import InputController
from gesture_mapping import (
    ACTIONS, COMBINATION, COOLDOWN_CHOICES, GESTURE_NAMES, GestureMapping, HOLD,
    MACRO, MAX_INPUTS, MAX_STEPS, MappingError,
    NONE, PRESS, REPEAT_MAX, REPEAT_MIN, SPEED_CHOICES, STEP_ACTIONS, STEP_COMBINATION,
    normalize_action,
    STEP_HOLD, STEP_PRESS, STEP_RELEASE, STEP_REPEAT_BEGIN, STEP_REPEAT_END, STEP_WAIT, WAIT_MAX_MS,
    WAIT_MIN_MS, WAIT_STEP_MS, combination_press_inputs, describe, describe_step, mapping_hold_inputs,
    normalize_steps, preview,
    side_gesture_names,
)
import ui_context
from localization import (
    action_label, apply_language, fold_ascii, input_option_label,
    language_configured, language_display_name, persist_language, t,
)

# Diagnostic clavier au demarrage : desactive en version finale.
# Passer a True, ou lancer avec la variable d'environnement HANDCTRL_KEYTEST=1.
KEYBOARD_DIAGNOSTIC = False

# Logs de transition de geste. Uniquement sur changement, jamais par frame.
# Surcharge par settings.json / --debug.
VERBOSE_GESTURE_LOG = False

# Rafraichissement de l'overlay de performance (secondes).
PERF_REFRESH = 0.25

WINDOW_TITLE = APP_NAME

UI_MODE_MAIN = "MAIN"
UI_MODE_CALIBRATION = "CALIBRATION"
UI_MODE_GESTURE_SETTINGS = "GESTURE_SETTINGS"
UI_MODE_COMMAND_SETTINGS = "COMMAND_SETTINGS"
UI_MODE_COMBINATION_SETTINGS = "COMBINATION_SETTINGS"
UI_MODE_MACRO = "MACRO"
UI_MODE_RENAME = "RENAME"
UI_MODE_LANGUAGE = "LANGUAGE"

# waitKeyEx codes. 81/83 collide with Q/S — never treat those ASCII values as arrows.
ARROW_LEFT = frozenset((2424832, 0x250000, 65361, 0xFF51))
ARROW_RIGHT = frozenset((2555904, 0x270000, 65363, 0xFF53))


def is_left_arrow(raw):
    return int(raw or 0) in ARROW_LEFT


def is_right_arrow(raw):
    return int(raw or 0) in ARROW_RIGHT

# LEFT -> points blancs, RIGHT -> points noirs. Purement visuel.
HAND_POINT_COLORS = {"LEFT": (255, 255, 255), "RIGHT": (0, 0, 0)}
SIDE_TEXT_COLORS = {"LEFT": (245, 245, 245), "RIGHT": (120, 180, 255)}

# Couleurs de la visualisation geometrique (paume, index).
PALM_LINE_COLOR = (0, 220, 255)
INDEX_LINE_COLOR = (120, 255, 120)
PINCH_LINE_COLOR = (0, 120, 255)


def hand_point_color(side):
    return HAND_POINT_COLORS.get(side, (150, 150, 150))


def side_label(side):
    if side == "LEFT":
        return t("hand.left")
    if side == "RIGHT":
        return t("hand.right")
    return "?"


def text_size(text, scale=0.5, thick=1):
    shown = fold_ascii(str(text))
    if not shown:
        return 0, 0
    (width, height), baseline = cv2.getTextSize(shown, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    return width, height + baseline


def fit_text(text, max_width, scale=0.5, thick=1):
    shown = fold_ascii(str(text))
    if max_width <= 0:
        return shown
    while shown and text_size(shown, scale, thick)[0] > max_width:
        shown = shown[:-1]
    return shown


def put(frame, text, xy, scale=0.5, color=(255, 255, 255), thick=1, max_width=None):
    """OpenCV overlay. Accents are folded because Hershey cannot draw them."""
    height, width = frame.shape[:2]
    margin = 8
    shown = fold_ascii(str(text))
    if max_width is None:
        max_width = max(8, width - margin - max(int(xy[0]), margin))
    shown = fit_text(shown, max_width, scale, thick)
    tw, th = text_size(shown, scale, thick)
    x = max(margin, min(int(xy[0]), max(margin, width - margin - 2)))
    y = max(th + 2, min(int(xy[1]), height - 4))
    if x + tw > width - margin:
        x = max(margin, width - margin - tw)
    cv2.putText(frame, shown, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)
    return tw, th


def resolve_hand_sides(detections):
    """Attribue LEFT / RIGHT. Si MediaPipe donne deux fois le meme cote,
    la detection la plus sure garde le sien et l'autre prend l'oppose."""
    assigned = {side: None for side in SIDES}
    if not detections:
        return assigned
    ordered = sorted(detections, key=lambda det: det["score"], reverse=True)
    for det in ordered:
        side = det["side"]
        if side in SIDES and assigned[side] is None:
            assigned[side] = det
            det["side"] = side
            continue
        opposite = "RIGHT" if side == "LEFT" else "LEFT"
        if assigned[opposite] is None:
            assigned[opposite] = det
            det["side"] = opposite
    return assigned


def show_user_error(message, detail=""):
    """Blocking error screen. Does not spin at 100% CPU."""
    log = get_logger()
    log.error("%s %s", message, detail)
    print(message)
    if detail:
        print(detail)
    frame = np.zeros((280, 820, 3), dtype=np.uint8)
    put(frame, APP_NAME, (16, 40), 0.8, (0, 220, 255), 2)
    y = 90
    for line in message.split("\n"):
        put(frame, line[:78], (16, y), 0.55, (255, 255, 255), 1)
        y += 28
    put(frame, t("app.press_any_key"), (16, 250), 0.5, (180, 180, 180), 1)
    cv2.imshow(WINDOW_TITLE, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def reset_user_configuration(keep_language=None):
    """Backup then restore gestures.json, mapping.json and settings.json defaults."""
    log = get_logger()
    gestures_path = user_data_path(GESTURES_FILE)
    mapping_path = user_data_path(MAPPING_FILE)
    log.info("Reset configuration requested")
    engine = GestureEngine(gestures_path)
    mapping = GestureMapping(mapping_path)
    ok_g = engine.reset_database()
    ok_m = mapping.reset_all()
    restored = default_settings()
    if keep_language in ("fr", "en"):
        restored["language"] = keep_language
    ok_s = save_settings(restored)
    if ok_g and ok_m and ok_s:
        print(t("app.config_reset_ok"))
        log.info("Configuration reset complete")
        return True
    print(t("app.config_reset_fail"))
    log.error("Configuration reset failed gestures=%s mapping=%s settings=%s", ok_g, ok_m, ok_s)
    return False


class HandControllerApp:
    def __init__(self, settings=None):
        self.settings = settings or load_settings()
        config_mod.EMERGENCY_KEY = str(self.settings.get("emergency_key", config_mod.EMERGENCY_KEY)).upper()
        self.debug = bool(self.settings.get("debug"))
        self.debug_verbose = False
        self.countdown_seconds = int(self.settings.get("countdown_seconds", 3))
        self.camera_index = int(self.settings.get("camera_index", 0))
        self.camera_width = int(self.settings.get("camera_width", 1280))
        self.camera_height = int(self.settings.get("camera_height", 720))
        self.camera_actual_width = self.camera_width
        self.camera_actual_height = self.camera_height
        self._camera_restart = False
        self._camera_strict = False
        self._camera_reopen_previous = False
        self._camera_previous = (self.camera_width, self.camera_height)
        self.preview_visible = bool(self.settings.get("display_preview", True))
        self.preview_toggle_pending = False

        self.model_path = resource_path(MODEL_FILE)
        self.samples_per_gesture = 30
        previous = dict(self.settings)
        self.settings = sysg.migrate_legacy_profiles(app_dir(), self.settings)
        if self.settings != previous:
            save_settings(self.settings)
        self.gestures_file = user_data_path(GESTURES_FILE)
        self.mapping_file = user_data_path(MAPPING_FILE)
        self.system_gestures = sysg.migrate_system_gestures(
            self.settings.get("system_gestures"), self.mapping_file
        )
        if self.settings.get("system_gestures") != self.system_gestures:
            self.settings["system_gestures"] = self.system_gestures
            save_settings(self.settings)

        runtime = self.runtime_settings()
        self.engine = GestureEngine(
            self.gestures_file,
            k_neighbors=int(runtime.get("knn_k", 5)),
            recognition_threshold=float(runtime.get("confidence_threshold", 0.55)),
            stable_frames=int(runtime.get("stable_frames", 2)),
            samples_per_gesture=self.samples_per_gesture,
            distance_margin=float(runtime.get("knn_distance_margin", 0.04)),
        )
        self.inputs = InputController()
        # Charge une seule fois. Les menus modifient la memoire puis sauvegardent.
        self.mapping = GestureMapping(self.mapping_file)
        self.log = get_logger()

        self.running = True
        self.emergency_cut = False
        self.kill_banner = False
        self.status_message = ""
        self.lang_mode = False
        self.help_mode = False
        self.sidebar_hidden = True
        self.settings_mode = False
        self.input_panel = False
        self.input_draft = "off"
        self.input_list_open = False
        self.settings_side = "LEFT"
        self.settings_scroll = 0
        self.camera_ok = True
        self._preview_icon_applied = False
        self.hud_overlay = bool(self.settings.get("hud_overlay", False))
        self.game_osd = None
        self._game_osd_failed = False

        # Calibration k-NN (bouton). N'ecrit que gestures.json.
        self.calib_mode = False
        self.calib_phase = "hand"
        self.calib_side = "LEFT"
        self.calib_index = 0
        self.calib_end_time = 0
        self.calib_name_buffer = ""
        self._ui_hits = []

        # Calibration geometrique (B). N'ecrit que les seuils de settings.json.
        self.geo_calib_mode = False
        self.geo_calib_phase = "neutral"
        self.geo_calib_samples = {}
        self.geo_proposal = None
        self.editor_mode = False
        self.editor_side = "LEFT"
        self.editor_index = 0
        self.editor_phase = "list"

        # Menu mapping
        self.map_mode = False
        self.map_phase = "hand"
        self.map_side = "LEFT"
        self.map_index = 0
        self.rename_buffer = ""
        self.edit_type = HOLD
        self.edit_inputs = ["W"]
        self.edit_modes = [HOLD]
        self.edit_slot = 0
        self.edit_cooldown = COOLDOWN_CHOICES.index(150)
        self.macro_steps = []
        self.macro_index = 0
        self.macro_speed = SPEED_CHOICES.index(1.0)
        self.step_backup = None

        # Test de macro depuis l'editeur
        self.test_active = False
        self.test_previous_enabled = False

        # Anti-spam : dernier declenchement par (main, geste)
        self.press_last_time = {}

        self.prev_gestures = {side: UNKNOWN for side in SIDES}
        self.curr_gestures = {side: UNKNOWN for side in SIDES}
        self.confidences = {side: 0.0 for side in SIDES}
        self.detected = {side: False for side in SIDES}
        self.last_ts = 0

        # Couche NumPy additive : moteur geometrique (pinch, inclinaison,
        # direction de l'index). Etat par main, independant du k-NN.
        self.geometry = self._build_geometry()
        self.prev_specials = {side: {} for side in SIDES}
        self.curr_specials = {side: {} for side in SIDES}

        # Micro-pertes MediaPipe : compteur par main. 0 = release immediat
        # (tests construits sans __init__, et settings absents).
        try:
            self.hand_loss_grace_frames = max(0, int(self.settings.get("hand_loss_grace_frames", 2)))
        except (TypeError, ValueError):
            self.hand_loss_grace_frames = 2
        self.missed_frames = {side: 0 for side in SIDES}
        self.track_state = {side: "OK" for side in SIDES}
        self.smoothing_enabled = bool(self.settings.get("landmark_smoothing", True))
        self.smoothers = {
            side: OneEuroFilter(
                min_cutoff=float(self.settings.get("smoothing_min_cutoff", 1.2)),
                beta=float(self.settings.get("smoothing_beta", 0.45)),
                d_cutoff=float(self.settings.get("smoothing_d_cutoff", 1.15)),
            )
            for side in SIDES
        }
        self.quality = LandmarkQualityChecker()
        self.landmark_hold = LandmarkHold(max_frames=max(1, int(self.hand_loss_grace_frames or 2)))
        self._hold_display = {side: None for side in SIDES}
        self.recognition = RecognitionEngine(self.engine, self.geometry, runtime)
        self.recording = False
        self.replay_handle = None
        self.replay_frame = 0
        self.finger_mode = False
        self.finger_side = "LEFT"
        self.passive = False
        self.sys_mode = False
        self.sys_phase = "list"
        self.sys_index = 0
        self.sys_edit = None
        self.sys_combo_capture = False
        self.system_test_mode = False

        self.perf = {key: 0.0 for key in (
            "camera", "mediapipe", "smoothing", "quality", "normalisation", "reconnaissance",
            "stabilisation", "mapping", "input", "special", "affichage", "total",
        )}
        self.perf_text = "FPS: -"
        self.fps = 0.0
        self._fps_frames = 0
        self._fps_start = perf_counter()

    # ---- Horloge et mesures -------------------------------------------------
    def _build_geometry(self):
        runtime = self.runtime_settings()
        return geo.GeometryEngine(
            direction_fingers={
                "LEFT": str(runtime.get("direction_finger_left", "INDEX")).upper(),
                "RIGHT": str(runtime.get("direction_finger_right", "INDEX")).upper(),
            },
            **geo.overrides_from_settings(runtime),
        )

    def runtime_settings(self):
        return dict(self.settings or {})

    def _save_system_gestures(self):
        self.settings["system_gestures"] = sysg.normalize_system_gestures(getattr(self, "system_gestures", None))
        mode = (self.settings["system_gestures"].get("finger_up") or {}).get("mode")
        if mode:
            self.settings["finger_up_mode"] = mode
        save_settings(self.settings)

    def _system_allowed(self, side, gesture):
        if getattr(self, "system_test_mode", False):
            return False
        cfg = getattr(self, "system_gestures", None)
        if not cfg:
            return True
        return sysg.is_system_enabled(cfg, side, gesture)

    def _enable_preset_gate(self, side, gesture):
        cfg = getattr(self, "system_gestures", None)
        if not cfg:
            return
        self.system_gestures = sysg.enable_mapped_preset(cfg, side, gesture)
        self._save_system_gestures()
        builder = getattr(self, "_build_geometry", None)
        if callable(builder):
            self.geometry = builder()

    def _macro_running_for_side(self, side):
        if self.inputs.is_macro_running(side):
            return True
        return False

    def get_timestamp(self):
        """Timestamp monotonic reel, strictement croissant pour MediaPipe."""
        ts = int(time.monotonic() * 1000)
        if ts <= self.last_ts:
            ts = self.last_ts + 1
        self.last_ts = ts
        return ts

    def _preview_key_allowed(self):
        """Preview restores a hidden overlay. Otherwise it only acts when this window is in front."""
        if not getattr(self, "preview_visible", True):
            return True
        return own_window_is_foreground(WINDOW_TITLE)

    def _calib_ids(self):
        if getattr(self, "engine", None) is None:
            return []
        return self.recorded_gesture_ids(self.calib_side)

    def recorded_gesture_ids(self, side):
        """Gxx that actually exist in gestures.json (samples or a custom name)."""
        ids = []
        engine = getattr(self, "engine", None)
        if engine is None:
            return ids
        bank = engine.database["gestures"].get(side) or {}
        for gid in engine.editable_ids(side):
            entry = bank.get(gid)
            if list(editor.samples_of(entry)) or name_of(entry):
                ids.append(gid)
        return ids

    def measure(self, key, value):
        previous = self.perf.get(key, 0.0)
        self.perf[key] = previous * 0.9 + value * 0.1

    def update_perf_text(self):
        now = perf_counter()
        self._fps_frames += 1
        elapsed = now - self._fps_start
        if elapsed >= PERF_REFRESH:
            self.fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_start = now
            p = self.perf
            self.perf_text = (
                f"FPS: {self.fps:4.1f} | Cam: {p['camera'] * 1000:.1f} | "
                f"MP: {p['mediapipe'] * 1000:.1f} | Euro: {p['smoothing'] * 1000:.2f} | "
                f"Qual: {p.get('quality' , 0.0) * 1000:.2f} | "
                f"KNN: {p['reconnaissance'] * 1000:.2f} | Geo: {p['special'] * 1000:.2f} | "
                f"Stab: {p['stabilisation'] * 1000:.2f} | "
                f"In: {p['input'] * 1000:.2f} | Total: {p['total'] * 1000:.1f} ms"
            )

    # ---- Etats globaux ------------------------------------------------------
    def on_emergency_stop(self):
        self.emergency_cut = True
        self.kill_banner = True
        self.test_active = False  # emergency already stopped the test macro
        self.system_test_mode = False
        self.status_message = t("status.input_kill")
        if getattr(self, "log", None):
            self.log.warning("Emergency stop")
        if getattr(self, "inputs", None):
            self.inputs.end_key_capture()

    def on_preview_toggle(self):
        """Preview from the global listener. Applied on the next frame (OpenCV is not thread-safe)."""
        self.preview_toggle_pending = True

    def handle_preview_toggle(self):
        if not getattr(self, "preview_toggle_pending", False):
            return
        self.preview_toggle_pending = False
        self.preview_visible = not getattr(self, "preview_visible", True)
        if self.preview_visible:
            self.status_message = t("status.preview_on")
        else:
            hide_preview(WINDOW_TITLE)
            self._preview_open = False
            self._preview_icon_applied = False
            self.status_message = t("status.preview_off")
        if getattr(self, "log", None):
            self.log.info("Preview %s", "shown" if self.preview_visible else "hidden")

    def ensure_preview_visible(self):
        """Menus need the overlay. Does not force the game window to the front."""
        if not getattr(self, "preview_visible", True):
            self.preview_visible = True
            self.status_message = t("status.preview_required")

    def handle_emergency_flag(self):
        if self.emergency_cut:
            self.reset_state(keep_message=True)
            self.emergency_cut = False

    def reset_state(self, keep_message=False):
        """Remise a zero GLOBALE : urgence, Input OFF, entree/sortie de mode. Pas pour une seule main."""
        self.inputs.stop_all_macros()
        self.inputs.release_all()
        self.engine.reset_all()
        self.geometry.reset()
        self.press_last_time.clear()
        self.prev_gestures = {side: UNKNOWN for side in SIDES}
        self.curr_gestures = {side: UNKNOWN for side in SIDES}
        self.confidences = {side: 0.0 for side in SIDES}
        self.detected = {side: False for side in SIDES}
        self.prev_specials = {side: {} for side in SIDES}
        self.curr_specials = {side: {} for side in SIDES}
        self.missed_frames = {side: 0 for side in SIDES}
        self.track_state = {side: "OK" for side in SIDES}
        for smoother in getattr(self, "smoothers", {}).values():
            smoother.reset()
        checker = getattr(self, "quality", None)
        if checker is not None:
            checker.reset()
        if not keep_message:
            self.status_message = ""

    # ---- Chaine geste -> input ---------------------------------------------
    def execute_gesture(self, side, gesture):
        """PRESS et MACRO uniquement sur transition. HOLD via update_held_inputs."""
        if gesture == UNKNOWN or self.prev_gestures[side] == gesture:
            return
        entry = self.mapping.get(side, gesture)
        if entry is None or entry.get("enabled", True) is False:
            return
        if getattr(self, "debug", False) or VERBOSE_GESTURE_LOG:
            detail = f"[{', '.join(entry.get('inputs', []))}]" if entry["type"] != MACRO else f"{len(entry.get('steps', []))} etapes"
            print(f"[{side}][GESTURE] {gesture}")
            print(f"[{side}][MAPPING] {gesture} -> {entry['type']} {detail}")
        if entry["type"] == PRESS:
            if self.on_cooldown(side, gesture, entry):
                return
            self.inputs.press_combo(entry["inputs"])
        elif entry["type"] == COMBINATION:
            pressed = combination_press_inputs(entry)
            if pressed:
                if self.on_cooldown(side, gesture, entry):
                    return
                self.inputs.press_combo(pressed)
        elif entry["type"] == MACRO:
            self.inputs.start_macro(side, entry["steps"], gesture, self.mapping.get_speed(side, gesture))

    def on_cooldown(self, side, gesture, entry):
        """Anti-spam : une seule activation par fenetre de cooldown_ms."""
        cooldown = entry.get("cooldown_ms", 0) / 1000.0
        if cooldown <= 0:
            return False
        key = (side, gesture)
        now = time.monotonic()
        last = self.press_last_time.get(key, 0.0)
        if now - last < cooldown:
            if getattr(self, "debug", False) or VERBOSE_GESTURE_LOG:
                remaining = (cooldown - (now - last)) * 1000
                print(f"[{side}][COOLDOWN] {gesture} ignore ({remaining:.0f} ms restantes)")
            return True
        self.press_last_time[key] = now
        return False

    def update_held_inputs(self):
        """Chaque main declare SES maintiens k-NN. L'union est calculee par le controleur.

        Gxx et les detecteurs geometriques sont independants : une meme main
        peut tenir W (G01) et SHIFT (PINCH) en meme temps. Une touche demandee
        par deux sources n'est relachee que par la derniere.
        """
        for side in SIDES:
            entry = self.mapping.get(side, self.curr_gestures[side])
            if entry and entry.get("enabled", True) is False:
                held = []
            else:
                held = mapping_hold_inputs(entry) if entry else []
            self.inputs.set_source_hold(side, held)

    def _grace_limit(self):
        try:
            return max(0, int(getattr(self, "hand_loss_grace_frames", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _in_loss_grace(self, side):
        missed = getattr(self, "missed_frames", {}).get(side, 0)
        return missed > 0 and missed <= self._grace_limit()

    def _loss_expired(self, side):
        missed = getattr(self, "missed_frames", {}).get(side, 0)
        return missed > self._grace_limit()

    def observe_presence(self, assigned):
        """Compte les frames sans main, independamment pour LEFT et RIGHT.

        Appeler une fois par frame, avant de relacher quoi que ce soit.
        Pendant la grace : les etats (gestes, geometrie) restent geles, donc
        set_source_hold continue de declarer les memes sources.
        """
        missed = getattr(self, "missed_frames", None)
        if missed is None:
            self.missed_frames = {side: 0 for side in SIDES}
            missed = self.missed_frames
        track = getattr(self, "track_state", None)
        if track is None:
            self.track_state = {side: "OK" for side in SIDES}
            track = self.track_state
        for side in SIDES:
            if assigned.get(side) is not None:
                missed[side] = 0
                track[side] = "OK"
                self.detected[side] = True
            else:
                missed[side] = missed.get(side, 0) + 1
                self.detected[side] = False
                if missed[side] <= self._grace_limit():
                    track[side] = "GRACE"
                else:
                    track[side] = "RELEASED"

    # ---- Chaine geste geometrique -> input ----------------------------------
    def update_special_gestures(self, assigned):
        """Couche geometrique NumPy, en parallele des gestes appris.

        Meme decoupage que pour G01-G30 : detection, puis PRESS/MACRO sur
        transition, puis maintiens. Chaque famille de detecteurs (PINCH, TILT,
        INDEX) a sa PROPRE source de maintien par main, donc les familles, les
        deux mains et les gestes appris n'interferent jamais entre eux, et une
        main perdue ne relache que ses propres commandes.
        """
        for side in SIDES:
            det = assigned[side]
            if det is None:
                if self._in_loss_grace(side):
                    continue
                self.curr_specials[side] = dict(self.geometry.hand_lost(side))
                continue
            if self._quality_frozen(det):
                continue
            # Landmarks filtres (image). Jamais les features k-NN palm-frame :
            # TILT / PINCH / FINGER UP ont besoin de l'orientation reelle.
            points = det.get("geometry_points")
            if points is None:
                points = det.get("stable_points")
            if points is None:
                points = geo.points_from_features(det.get("features"))
            self.curr_specials[side] = dict(self.geometry.update(side, points))

        for side in SIDES:
            for family, gesture in self.curr_specials[side].items():
                self.execute_special_gesture(side, family, gesture)
        self.update_special_holds()
        for side in SIDES:
            self.prev_specials[side] = dict(self.curr_specials[side])

    def execute_special_gesture(self, side, family, gesture):
        """PRESS et MACRO uniquement sur transition, comme execute_gesture.

        L'hysteresis des detecteurs garantit qu'un etat tenu ne produit qu'une
        seule transition : un pincement maintenu declenche un seul PRESS.
        """
        if gesture is None or self.prev_specials[side].get(family) == gesture:
            return
        if not self._system_allowed(side, gesture):
            return
        entry = self.mapping.get(side, gesture)
        if entry is None or entry.get("enabled", True) is False:
            return
        if getattr(self, "debug", False) or VERBOSE_GESTURE_LOG:
            detail = f"[{', '.join(entry.get('inputs', []))}]" if entry["type"] != MACRO else f"{len(entry.get('steps', []))} etapes"
            print(f"[{side}][GEO:{family}] {gesture}")
            print(f"[{side}][MAPPING] {gesture} -> {entry['type']} {detail}")
        if entry["type"] == PRESS:
            if self.on_cooldown(side, gesture, entry):
                return
            self.inputs.press_combo(entry["inputs"])
        elif entry["type"] == COMBINATION:
            pressed = combination_press_inputs(entry)
            if pressed:
                if self.on_cooldown(side, gesture, entry):
                    return
                self.inputs.press_combo(pressed)
        elif entry["type"] == MACRO:
            # Owner distinct du geste appris : une macro geometrique et une
            # macro G01-G30 peuvent tourner en parallele sur la meme main.
            self.inputs.start_macro(geo.special_source(side, family), entry["steps"], gesture,
                                    self.mapping.get_speed(side, gesture))

    def update_special_holds(self):
        """Une source de maintien par (main, famille de detecteurs).

        set_source_hold recalcule l'union : le passage TILT_LEFT -> TILT_RIGHT
        relache A avant d'enfoncer D dans la meme mise a jour, et une touche
        partagee avec un geste appris n'est relachee que par sa derniere source.
        """
        for side in SIDES:
            for family in self.geometry.families(side):
                gesture = self.curr_specials[side].get(family)
                entry = self.mapping.get(side, gesture) if gesture else None
                if gesture and not self._system_allowed(side, gesture):
                    held = []
                elif entry and entry.get("enabled", True) is False:
                    held = []
                else:
                    held = mapping_hold_inputs(entry) if entry else []
                self.inputs.set_source_hold(geo.special_source(side, family), held)

    def geometry_overlay_text(self, side):
        """Etiquette compacte des detecteurs actifs d'une main, ou ''."""
        parts = []
        pinch = self.geometry.detector(side, geo.PINCH)
        if pinch is not None and pinch.active and self._system_allowed(side, pinch.gesture_id):
            parts.append("PINCH")
        for family, label in ((geo.TILT, "TILT"), (geo.INDEX, "IDX")):
            detector = self.geometry.detector(side, family)
            if detector is None or detector.direction == geo.DIR_NEUTRAL:
                continue
            gid = detector.ids.get(detector.direction) if hasattr(detector, "ids") else None
            if gid and not self._system_allowed(side, gid):
                continue
            parts.append(f"{label} {detector.direction}")
        raised = []
        for finger in geo.FINGER_UP_FINGERS:
            detector = self.geometry.detector(side, geo.finger_up_family(finger))
            if detector is not None and detector.active and self._system_allowed(side, detector.gesture_id):
                raised.append(f"{finger} UP")
        if raised:
            parts.append(" ".join(raised))
        return " ".join(parts)

    def _hud_system_state(self, side):
        raised = []
        for finger in geo.FINGER_UP_FINGERS:
            detector = self.geometry.detector(side, geo.finger_up_family(finger)) if getattr(self, "geometry", None) else None
            if detector is not None and detector.active:
                raised.append(f"{finger} UP")
        pinch = self.geometry.detector(side, geo.PINCH) if getattr(self, "geometry", None) else None
        tilt = self.geometry.detector(side, geo.TILT) if getattr(self, "geometry", None) else None
        tilt_text = tilt.direction if tilt is not None and tilt.direction != geo.DIR_NEUTRAL else "NONE"
        return {
            "finger": " ".join(raised) if raised else "NONE",
            "pinch": "ON" if pinch is not None and pinch.active else "OFF",
            "tilt": tilt_text,
        }

    def debug_hud_lines(self):
        """Texte de diagnostic. Ne rien calculer de lourd : lecture d'etat."""
        lines = []
        lines.append(t("debug.recording", state="ON" if getattr(self, "recording", False) else "OFF"))
        lines.append(t("debug.safe", state="ON" if getattr(self, "safe_mode", False) else "OFF"))
        grace_limit = self._grace_limit()
        for side in SIDES:
            lines.append(f"--- {side} ---")
            detected = "YES" if self.detected.get(side) else "NO"
            state = getattr(self, "track_state", {}).get(side, "OK")
            missed = getattr(self, "missed_frames", {}).get(side, 0)
            if state == "GRACE":
                tracking = t("debug.lost_grace", missed=missed, limit=grace_limit)
            elif state == "RELEASED":
                tracking = t("debug.lost_released")
            else:
                tracking = t("debug.ok")
            gesture = self.curr_gestures.get(side, UNKNOWN)
            specials = [
                name for name in (getattr(self, "curr_specials", {}).get(side) or {}).values()
                if name and name != UNKNOWN
            ]
            if gesture != UNKNOWN:
                gesture_text = gesture
            elif specials:
                gesture_text = " / ".join(specials)
            else:
                gesture_text = t("hand.none")
            extra = " ".join(specials) if specials and gesture != UNKNOWN else ""
            if extra:
                gesture_text = f"{gesture_text} + {extra}"
            stable = self.engine.stable_gestures.get(side, UNKNOWN) if getattr(self, "engine", None) else "?"
            sources = []
            held_names = []
            hold_sources = getattr(self.inputs, "hold_sources", {})
            prefix_geo = f"GEO:{side}:"
            for source, targets in hold_sources.items():
                if source == side or source.startswith(prefix_geo) or source == f"MACRO:{side}" or source.startswith(f"MACRO:{prefix_geo}"):
                    sources.append(source)
                    held_names.extend(self.inputs._name(target) for _, target in targets)
            lines.append(t("debug.detected", state=detected))
            lines.append(t("debug.tracking", state=tracking))
            lines.append(t("debug.gesture", name=gesture_text))
            label = ""
            if getattr(self, "engine", None) is not None and hasattr(self.engine, "display_name"):
                label = self.engine.display_name(side, gesture)
            lines.append(t("debug.name", name=label if label else "—"))
            lines.append(t("debug.confidence", value=f"{self.confidences.get(side, 0.0):.2f}"))
            stable_flag = t("debug.yes") if stable == gesture and gesture not in (UNKNOWN, "?") else t("debug.no")
            lines.append(t("debug.stable", state=stable_flag, value=stable))
            lines.append(t("debug.hold_source", value=", ".join(sources) if sources else "-"))
            lines.append(t("debug.commands_held", value=" + ".join(held_names) if held_names else "-"))
            if getattr(self, "geometry", None) is not None:
                overlay = self.geometry_overlay_text(side) or "-"
                hand = self.geometry.hands.get(side)
                finger = getattr(hand, "finger", "INDEX") if hand else "INDEX"
                detector = self.geometry.detector(side, geo.INDEX)
                direction = detector.direction if detector is not None else "-"
                sys_state = self._hud_system_state(side)
                lines.append(t("debug.geometry", value=overlay))
                lines.append(t("debug.direction", finger=finger, direction=direction))
                lines.append(t("debug.finger", value=sys_state["finger"]))
                lines.append(t("debug.pinch", value=sys_state["pinch"]))
                lines.append(t("debug.tilt", value=sys_state["tilt"]))
        lines.append("--- INPUT ---")
        if getattr(self, "kill_banner", False) and not self.inputs.enabled:
            lines.append(t("debug.emergency"))
        elif self.inputs.enabled:
            lines.append(t("debug.input", state=t("hud.enabled")))
        else:
            lines.append(t("debug.input", state=t("hud.disabled")))
        lines.append(t("debug.smoothing", state="ON" if getattr(self, "smoothing_enabled", False) else "OFF"))
        active = self.inputs.get_pressed_names()
        macros = self.inputs.running_macros()
        lines.append(t("debug.holds", value=" + ".join(active) if active else "-"))
        lines.append(t("debug.macro", value=", ".join(macros) if macros else "-"))
        return lines

    def debug_status_line(self):
        """One compact debug line. Never dumped at camera rate to the console."""
        parts = []
        for side in SIDES:
            gesture = self.curr_gestures.get(side, UNKNOWN)
            specials = [
                name for name in (getattr(self, "curr_specials", {}).get(side) or {}).values()
                if name and name != UNKNOWN
            ]
            label = gesture if gesture != UNKNOWN else (specials[0] if specials else UNKNOWN)
            conf = float(self.confidences.get(side, 0.0) or 0.0)
            if label == UNKNOWN:
                parts.append(f"{side}: {UNKNOWN}")
            else:
                parts.append(f"{side}: {label} [{conf:.2f}]")
        return " | ".join(parts)

    def _apply_smoothing(self, assigned):
        """Filter each hand before k-NN and geometry. Grace keeps the filter state."""
        smoothers = getattr(self, "smoothers", None)
        if not smoothers:
            return
        now = (getattr(self, "last_ts", 0) or 0) / 1000.0
        if now <= 0:
            now = time.monotonic()
        enabled = getattr(self, "smoothing_enabled", False)
        for side in SIDES:
            det = assigned.get(side)
            smoother = smoothers.get(side)
            if det is None:
                if smoother is not None and not self._in_loss_grace(side):
                    smoother.reset()
                continue
            raw = geo.landmark_points(det.get("hand"))
            if raw is None:
                continue
            det["raw_hand"] = det.get("hand")
            det["raw_points"] = raw
            stable = smoother.filter(raw, now) if enabled and smoother is not None else raw
            det["stable_points"] = stable

    def _apply_quality(self, assigned):
        """Accept, hold, or reject each hand before k-NN. Geometry uses the same gate."""
        checker = getattr(self, "quality", None)
        holder = getattr(self, "landmark_hold", None)
        held = getattr(self, "_hold_display", None)
        if held is None:
            self._hold_display = {side: None for side in SIDES}
            held = self._hold_display
        now = (getattr(self, "last_ts", 0) or 0) / 1000.0
        if now <= 0:
            now = time.monotonic()
        for side in SIDES:
            det = assigned.get(side)
            if det is None:
                in_grace = self._in_loss_grace(side)
                if in_grace and holder is not None:
                    reused, interpolated = holder.observe(side, None, QUALITY_HOLD, now)
                    held[side] = reused if interpolated else None
                if not in_grace:
                    if checker is not None:
                        checker.reset(side)
                    if holder is not None:
                        holder.reset(side)
                    held[side] = None
                continue
            if checker is None:
                det["quality"] = ACCEPT
                det["quality_score"] = 1.0
                det["detection_quality"] = 1.0
                det["finger_quality"] = 1.0
                det["handedness_quality"] = 1.0
                det["temporal_quality"] = 1.0
                det["interpolated"] = False
                det["display_points"] = det.get("raw_points")
                continue
            points = det.get("stable_points") if det.get("stable_points") is not None else det.get("raw_points")
            report = checker.evaluate_report(side, points, det.get("score", 1.0))
            det["quality"] = report.action
            det["quality_score"] = report.quality_score
            det["detection_quality"] = report.detection_quality
            det["finger_quality"] = report.finger_quality
            det["handedness_quality"] = report.handedness_quality
            det["temporal_quality"] = report.temporal_quality
            reused, interpolated = (None, False)
            if holder is not None:
                reused, interpolated = holder.observe(side, points, report.action, now)
            report.interpolated = interpolated
            det["interpolated"] = interpolated
            det["quality_report"] = report
            # Interpolated points are for drawing only. Recognition stays frozen on HOLD/REJECT.
            if interpolated and reused is not None:
                det["display_points"] = reused
                held[side] = reused
            else:
                det["display_points"] = det.get("raw_points")
                held[side] = points if report.action == ACCEPT else held.get(side)

    def _quality_frozen(self, det):
        return det is not None and det.get("quality") in (QUALITY_HOLD, QUALITY_REJECT)

    def _record_assigned(self, assigned):
        handle = getattr(self, "replay_handle", None)
        if handle is None:
            return
        hands = []
        for side in SIDES:
            det = assigned.get(side)
            if not det or det.get("raw_points") is None:
                continue
            hands.append(replay.hand_record(
                side, side, det.get("score", 0.0),
                det["raw_points"], det.get("stable_points", det["raw_points"]),
            ))
        replay.write_frame(handle, replay.frame_record(
            self.replay_frame, int(time.monotonic() * 1000), hands,
        ))
        self.replay_frame += 1

    def toggle_recording(self):
        """Landmark capture only. Game keys stay released for the whole recording."""
        if getattr(self, "recording", False):
            self.recording = False
            if self.replay_handle is not None:
                self.replay_handle.close()
                self.replay_handle = None
            self.status_message = t("status.recording_stop")
            return
        self.inputs.disable()
        self.reset_state()
        directory = os.path.join(app_dir(), "recordings")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, time.strftime("replay-%Y%m%d-%H%M%S.jsonl"))
        self.replay_handle = open(path, "w", encoding="utf-8")
        self.replay_frame = 0
        self.recording = True
        self.status_message = t("status.recording_start")

    def _poll_system_capture(self):
        name = self.inputs.take_captured_key()
        if not name:
            return
        edit = self.sys_edit if isinstance(self.sys_edit, dict) else {}
        if getattr(self, "sys_combo_capture", False):
            combo = self.inputs.take_captured_combo()
            if not combo:
                if name == "ESC":
                    self.inputs.end_key_capture()
                    self.sys_combo_capture = False
                    self.sys_phase = "assign"
                    self.status_message = t("status.capture_cancel")
                return
            try:
                for item in combo:
                    normalize_action(item)
            except MappingError:
                self.status_message = t("sys.reserved")
                self.inputs.begin_combo_capture()
                return
            edit["inputs"] = list(combo)
            self.sys_edit = edit
            self.inputs.end_key_capture()
            self.sys_combo_capture = False
            self.sys_phase = "assign"
            self.status_message = t("sys.detected", key="+".join(combo))
            return
        if name == "ESC":
            self.inputs.end_key_capture()
            self.sys_phase = "assign"
            self.status_message = t("status.capture_cancel")
            return
        try:
            action = normalize_action(name)
        except MappingError:
            self.status_message = t("sys.reserved")
            return
        if action is None:
            self.status_message = t("status.unsupported_key")
            return
        edit["inputs"] = [action]
        self.sys_edit = edit
        self.inputs.end_key_capture()
        self.sys_phase = "assign"
        self.status_message = t("sys.detected", key=action)

    def poll_key_capture(self):
        if getattr(self, "sys_phase", "") == "capture":
            self._poll_system_capture()
            return
        if getattr(self, "map_phase", "") != "capture":
            return
        name = self.inputs.take_captured_key()
        if not name:
            return
        if getattr(self, "capture_combo", False):
            combo = self.inputs.take_captured_combo()
            if not combo:
                if name == "ESC":
                    self.inputs.end_key_capture()
                    self.capture_combo = False
                    self.map_phase = "combo"
                    self.status_message = t("status.capture_cancel")
                return
            try:
                for item in combo:
                    normalize_action(item)
            except MappingError as error:
                self.status_message = str(error)
                self.inputs.begin_combo_capture()
                return
            self.edit_inputs = list(combo)
            self.edit_slot = len(self.edit_inputs) - 1
            self.edit_type = COMBINATION
            self.inputs.end_key_capture()
            self.capture_combo = False
            self.map_phase = "combo"
            self.status_message = t("status.combo", keys="+".join(combo))
            return
        try:
            action = normalize_action(name)
        except MappingError as error:
            self.status_message = str(error)
            return
        if action is None:
            self.status_message = t("status.unsupported_key")
            return
        self.edit_inputs[self.edit_slot] = action
        self.inputs.end_key_capture()
        self.map_phase = "combo" if len(self.edit_inputs) > 1 else "simple"
        self.status_message = t("status.key", key=action)
        gesture = self.current_map_gesture()
        cooldown = COOLDOWN_CHOICES[self.edit_cooldown]
        self.mapping.set_command(
            self.map_side, gesture, self.edit_type, self.edit_inputs,
            cooldown if self.edit_type == PRESS else None,
        )

    def collect_detections(self, result):
        """Une detection par main, avec son cote. La banque suit le cote."""
        detections = []
        if not result.hand_landmarks:
            return detections
        for index, hand in enumerate(result.hand_landmarks[:2]):
            detections.append({
                "hand": hand,
                "side": self.engine.get_hand_side(result, index),
                "score": self.engine.handedness_score(result, index),
                "raw": UNKNOWN,
                "conf": 0.0,
            })
        return detections

    def process_frame(self, frame, result, draw_overlay=True):
        detections = self.collect_detections(result)
        assigned = resolve_hand_sides(detections)
        self.observe_presence(assigned)
        start = perf_counter()
        self._apply_smoothing(assigned)
        self.measure("smoothing", perf_counter() - start)
        start = perf_counter()
        self._apply_quality(assigned)
        self.measure("quality", perf_counter() - start)
        if getattr(self, "recording", False):
            self._record_assigned(assigned)

        for side in SIDES:
            det = assigned[side]
            if det is None or self._quality_frozen(det):
                continue
            start = perf_counter()
            stable = det.get("stable_points")
            if stable is not None:
                det["geometry_points"] = stable
                features = self.engine.normalize_points(stable)
            else:
                features = self.engine.normalize_landmarks(det["hand"])
                det["geometry_points"] = None
            mid = perf_counter()
            recognizer = getattr(self, "recognition", None)
            if recognizer is not None:
                ml = recognizer.recognize_ml(
                    features, side,
                    quality=det.get("quality", ACCEPT),
                    quality_report=det.get("quality_report"),
                )
                det["raw"], det["conf"] = ml.gesture_id, ml.confidence
                det["result"] = ml
            else:
                det["raw"], det["conf"] = self.engine.recognize(features, side)
            # Features k-NN orientation-normalisees. La geometrie lit geometry_points.
            det["features"] = features
            self.measure("normalisation", mid - start)
            self.measure("reconnaissance", perf_counter() - mid)

        start = perf_counter()
        for side in SIDES:
            det = assigned[side]
            if det is None:
                if self._in_loss_grace(side):
                    # Micro-perte : on ne touche ni au k-NN ni aux holds.
                    continue
                # Grace epuisee (ou grace=0) : INCONNU. Seuls SES holds tomberont.
                self.engine.reset_hand(side)
                self.curr_gestures[side] = UNKNOWN
                self.confidences[side] = 0.0
            else:
                if self._quality_frozen(det):
                    continue
                stable = self.engine.stabilize(side, det["raw"], det["conf"])
                self.curr_gestures[side] = stable
                self.confidences[side] = det["conf"] if det["raw"] == stable else 0.0
        self.measure("stabilisation", perf_counter() - start)

        passive = getattr(self, "recording", False) or getattr(self, "passive", False) or getattr(self, "safe_mode", False)
        test_only = getattr(self, "system_test_mode", False)
        start = perf_counter()
        if not passive and not test_only:
            for side in SIDES:
                self.execute_gesture(side, self.curr_gestures[side])
        self.measure("mapping", perf_counter() - start)

        start = perf_counter()
        if not passive:
            self.update_special_gestures(assigned)
        self.measure("special", perf_counter() - start)

        # Holds k-NN et geometriques : sources independantes, union dans InputController.
        start = perf_counter()
        if not passive and not test_only:
            self.update_held_inputs()
        self.measure("input", perf_counter() - start)

        for side in SIDES:
            self.prev_gestures[side] = self.curr_gestures[side]

        start = perf_counter()
        show_points = bool((getattr(self, "settings", None) or {}).get("show_landmarks", True))
        if show_points:
            held = getattr(self, "_hold_display", {}) or {}
            for side in SIDES:
                det = assigned[side]
                display = None
                if det is not None:
                    display = det.get("display_points")
                    if display is None:
                        display = det.get("raw_points")
                    if display is None:
                        display = det.get("raw_hand") or det.get("hand")
                elif self._in_loss_grace(side):
                    display = held.get(side)
                if display is not None:
                    self._draw_landmarks(frame, display, hand_point_color(side))
                    self._draw_geometry(frame, display, side)
        if draw_overlay:
            self.draw_ui(frame)
        self.measure("affichage", perf_counter() - start)

    # ---- Affichage ----------------------------------------------------------
    def _landmark_xy(self, hand, index, width, height):
        item = hand[index]
        if hasattr(item, "x"):
            return int(item.x * width), int(item.y * height)
        return int(float(item[0]) * width), int(float(item[1]) * height)

    def _draw_landmarks(self, frame, hand, color=(150, 150, 150)):
        """Couleur imposee par l'identite de la main : gauche blanc, droite noir."""
        draw_hand_landmarks(frame, hand, color)

    def _finger_arrow_enabled(self, side, finger):
        gid = sysg.system_gesture_id("finger_up", side, finger=finger)
        return sysg.is_system_enabled(getattr(self, "system_gestures", None), side, gid)

    def _draw_geometry(self, frame, hand, side):
        """Palm skeleton plus one arrow per enabled finger-up gesture."""
        height, width = frame.shape[:2]

        def pixel(index):
            return self._landmark_xy(hand, index, width, height)

        wrist_px = pixel(geo.WRIST)
        palm_px = pixel(geo.MIDDLE_MCP)
        cv2.arrowedLine(frame, wrist_px, palm_px, PALM_LINE_COLOR, 2, tipLength=0.25)

        points = hand
        if not hasattr(hand, "shape"):
            points = geo.landmark_points(hand)
        if points is not None:
            for finger in geo.FINGER_UP_FINGERS:
                if not self._finger_arrow_enabled(side, finger):
                    continue
                if not geo.is_finger_extended(points, finger):
                    continue
                mcp, tip = geo.FINGER_SEGMENT[finger]
                cv2.arrowedLine(frame, pixel(mcp), pixel(tip), INDEX_LINE_COLOR, 2, tipLength=0.25)

        pinch = self.geometry.detector(side, geo.PINCH) if getattr(self, "geometry", None) else None
        if pinch is not None and pinch.active:
            cv2.line(frame, pixel(geo.THUMB_TIP), pixel(geo.INDEX_TIP), PINCH_LINE_COLOR, 2)

        label = self.geometry_overlay_text(side)
        if label:
            put(frame, label, (wrist_px[0] - 30, min(wrist_px[1] + 22, height - 4)), 0.42, PALM_LINE_COLOR, 1)

    def _compact_hud(self):
        """Gaming mode hides the debug and performance lines. --debug still shows them."""
        settings = getattr(self, "settings", None) or {}
        return bool(settings.get("gaming_mode", False)) and not getattr(self, "debug", False)

    def _hud_metrics(self, frame):
        height, width = frame.shape[:2]
        compact = width < 720 or height < 440
        return {
            "width": width,
            "height": height,
            "compact": compact,
            "mx": max(12, int(width * 0.02)),
            "my": max(12, int(height * 0.025)),
            "line": 15 if compact else 18,
            "title": 0.44 if compact else 0.52,
            "body": 0.36 if compact else 0.44,
            "small": 0.32 if compact else 0.38,
        }

    def _hud_command_rows(self):
        return [
            ("", t("ui.record_gestures")),
            ("", t("ui.gesture_settings")),
            ("", t("ui.input")),
            ("", t("ui.help")),
            ("", t("ui.hide_sidebar")),
        ]

    def _reset_hits(self):
        self._ui_hits = []

    def _add_hit(self, x0, y0, x1, y1, kind, *payload):
        if not hasattr(self, "_ui_hits") or self._ui_hits is None:
            self._ui_hits = []
        self._ui_hits.append((int(x0), int(y0), int(x1), int(y1), kind, payload))

    def _draw_button(self, frame, x, y, w, h, label, kind, *payload, fill=(52, 50, 48), text=(240, 240, 240), accent=None):
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), fill, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), accent or (96, 94, 90), 1)
        put(frame, label, (x + 8, y + max(16, h - 8)), 0.4, text, 1, max(8, w - 14))
        self._add_hit(x, y, x + w, y + h, kind, *payload)
        return y + h

    def _hud_input_state(self):
        if self.kill_banner and not self.inputs.enabled:
            return t("hud.input_kill"), (0, 0, 255)
        if self.inputs.enabled:
            return t("hud.enabled"), (0, 255, 0)
        return t("hud.disabled"), (0, 0, 255)

    def _sidebar_width(self, width):
        from preview_window import sidebar_width_for
        if getattr(self, "sidebar_hidden", False):
            return 0
        stored = getattr(self, "_ui_sidebar_w", 0) or 0
        if 80 <= stored < width - 16:
            return int(stored)
        return sidebar_width_for(width)

    def _sidebar_items(self):
        return [
            ("record", t("ui.record_gestures"), (0, 180, 255), False),
            ("settings", t("ui.gesture_settings"), (180, 160, 255), False),
            ("input", t("ui.input"), (0, 220, 80) if getattr(self.inputs, "enabled", False) else (50, 50, 220), False),
            ("help", t("ui.help"), (220, 220, 220), False),
        ]

    def _close_other_ui(self, keep=()):
        keep = set(keep or ())
        if "help" not in keep:
            self.help_mode = False
        if "lang" not in keep:
            self.lang_mode = False
        if "settings" not in keep:
            self.settings_mode = False
        if "input" not in keep:
            self.input_panel = False
        if "editor" not in keep:
            self.editor_mode = False
        if "map" not in keep:
            self.map_mode = False
        if "sys" not in keep:
            self.sys_mode = False
        if "sys_test" not in keep:
            self.system_test_mode = False
        if "finger" not in keep:
            self.finger_mode = False
        if "geo_calib" not in keep:
            self.geo_calib_mode = False
        if "calib" not in keep:
            if getattr(self, "calib_mode", False):
                self.engine.cancel_calibration()
            self.calib_mode = False

    def _activate_sidebar(self, action):
        self._close_other_ui()
        if action == "record":
            self.enter_record_gesture()
        elif action == "settings":
            self.enter_gesture_settings()
        elif action == "input":
            self.enter_input_panel()
        elif action == "help":
            self.help_mode = True
            self.status_message = ""
        elif action == "hide":
            self.sidebar_hidden = True
        elif action == "show":
            self.sidebar_hidden = False

    def trigger_ui_emergency(self):
        if getattr(self, "inputs", None):
            self.inputs.disable()
            self.inputs.end_key_capture()
        self.on_emergency_stop()
        self.handle_emergency_flag()
        self.status_message = t("ui.emergency_done")

    def enter_record_gesture(self):
        self.enter_calibration()
        self.calib_phase = "create"
        self.calib_name_buffer = ""
        self.calib_side = getattr(self, "calib_side", None) or "LEFT"
        self.status_message = t("ui.record_gestures")

    def enter_gesture_settings(self):
        self.help_mode = False
        self.lang_mode = False
        self.input_panel = False
        self.ensure_preview_visible()
        self.settings_mode = True
        self.settings_side = getattr(self, "settings_side", None) or "LEFT"
        self.settings_scroll = 0
        self.reset_state()

    def enter_input_panel(self):
        self.help_mode = False
        self.lang_mode = False
        self.settings_mode = False
        self.ensure_preview_visible()
        self.input_panel = True
        self.input_draft = "on" if getattr(self.inputs, "enabled", False) else "off"
        self.input_list_open = False
        self.reset_state()

    def _handle_settings_click(self, action, *payload):
        if action == "back":
            self.settings_mode = False
        elif action == "side":
            self.settings_side = payload[0] if payload else "LEFT"
            self.settings_scroll = 0
        elif action == "delete_saved" and payload:
            rows = self._recorded_rows()
            index = int(payload[0])
            if 0 <= index < len(rows) and rows[index].get("kind") == "STATIC":
                self._delete_recorded_gesture(self.settings_side, rows[index]["id"])
        elif action == "saved" and payload:
            self.editor_side = self.settings_side
            rows = self._recorded_rows()
            index = int(payload[0])
            if 0 <= index < len(rows):
                self.editor_index = index
                self.settings_mode = False
                self.editor_mode = True
                self.editor_phase = "detail"
        elif action == "sys_toggle" and payload:
            rows = [row for row in self._sys_rows() if row.get("side") == self.settings_side]
            index = int(payload[0])
            if 0 <= index < len(rows):
                self._sys_toggle_row(rows[index])
        elif action == "sys_edit" and payload:
            rows = [row for row in self._sys_rows() if row.get("side") == self.settings_side]
            index = int(payload[0])
            if 0 <= index < len(rows) and rows[index].get("kind") in ("pinch", "tilt", "finger"):
                self.settings_mode = False
                self.sys_mode = True
                self.sys_index = self._sys_rows().index(rows[index])
                self._sys_begin_edit(rows[index])
        elif action == "scroll" and payload:
            self.settings_scroll = max(0, int(self.settings_scroll or 0) + int(payload[0]))
        elif action == "camera" and len(payload) >= 2:
            self.apply_camera_resolution(payload[0], payload[1])

    def _handle_input_panel_click(self, action, *payload):
        if action == "toggle":
            self.input_list_open = not bool(getattr(self, "input_list_open", False))
        elif action == "pick":
            choice = payload[0] if payload else "off"
            self.input_draft = "on" if choice == "on" else "off"
            self.input_list_open = False
        elif action == "ok":
            self.apply_input_panel(close=True)
        elif action == "cancel" or action == "back":
            self.cancel_input_panel()
        elif action == "emergency":
            self.trigger_ui_emergency()
            self.input_draft = "off"
            self.input_list_open = False

    def apply_input_panel(self, close=False):
        if getattr(self, "input_draft", "off") == "on":
            if not self.inputs.enabled:
                self.inputs.enable()
            self.kill_banner = False
            self.status_message = ""
        else:
            if self.inputs.enabled:
                self.inputs.disable()
            self.reset_state(keep_message=True)
        if close:
            self.input_panel = False
            self.input_list_open = False

    def cancel_input_panel(self):
        self.input_panel = False
        self.input_list_open = False
        self.input_draft = "on" if self.inputs.enabled else "off"

    def draw_gesture_settings(self, frame):
        height, width = frame.shape[:2]
        self._reset_hits()
        cv2.rectangle(frame, (0, 0), (width, height), (18, 18, 18), -1)
        put(frame, t("ui.gesture_settings"), (12, 28), 0.7, (0, 220, 255), 2)
        self._draw_button(frame, 12, 40, 110, 26, t("ui.left"), "settings", "side", "LEFT",
                          fill=(40, 80, 40) if self.settings_side == "LEFT" else (40, 40, 40))
        self._draw_button(frame, 130, 40, 110, 26, t("ui.right"), "settings", "side", "RIGHT",
                          fill=(40, 80, 40) if self.settings_side == "RIGHT" else (40, 40, 40))
        self._draw_button(frame, width - 120, 8, 108, 28, t("ui.back"), "settings", "back")
        y = 86
        put(frame, t("ui.recorded_gestures"), (12, y), 0.5, (0, 220, 255), 1)
        y += 18
        self.editor_side = self.settings_side
        saved = self._recorded_rows()
        if not saved:
            put(frame, t("osd.none"), (12, y + 16), 0.42, (180, 180, 180), 1)
            y += 28
        for index, row in enumerate(saved[:6]):
            entry = self.mapping.get(self.settings_side, row["id"]) or {}
            state = t("ui.on") if row["enabled"] else t("ui.off")
            put(frame, f"{editor.format_gesture_label(row['id'], row['name'])}  {describe(entry)}  {action_label(entry.get('type') or row['action'])}  {state}", (12, y + 14), 0.38, (230, 230, 230), 1, width - 150)
            self._draw_button(frame, width - 128, y - 6, 108, 22, t("ui.delete"), "settings", "delete_saved", index, fill=(20, 20, 140))
            self._add_hit(8, y - 4, width - 136, y + 22, "settings", "saved", index)
            y += 26
        y += 10
        put(frame, t("ui.predefined"), (12, y), 0.5, (0, 220, 255), 1)
        y += 18
        predefined = [row for row in self._sys_rows() if row.get("side") == self.settings_side and row.get("kind") in ("pinch", "tilt", "finger")]
        row_h = 24
        visible = max(3, (height - y - 110) // row_h)
        max_scroll = max(0, len(predefined) - visible)
        self.settings_scroll = max(0, min(int(self.settings_scroll or 0), max_scroll))
        if max_scroll:
            self._draw_button(frame, width - 92, y - 22, 36, 20, "^", "settings", "scroll", -1)
            self._draw_button(frame, width - 50, y - 22, 36, 20, "v", "settings", "scroll", 1)
        shown = predefined[self.settings_scroll:self.settings_scroll + visible]
        for offset, row in enumerate(shown):
            index = self.settings_scroll + offset
            side, gid = self._sys_row_ids(row)
            enabled = sysg.is_system_enabled(self.system_gestures, side, gid)
            entry = self.mapping.get(side, gid) or {}
            put(frame, f"{self._sys_row_title(row)}  {describe(entry)}  {action_label(entry.get('type'))}", (12, y + 14), 0.36, (220, 220, 220), 1)
            self._draw_button(frame, width - 220, y - 4, 56, 22, t("ui.on") if enabled else t("ui.off"), "settings", "sys_toggle", index,
                              fill=(30, 110, 40) if enabled else (50, 40, 40))
            self._draw_button(frame, width - 156, y - 4, 100, 22, t("ui.configure"), "settings", "sys_edit", index)
            y += row_h
        self._draw_camera_resolution_row(frame, width, height)

    def _draw_camera_resolution_row(self, frame, width, height):
        y = height - 86
        put(frame, t("ui.camera_resolution"), (12, y), 0.45, (0, 220, 255), 1)
        current = (
            int(getattr(self, "camera_width", 1280) or 1280),
            int(getattr(self, "camera_height", 720) or 720),
        )
        x = 12
        for preset in CAMERA_RESOLUTIONS:
            label = camera_resolution_label(*preset)
            fill = (30, 110, 40) if preset == current else (52, 50, 48)
            self._draw_button(frame, x, y + 8, 130, 28, label, "settings", "camera", preset[0], preset[1], fill=fill)
            x += 138
        actual = (int(getattr(self, "camera_actual_width", 0) or 0), int(getattr(self, "camera_actual_height", 0) or 0))
        if actual[0] and actual[1] and actual != current:
            put(frame, t("ui.camera_actual", width=actual[0], height=actual[1]), (12, height - 12), 0.34, (180, 180, 180), 1)

    def apply_camera_resolution(self, width, height):
        try:
            width, height = int(width), int(height)
        except (TypeError, ValueError):
            return False
        if (width, height) not in CAMERA_RESOLUTIONS:
            width, height = nearest_camera_preset(width, height)
        previous = (
            int(getattr(self, "camera_width", 1280) or 1280),
            int(getattr(self, "camera_height", 720) or 720),
        )
        actual = (
            int(getattr(self, "camera_actual_width", 0) or 0),
            int(getattr(self, "camera_actual_height", 0) or 0),
        )
        if (width, height) == previous and camera_size_matches(actual[0], actual[1], width, height):
            return True
        self._camera_previous = previous
        self._camera_strict = True
        self._camera_reopen_previous = False
        self.camera_width = width
        self.camera_height = height
        self.settings["camera_width"] = width
        self.settings["camera_height"] = height
        save_settings(self.settings)
        self._camera_restart = True
        self.status_message = t("ui.camera_apply", width=width, height=height)
        return True

    def _revert_camera_resolution(self):
        previous = getattr(self, "_camera_previous", None)
        if not previous:
            self._camera_strict = False
            return
        width, height = int(previous[0]), int(previous[1])
        self.camera_width = width
        self.camera_height = height
        self.settings["camera_width"] = width
        self.settings["camera_height"] = height
        save_settings(self.settings)
        self._camera_strict = False

    def draw_input_panel(self, frame):
        height, width = frame.shape[:2]
        self._reset_hits()
        cv2.rectangle(frame, (0, 0), (width, height), (36, 34, 32), -1)
        put(frame, t("ui.input"), (20, 36), 0.7, (0, 220, 255), 2)
        put(frame, t("ui.game_inputs"), (20, 78), 0.55, (240, 240, 240), 1)
        draft = "on" if getattr(self, "input_draft", "off") == "on" else "off"
        box = (20, 104, width - 20, 140)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (52, 50, 48), -1)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (120, 118, 110), 1)
        put(frame, input_option_label(draft), (28, 128), 0.55, (255, 255, 255), 1)
        put(frame, "v", (width - 42, 128), 0.55, (200, 200, 200), 1)
        self._add_hit(box[0], box[1], box[2], box[3], "input_panel", "toggle")
        y = 148
        if getattr(self, "input_list_open", False):
            for value in ("on", "off"):
                selected = value == draft
                fill = (40, 90, 40) if selected else (48, 46, 44)
                cv2.rectangle(frame, (20, y), (width - 20, y + 28), fill, -1)
                put(frame, input_option_label(value), (28, y + 20), 0.5, (255, 255, 255), 1)
                self._add_hit(20, y, width - 20, y + 28, "input_panel", "pick", value)
                y += 30
        self._draw_button(frame, 20, height - 48, 110, 32, t("ui.cancel"), "input_panel", "cancel")
        self._draw_button(frame, width - 130, height - 48, 110, 32, t("ui.ok"), "input_panel", "ok", fill=(30, 110, 50))
        if self.status_message:
            put(frame, self.status_message, (20, height - 64), 0.38, (0, 220, 255), 1)

    def _on_preview_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONUP:
            return
        capturing = (
            (getattr(self, "map_mode", False) and getattr(self, "map_phase", "") == "capture")
            or (getattr(self, "sys_mode", False) and getattr(self, "sys_phase", "") == "capture")
        )
        if not capturing and not interface_keys_enabled(own_window_is_foreground(WINDOW_TITLE)):
            return
        for x0, y0, x1, y1, kind, payload in getattr(self, "_ui_hits", []) or []:
            if not (x0 <= x < x1 and y0 <= y < y1):
                continue
            if capturing and kind not in ("map_cancel", "sys_cancel", "capture_cancel"):
                continue
            self._handle_ui_click(kind, *payload)
            return

    def _handle_ui_click(self, kind, *payload):
        if kind == "sidebar":
            self._activate_sidebar(payload[0] if payload else "")
        elif kind == "lang":
            code = payload[0] if payload else "fr"
            persist_language(getattr(self, "settings", {}) or {}, code)
            self.lang_mode = False
            self.status_message = t("lang.changed", language=language_display_name(code))
        elif kind == "help_close":
            self.help_mode = False
        elif kind == "editor":
            self._handle_editor_click(payload[0] if payload else "", *(payload[1:] if payload else ()))
        elif kind == "map":
            self._handle_map_click(payload[0] if payload else "", *(payload[1:] if payload else ()))
        elif kind == "sys":
            self._handle_sys_click(payload[0] if payload else "", *(payload[1:] if payload else ()))
        elif kind == "settings":
            self._handle_settings_click(payload[0] if payload else "", *(payload[1:] if payload else ()))
        elif kind == "input_panel":
            self._handle_input_panel_click(payload[0] if payload else "", *(payload[1:] if payload else ()))
        elif kind == "calib":
            self._handle_calib_click(payload[0] if payload else "", *(payload[1:] if payload else ()))
        elif kind == "map_cancel":
            self.handle_mapping_key(27)
        elif kind == "sys_cancel":
            self.handle_system_key(27)
        elif kind == "capture_cancel":
            if getattr(self, "map_mode", False):
                self.handle_mapping_key(27)
            elif getattr(self, "sys_mode", False):
                self.handle_system_key(27)

    def _blend_rect(self, frame, pt1, pt2, color, alpha=0.45):
        x0, y0 = max(0, int(pt1[0])), max(0, int(pt1[1]))
        x1, y1 = min(frame.shape[1], int(pt2[0])), min(frame.shape[0], int(pt2[1]))
        if x1 <= x0 or y1 <= y0:
            return
        roi = frame[y0:y1, x0:x1]
        overlay = np.empty_like(roi)
        overlay[:] = color
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)

    def _draw_sidebar(self, frame, sidebar_w):
        height, width = frame.shape[:2]
        sidebar_w = max(80, min(int(sidebar_w), width - 16))
        cv2.rectangle(frame, (0, 0), (sidebar_w, height), (32, 30, 28), -1)
        cv2.line(frame, (sidebar_w - 1, 0), (sidebar_w - 1, height), (60, 58, 54), 1)
        compact = width < 720 or height < 440
        body = 0.36 if compact else 0.42
        small = 0.32 if compact else 0.36
        pad = 14 if compact else 18
        y = pad + 8
        row_h = 26 if compact else 32
        self._reset_hits()
        self._sidebar_hits = []
        for action, label, color, _lit in self._sidebar_items():
            top, bot = y - 10, y + row_h + 4
            cv2.rectangle(frame, (8, top), (sidebar_w - 8, bot), (48, 46, 44), -1)
            cv2.rectangle(frame, (8, top), (sidebar_w - 8, bot), color, 1)
            self._sidebar_hits.append((8, top, sidebar_w - 8, bot, action))
            self._add_hit(8, top, sidebar_w - 8, bot, "sidebar", action)
            put(frame, label, (pad + 8, y + 8), body, (240, 240, 240), 1, sidebar_w - pad - 20)
            y += row_h + 18
        hide_top = height - 52
        cv2.rectangle(frame, (8, hide_top), (sidebar_w - 8, hide_top + 32), (42, 40, 38), -1)
        put(frame, "[ ]  " + t("ui.hide_sidebar"), (pad, hide_top + 22), small, (210, 210, 210), 1, sidebar_w - pad * 2)
        self._add_hit(8, hide_top, sidebar_w - 8, hide_top + 32, "sidebar", "hide")
        if self.status_message:
            put(frame, self.status_message, (pad, height - 16), small, (0, 220, 255), 1, sidebar_w - pad * 2)

    def _draw_hand_card(self, frame, side, align_right, sidebar_w):
        height, width = frame.shape[:2]
        cam_left = int(sidebar_w)
        cam_w = max(8, width - cam_left)
        margin = 24
        compact = width < 720 or height < 440
        line = 16 if compact else 18
        body = 0.36 if compact else 0.42
        small = 0.32 if compact else 0.38
        present, labels = hand_overlay_state(self, side)
        title = t("osd.left") if side == "LEFT" else t("osd.right")
        lines = [title]
        if labels:
            lines.extend(labels)
        else:
            lines.append(t("osd.none"))
        panel_w = min(220, max(120, int(cam_w * 0.42)))
        panel_h = 10 + line * len(lines) + 8
        if align_right:
            x0 = width - margin - panel_w
        else:
            x0 = cam_left + margin
        x0 = max(cam_left + 8, min(x0, width - 8 - 40))
        y0 = margin
        self._blend_rect(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (18, 18, 18), 0.42)
        y = y0 + 16
        max_w = panel_w - 16
        for index, text in enumerate(lines):
            if index == 0:
                color = SIDE_TEXT_COLORS.get(side, (255, 255, 255))
                scale = body
            elif index == 1 and present:
                color = (160, 255, 160)
                scale = small
            else:
                color = (235, 235, 235)
                scale = body
            put(frame, text, (x0 + 8, y), scale, color, 1, max_w)
            y += line

    def _draw_camera_hands(self, frame, sidebar_w):
        self._draw_hand_card(frame, "LEFT", False, sidebar_w)
        self._draw_hand_card(frame, "RIGHT", True, sidebar_w)

    def draw_help(self, frame, sidebar_w=0):
        height, width = frame.shape[:2]
        x0 = max(0, int(sidebar_w))
        self._blend_rect(frame, (x0, 0), (width, height), (16, 16, 18), 0.72)
        mx = x0 + 24
        max_w = width - mx - 16
        y = 36
        put(frame, t("help.title"), (mx, y), 0.7, (0, 220, 255), 2, max_w)
        y += 34
        blocks = (
            (t("help.gestures"), (t("help.record"), t("help.predefined"))),
            (t("help.keys"), (t("help.key_settings"),)),
            (t("help.input"), (t("help.input_body"),)),
            (t("help.hud"), (t("help.hud_body"),)),
            (t("help.commands"), (
                t("help.cmd_calibrate"),
                t("help.cmd_record"),
                t("help.cmd_replay"),
                t("help.cmd_input"),
                t("help.cmd_osd"),
                t("help.cmd_keys"),
                t("help.cmd_saved"),
                t("help.cmd_language"),
                t("help.cmd_help"),
                t("help.cmd_quit"),
                t("help.cmd_emergency"),
            )),
        )
        for title, lines in blocks:
            put(frame, title, (mx, y), 0.5, (0, 220, 255), 1, max_w)
            y += 22
            for line in lines:
                put(frame, line, (mx, y), 0.4, (230, 230, 230), 1, max_w)
                y += 18
            y += 8
            if y > height - 50:
                break
        put(frame, t("help.close"), (mx, min(height - 36, y + 4)), 0.38, (180, 180, 180), 1, max_w)
        put(frame, t("help.footer"), (mx, height - 16), 0.38, (160, 160, 160), 1, max_w)

    def draw_ui(self, frame):
        compact = self._compact_hud()
        height, width = frame.shape[:2]
        self._reset_hits()
        draw_preview(self, frame, put, compact=compact)
        if getattr(self, "debug", False):
            draw_debug_info(self, frame, put, compact=compact)
        if getattr(self, "help_mode", False):
            self.draw_help(frame, 0)
        elif getattr(self, "lang_mode", False):
            box_top = min(height - 110, max(80, height // 2 - 40))
            cv2.rectangle(frame, (0, box_top), (width, min(box_top + 100, height)), (18, 18, 18), -1)
            mx = 16
            max_w = width - mx - 12
            put(frame, t("setup.choose_language"), (mx, box_top + 28), 0.55, (0, 220, 255), 1, max_w)
            put(frame, f"[1] {t('lang.fr')}    [2] {t('lang.en')}", (mx, box_top + 56), 0.5, (255, 255, 255), 1, max_w)
            put(frame, t("lang.footer"), (mx, box_top + 86), 0.4, (180, 180, 180), 1, max_w)
        elif getattr(self, "finger_mode", False):
            left = self.runtime_settings().get("direction_finger_left", "INDEX")
            right = self.runtime_settings().get("direction_finger_right", "INDEX")
            mark = self.finger_side
            box_top = min(height - 90, max(80, height // 2 - 30))
            cv2.rectangle(frame, (0, box_top), (width, min(box_top + 80, height)), (18, 18, 18), -1)
            mx = 16
            max_w = width - mx - 12
            put(frame, f"{t('hud.finger_title')}  [{'LEFT' if mark == 'LEFT' else 'left'} {left}]  [RIGHT {right}]", (mx, box_top + 28), 0.45, (0, 220, 255), 1, max_w)
            put(frame, t("hud.finger_help"), (mx, box_top + 56), 0.4, (180, 180, 180), 1, max_w)
        else:
            put(frame, t("hud.footer"), (12, height - 16), 0.38, (180, 180, 180), 1)

    def _handle_finger_key(self, key, raw=0):
        if key == 27:
            self.finger_mode = False
            self.status_message = ""
            return
        if key == ord("1"):
            self.finger_side = "LEFT"
        elif key == ord("2"):
            self.finger_side = "RIGHT"
        elif is_right_arrow(raw) or is_left_arrow(raw) or key in (ord("n"), ord("N"), ord("p"), ord("P")):
            setting = "direction_finger_left" if self.finger_side == "LEFT" else "direction_finger_right"
            names = list(geo.FINGERS)
            current = str(self.runtime_settings().get(setting, "INDEX")).upper()
            index = names.index(current) if current in names else 1
            if is_right_arrow(raw) or key in (ord("n"), ord("N")):
                step = 1
            else:
                step = -1
            self.settings[setting] = names[(index + step) % len(names)]
            save_settings(self.settings)
            self.geometry = self._build_geometry()
            self.status_message = f"{self.finger_side} : {self.settings[setting]}"

    def _handle_language_key(self, key):
        if key == 27:
            self.lang_mode = False
            return
        chosen = None
        if key == ord("1"):
            chosen = "fr"
        elif key == ord("2"):
            chosen = "en"
        if chosen is None:
            return
        persist_language(self.settings, chosen)
        self.lang_mode = False
        self.status_message = t("lang.changed", language=language_display_name(chosen))

    def ui_mode(self):
        ctx = ui_context.of(self)
        return {
            ui_context.MAIN: UI_MODE_MAIN,
            ui_context.CALIBRATION: UI_MODE_CALIBRATION,
            ui_context.GESTURE_LIST: UI_MODE_GESTURE_SETTINGS,
            ui_context.COMMAND_SETTINGS: UI_MODE_COMMAND_SETTINGS,
            ui_context.COMBINATION_SETTINGS: UI_MODE_COMBINATION_SETTINGS,
            ui_context.MACRO: UI_MODE_MACRO,
            ui_context.RENAME: UI_MODE_RENAME,
            ui_context.LANGUAGE: UI_MODE_LANGUAGE,
        }.get(ctx, ctx)

    def _ui_consumes_text(self):
        return ui_context.is_text_context(ui_context.of(self))

    def toggle_hud_overlay(self):
        self.hud_overlay = not bool(getattr(self, "hud_overlay", False))
        if isinstance(getattr(self, "settings", None), dict):
            self.settings["hud_overlay"] = self.hud_overlay
            if "camera_index" in self.settings:
                save_settings(self.settings)
        self.status_message = t("status.overlay_on") if self.hud_overlay else t("status.overlay_off")
        self._sync_game_osd()

    def _ensure_game_osd(self):
        osd = getattr(self, "game_osd", None)
        if osd is None:
            from game_osd import GameOSD
            osd = GameOSD()
            self.game_osd = osd
        return osd

    def _sync_game_osd(self):
        """External click-through OSD follows S (hud_overlay). Never takes focus.
        Hidden OSD creates no window; recognition keeps running either way."""
        osd = getattr(self, "game_osd", None)
        if not getattr(self, "hud_overlay", False):
            if osd is not None and getattr(osd, "visible", False):
                osd.hide()
            self._log_osd_state(False)
            return
        try:
            from overlay_labels import osd_paint_lines
            osd = self._ensure_game_osd()
            osd.show()
            osd.update(osd_paint_lines(self))
            self._log_osd_state(self.game_osd_active())
        except Exception:
            if not getattr(self, "_game_osd_failed", False):
                self._game_osd_failed = True
                log = getattr(self, "log", None)
                if log is not None:
                    log.exception("Game OSD update failed")

    def _log_osd_state(self, active):
        """One console/log line when the real OSD state changes, never per frame."""
        if getattr(self, "_osd_logged_state", None) is active:
            return
        self._osd_logged_state = active
        log = getattr(self, "log", None)
        if log is not None:
            log.info("Screen overlay: %s", "OK (external OSD window shown)" if active else "Off")

    def _log_gesture_changes(self):
        """Log stable recognitions when they change (k-NN id + predefined gestures), per hand."""
        from overlay_labels import active_preset_labels

        last = getattr(self, "_logged_gestures", None)
        if last is None:
            last = self._logged_gestures = {side: "no gesture" for side in SIDES}
        log = getattr(self, "log", None)
        for side in SIDES:
            gesture = (getattr(self, "curr_gestures", None) or {}).get(side, UNKNOWN)
            parts = ([gesture] if gesture and gesture != UNKNOWN else []) + active_preset_labels(self, side)
            text = " + ".join(parts) if parts else "no gesture"
            if last.get(side) == text:
                continue
            last[side] = text
            if log is not None:
                conf = float((getattr(self, "confidences", None) or {}).get(side, 0.0) or 0.0)
                log.info("[%s] %s%s", side, text, f" (conf {conf:.2f})" if gesture and gesture != UNKNOWN else "")

    def game_osd_active(self):
        """True only when the external OSD window is really shown."""
        osd = getattr(self, "game_osd", None)
        return bool(osd is not None and getattr(self, "hud_overlay", False) and osd.is_active())

    def handle_idle_key(self, key, raw=0):
        """Preview shortcuts. Ignored when the HandController window is not focused."""
        if key in (255, 0) and not is_left_arrow(raw) and not is_right_arrow(raw):
            return
        if getattr(self, "lang_mode", False):
            self._handle_language_key(key)
            return
        if getattr(self, "help_mode", False):
            if key == 27:
                self.help_mode = False
            return
        if getattr(self, "finger_mode", False):
            self._handle_finger_key(key, raw)
            return
        if key == 27:
            if getattr(self, "help_mode", False):
                self.help_mode = False
            elif getattr(self, "lang_mode", False):
                self.lang_mode = False
            elif getattr(self, "finger_mode", False):
                self.finger_mode = False
                self.status_message = ""
            return
        if key in (ord("g"), ord("G")):
            self.inputs.toggle()
            if self.inputs.enabled:
                self.kill_banner = False
                self.status_message = ""
            self.reset_state(keep_message=self.kill_banner)
        elif key in (ord("p"), ord("P")):
            self.enter_mapping_menu()
        elif key in (ord("e"), ord("E")):
            self._close_other_ui(keep=("editor",))
            self.editor_mode = True
            self.editor_phase = "list"
            self.reset_state()
        elif key in (ord("s"), ord("S")):
            if ui_context.allows_overlay_toggle(ui_context.of(self)):
                self.toggle_hud_overlay()
        elif key in (ord("c"), ord("C")):
            if ui_context.allows_calibration(ui_context.of(self)):
                self.enter_calibration()
        elif key in (ord("h"), ord("H")):
            self._close_other_ui(keep=("help",))
            self.help_mode = True
            self.status_message = ""
        elif key in (ord("l"), ord("L")):
            self._close_other_ui(keep=("lang",))
            self.lang_mode = True
            self.finger_mode = False
            self.status_message = t("lang.choose")
        elif key in (ord("i"), ord("I")):
            self.enter_input_panel()
        elif key in (ord("f"), ord("F")):
            self.finger_mode = True
            self.status_message = t("status.finger_mode")

    def _handle_editor_click(self, action, *payload):
        rows = self._editor_rows()
        if action == "side":
            self.editor_side = payload[0] if payload else "LEFT"
            self.editor_index = 0
            self.editor_phase = "list"
        elif action == "row" and payload:
            index = int(payload[0])
            if 0 <= index < len(rows):
                self.editor_index = index
                self.editor_phase = "detail"
        elif action == "back":
            if self.editor_phase in ("detail", "samples", "confirm"):
                self.editor_phase = "list"
            else:
                self.editor_mode = False
        elif action == "edit_key" and rows:
            self._editor_open_key_settings(rows[self.editor_index])
        elif action == "record" and rows:
            self._editor_record_samples(rows[self.editor_index], APPEND)
        elif action == "replace" and rows:
            self._editor_record_samples(rows[self.editor_index], REPLACE)
        elif action == "test":
            self.handle_editor_key(ord("t"))
        elif action == "toggle" and rows:
            self.handle_editor_key(ord("o"))
        elif action == "delete":
            if self.editor_phase == "confirm":
                self.handle_editor_key(ord("y"))
            else:
                self.editor_phase = "confirm"
        elif action == "delete_no":
            self.editor_phase = "list"
        elif action == "samples":
            self.editor_phase = "samples"
        elif action == "outliers":
            self.handle_editor_key(ord("x"))

    def _editor_open_key_settings(self, row):
        if not row:
            return
        self.editor_mode = False
        self.map_mode = True
        self.map_side = row.get("hand") or self.editor_side
        names = self.map_gesture_names()
        gid = row.get("id")
        self.map_index = names.index(gid) if gid in names else 0
        self.load_edit_buffers()
        self.map_phase = "combo" if len(self.edit_inputs) > 1 else "simple"

    def _editor_record_samples(self, row, mode):
        if not row or row.get("kind") != "STATIC":
            return
        self.calib_side = row.get("hand") or self.editor_side
        ids = self.engine.editable_ids(self.calib_side)
        gid = row.get("id")
        self.calib_index = ids.index(gid) if gid in ids else 0
        self.editor_mode = False
        self.enter_calibration()
        self.calib_side = row.get("hand") or self.calib_side
        self.calib_index = ids.index(gid) if gid in ids else 0
        self._begin_calibration_capture(mode)

    def _handle_map_click(self, action, *payload):
        if action == "hand":
            self.handle_mapping_key(ord("1") if payload and payload[0] == "LEFT" else ord("2"))
        elif action == "back":
            self.handle_mapping_key(27)
        elif action == "row" and payload:
            names = self.map_gesture_names()
            index = int(payload[0])
            if 0 <= index < len(names):
                self.map_index = index
                self.load_edit_buffers()
                self.map_phase = "combo" if len(self.edit_inputs) > 1 else "simple"
        elif action == "type":
            chosen = payload[0] if payload else HOLD
            if chosen == HOLD:
                self.handle_mapping_key(ord("1"))
            elif chosen == PRESS:
                self.handle_mapping_key(ord("2"))
            elif chosen == COMBINATION:
                self.handle_mapping_key(ord("3"))
            elif chosen == MACRO:
                self.map_phase = "macro"
                self.print_macro_editor()
        elif action == "simple":
            self.handle_mapping_key(ord("1"))
        elif action == "combo":
            self.handle_mapping_key(ord("2"))
        elif action == "macro":
            self.handle_mapping_key(ord("3"))
        elif action == "assign":
            self._begin_map_capture()
        elif action == "save":
            if self.map_phase in ("simple", "combo", "macro"):
                self.handle_mapping_key(ord(" "))
            else:
                self.handle_mapping_key(ord("s"))
        elif action == "cancel":
            self.handle_mapping_key(27)

    def _handle_sys_click(self, action, *payload):
        rows = self._sys_rows()
        if action == "row" and payload:
            index = int(payload[0])
            if 0 <= index < len(rows):
                self.sys_index = index
                row = rows[index]
                if row.get("kind") in ("pinch", "tilt", "finger"):
                    self._sys_begin_edit(row)
        elif action == "toggle" and payload:
            index = int(payload[0])
            if 0 <= index < len(rows):
                self.sys_index = index
                self._sys_toggle_row(rows[index])
        elif action == "edit" and payload:
            index = int(payload[0])
            if 0 <= index < len(rows):
                self.sys_index = index
                row = rows[index]
                if row.get("kind") in ("pinch", "tilt", "finger"):
                    self._sys_begin_edit(row)
        elif action == "type":
            chosen = payload[0] if payload else PRESS
            if chosen == PRESS:
                self.handle_system_key(ord("1"))
            elif chosen == HOLD:
                self.handle_system_key(ord("2"))
            elif chosen == COMBINATION:
                self.handle_system_key(ord("3"))
            elif chosen == MACRO:
                self.handle_system_key(ord("4"))
        elif action == "assign":
            self._sys_begin_capture()
        elif action == "save":
            self.handle_system_key(ord("s"))
        elif action == "back":
            self.handle_system_key(27)
        elif action == "test":
            self.handle_system_key(ord("t"))
        elif action == "finger":
            step = int(payload[0]) if payload else 1
            self._sys_cycle_finger(step)

    def _handle_calib_click(self, action, *payload):
        if action == "hand":
            self.handle_calibration_key(ord("1") if payload and payload[0] == "LEFT" else ord("2"))
        elif action == "side":
            self.calib_side = payload[0] if payload else "LEFT"
        elif action == "start":
            if self.calib_phase == "create":
                self._start_new_recorded_gesture(REPLACE)
            elif self.calib_phase == "menu":
                self.handle_calibration_key(13)
            elif self.calib_phase == "mode":
                self.handle_calibration_key(ord("1"))
        elif action == "replace":
            self.handle_calibration_key(ord("1"))
        elif action == "append":
            if self.calib_phase == "create":
                self._start_new_recorded_gesture(APPEND)
            else:
                self.handle_calibration_key(ord("2"))
        elif action == "back":
            self.handle_calibration_key(27)
        elif action == "next":
            self.handle_calibration_key(ord(">"))
        elif action == "prev":
            self.handle_calibration_key(ord("<"))
        elif action == "new":
            self.handle_calibration_key(ord("a"))

    def _find_recorded_gesture(self, side, label):
        wanted = clean_gesture_name(label)
        if not wanted:
            return None
        for gid in self.engine.editable_ids(side):
            if name_of(self.engine.database["gestures"][side].get(gid)) == wanted:
                return gid
        return None

    def _start_new_recorded_gesture(self, mode=REPLACE):
        label = clean_gesture_name(self.calib_name_buffer)
        created = None
        if mode == APPEND:
            created = self._find_recorded_gesture(self.calib_side, label)
        if created is None:
            created = self.engine.allocate_gesture(self.calib_side)
            if not created:
                self.status_message = t("status.copy_fail")
                return
            if label:
                self.engine.rename_gesture(self.calib_side, created, label)
        ids = self._calib_ids()
        self.calib_index = ids.index(created) if created in ids else 0
        self.calib_phase = "mode"
        self.status_message = t("status.calib_new", id=created)
        self._begin_calibration_capture(mode)

    def _begin_calibration_capture(self, mode):
        ids = self._calib_ids()
        if not ids:
            return False
        gesture_name = ids[self.calib_index % len(ids)]
        if not self.engine.start_calibration(gesture_name, self.calib_side, mode):
            return False
        delay = getattr(self, "countdown_seconds", 3)
        if delay <= 0:
            self.calib_phase = "capture"
        else:
            self.calib_phase = "countdown"
            self.calib_end_time = time.monotonic() + delay
        self.status_message = t("status.calib_capture", side=self.calib_side, gesture=gesture_name)
        return True

    def _recorded_rows(self):
        side = getattr(self, "editor_side", None) or getattr(self, "settings_side", "LEFT")
        return [
            row for row in editor.gesture_rows(self.engine, self.mapping, side)
            if row.get("kind") == "STATIC"
        ]

    def _editor_rows(self):
        return self._recorded_rows()

    def _delete_recorded_gesture(self, side, gid):
        """Drop a learned gesture. System IDs (PINCH, TILT, FINGER) are refused."""
        if gesture_number(gid) is None:
            return False
        if getattr(self, "gestures_file", None) and getattr(self, "mapping_file", None):
            editor.backup_config(self.gestures_file, self.mapping_file)
        if not self.engine.delete_gesture(side, gid):
            return False
        recognizer = getattr(self, "recognition", None)
        if recognizer is not None:
            recognizer.refresh_models()
        self.mapping.disable(side, gid)
        current = getattr(self, "curr_gestures", None)
        if isinstance(current, dict) and current.get(side) == gid:
            current[side] = UNKNOWN
            self.engine.reset_hand(side)
        self.status_message = t("status.deleted", id=gid)
        return True

    def draw_editor(self, frame):
        height, width = frame.shape[:2]
        self._reset_hits()
        cv2.rectangle(frame, (0, 0), (width, height), (18, 18, 18), -1)
        put(frame, t("editor.title", side=self.editor_side), (12, 32), 0.7, (0, 220, 255), 2)
        self._draw_button(frame, 12, 44, 110, 28, t("ui.left"), "editor", "side", "LEFT",
                          fill=(40, 80, 40) if self.editor_side == "LEFT" else (40, 40, 40))
        self._draw_button(frame, 130, 44, 110, 28, t("ui.right"), "editor", "side", "RIGHT",
                          fill=(40, 80, 40) if self.editor_side == "RIGHT" else (40, 40, 40))
        self._draw_button(frame, width - 120, 12, 108, 28, t("ui.back"), "editor", "back")
        rows = self._editor_rows()
        if self.editor_phase == "confirm":
            put(frame, t("editor.delete_confirm"), (12, 120), 0.6, (0, 80, 255), 2)
            self._draw_button(frame, 12, 160, 100, 30, t("ui.yes"), "editor", "delete", fill=(20, 20, 140))
            self._draw_button(frame, 124, 160, 100, 30, t("ui.no"), "editor", "delete_no")
            return
        if self.editor_phase == "samples":
            row = rows[self.editor_index] if rows else None
            if row:
                outliers = editor.outlier_indices(self.engine.database["gestures"][self.editor_side].get(row["id"], []))
                put(frame, t("editor.samples", id=row["id"], samples=row["samples"], outliers=len(outliers)), (12, 100), 0.55, (255, 255, 255), 1)
                put(frame, t("editor.samples_help"), (12, 150), 0.5, (180, 180, 180), 1)
                self._draw_button(frame, 12, 180, 180, 30, t("ui.delete"), "editor", "outliers", fill=(20, 20, 140))
            return
        if self.editor_phase == "detail":
            row = rows[self.editor_index] if rows else None
            if not row:
                self.editor_phase = "list"
            else:
                entry = self.mapping.get(self.editor_side, row["id"]) or {}
                put(frame, t("editor.detail_title", id=row["id"]), (12, 96), 0.65, (0, 220, 255), 2)
                shown = row["name"] if row.get("name") and row["name"] != row["id"] else "—"
                put(frame, f"{t('ui.name')} : {shown}", (12, 130), 0.5, (255, 255, 255), 1)
                put(frame, f"{t('editor.hand_label')} : {row.get('hand') or self.editor_side}", (12, 156), 0.5, (255, 255, 255), 1)
                put(frame, f"{t('editor.action_label')} : {describe(entry)} — {action_label(entry.get('type') or row['action'])}", (12, 182), 0.5, (255, 255, 255), 1)
                buttons = (
                    (t("ui.edit_key"), "edit_key"),
                    (t("ui.record_samples"), "replace"),
                    (t("ui.add_samples"), "record"),
                    (t("ui.test"), "test"),
                    (t("ui.disable") if row["enabled"] else t("ui.enable"), "toggle"),
                    (t("ui.delete"), "delete"),
                )
                y = 214
                for label, action in buttons:
                    fill = (20, 20, 140) if action == "delete" else (52, 50, 48)
                    self._draw_button(frame, 12, y, 240, 30, label, "editor", action, fill=fill)
                    y += 36
                return
        panel_w = 220
        split = width - panel_w - 12 if width >= 560 else width
        list_w = split - 24
        put(frame, t("mapping.user_gestures").upper(), (12, 90), 0.45, (0, 220, 255), 1, list_w)
        visible = rows[:10]
        for index, row in enumerate(visible):
            entry = self.mapping.get(self.editor_side, row["id"]) or {}
            state = t("ui.on") if row["enabled"] else t("ui.off")
            top = 118 + index * 36
            selected = index == self.editor_index
            color = (0, 255, 0) if selected else (210, 210, 210)
            put(frame, editor.format_gesture_label(row["id"], row.get("name")), (12, top), 0.48, color, 1, list_w)
            put(frame, f"{row.get('hand') or self.editor_side}   {describe(entry)} — {action_label(entry.get('type') or row['action'])}   {state}", (12, top + 16), 0.36, (180, 180, 180), 1, list_w)
            self._add_hit(8, top - 16, split - 8, top + 20, "editor", "row", index)
        if split < width:
            self._draw_preset_panel(frame, split, 90, panel_w)
        put(frame, t("editor.help"), (8, height - 16), 0.36, (180, 180, 180), 1)

    def _draw_preset_panel(self, frame, x, y, panel_w):
        """Read-only list of the predefined gestures the engine really runs for this hand.
        White = enabled, gray = disabled. Never mixed with Gxx rows. Recognition is
        paused on this screen, so live detection is shown by the status block and OSD."""
        put(frame, t("mapping.preset_gestures").upper(), (x, y), 0.45, (0, 220, 255), 1, panel_w)
        y += 28
        for row in preset_catalog(self, self.editor_side):
            color = (235, 235, 235) if row["enabled"] else (120, 120, 120)
            put(frame, row["label"], (x, y), 0.42, color, 1, panel_w - 50)
            put(frame, t("ui.on") if row["enabled"] else t("ui.off"), (x + panel_w - 40, y), 0.38, color, 1, 40)
            y += 22
        put(frame, t("editor.presets_note"), (x, y + 8), 0.34, (150, 150, 150), 1, panel_w)

    def handle_editor_key(self, key, raw=0):
        rows = self._editor_rows()
        if self.editor_phase == "confirm":
            if key in (ord("y"), ord("Y")):
                row = rows[self.editor_index] if rows else None
                if row:
                    self._delete_recorded_gesture(self.editor_side, row["id"])
            self.editor_phase = "list"
            return
        if self.editor_phase == "samples":
            row = rows[self.editor_index] if rows else None
            if key in (ord("x"), ord("X")) and row:
                editor.backup_config(self.gestures_file, self.mapping_file)
                bad = editor.outlier_indices(self.engine.database["gestures"][self.editor_side].get(row["id"], []))
                editor.remove_samples(self.engine, self.editor_side, row["id"], bad)
                self.status_message = t("status.outliers", count=len(bad))
            if key == 27 or key in (ord("k"), ord("K"), ord("x"), ord("X")):
                self.editor_phase = "list"
            return
        if key == 27:
            if self.editor_phase == "detail":
                self.editor_phase = "list"
            else:
                self.editor_mode = False
            return
        if key in (13, 10) and rows:
            self.editor_phase = "detail"
            return
        if key == ord("1"):
            self.editor_side = "LEFT"
            self.editor_index = 0
        elif key == ord("2"):
            self.editor_side = "RIGHT"
            self.editor_index = 0
        elif is_right_arrow(raw) and rows:
            self.editor_index = (self.editor_index + 1) % len(rows)
        elif is_left_arrow(raw) and rows:
            self.editor_index = (self.editor_index - 1) % len(rows)
        elif key in (ord("c"), ord("C")):
            created = self.engine.allocate_gesture(self.editor_side)
            if created:
                self.calib_side = self.editor_side
                ids = self.engine.editable_ids(self.editor_side)
                self.calib_index = ids.index(created) if created in ids else 0
                self.editor_mode = False
                self.enter_calibration()
                self.calib_phase = "menu"
                self.calib_side = self.editor_side
        elif key in (ord("u"), ord("U")) and rows:
            new_id = editor.duplicate_gesture(self.engine, self.mapping, self.editor_side, rows[self.editor_index]["id"])
            self.status_message = t("status.copy_ok", id=new_id) if new_id else t("status.copy_fail")
        elif key in (ord("o"), ord("O")) and rows:
            editor.toggle_enabled(self.mapping, self.editor_side, rows[self.editor_index]["id"])
        elif key in (ord("a"), ord("A")) and rows and rows[self.editor_index]["kind"] == "STATIC":
            self.calib_side = self.editor_side
            ids = self.engine.editable_ids(self.editor_side)
            gid = rows[self.editor_index]["id"]
            self.calib_index = ids.index(gid) if gid in ids else 0
            self.editor_mode = False
            self.enter_calibration()
            self.calib_phase = "mode"
        elif key in (ord("s"), ord("S")) and rows and rows[self.editor_index]["kind"] == "STATIC":
            self.editor_phase = "samples"
        elif key in (ord("x"), ord("X")) and rows and rows[self.editor_index]["kind"] == "STATIC":
            self.editor_phase = "confirm"
        elif key in (ord("t"), ord("T")):
            self.safe_mode = True
            self.inputs.disable()
            self.editor_mode = False
            self.status_message = t("status.safe_test")

    def _sys_rows(self):
        rows = []
        for side in SIDES:
            rows.append({"kind": "pinch", "side": side})
        for side in SIDES:
            for direction in sysg.DIRECTIONS:
                rows.append({"kind": "tilt", "side": side, "direction": direction})
        rows.append({"kind": "mode"})
        for side in SIDES:
            for finger in geo.FINGER_UP_FINGERS:
                rows.append({"kind": "finger", "side": side, "finger": finger})
        rows.append({"kind": "test"})
        return rows

    def _sys_row_label(self, row):
        kind = row.get("kind")
        side = row.get("side")
        if kind == "pinch":
            enabled = sysg.is_system_enabled(self.system_gestures, side, f"{side}_PINCH")
            gid = f"{side}_PINCH"
            entry = self.mapping.get(side, gid) or {}
            return f"PINCH {side}  [{t('sys.on') if enabled else t('sys.off')}]  {describe(entry)}"
        if kind == "tilt":
            direction = row.get("direction")
            gid = sysg.system_gesture_id(kind, side, direction)
            enabled = sysg.is_system_enabled(self.system_gestures, side, gid)
            entry = self.mapping.get(side, gid) or {}
            return f"{kind.upper()} {side} {direction.upper()}  [{t('sys.on') if enabled else t('sys.off')}]  {describe(entry)}"
        if kind == "mode":
            mode = (self.system_gestures.get("finger_up") or {}).get("mode", "single")
            return t("sys.finger_mode", mode=t("sys.mode_single") if mode == "single" else t("sys.mode_multi"))
        if kind == "finger":
            finger = row.get("finger")
            gid = sysg.system_gesture_id("finger_up", side, finger=finger)
            enabled = sysg.is_system_enabled(self.system_gestures, side, gid)
            entry = self.mapping.get(side, gid) or {}
            finger_label = t(f"sys.{finger.lower()}") if finger else finger
            return f"{finger_label} UP {side}  [{t('sys.on') if enabled else t('sys.off')}]  {describe(entry)}"
        if kind == "test":
            return t("sys.test_enter")
        return ""

    def _sys_toggle_row(self, row):
        kind = row.get("kind")
        if kind == "mode":
            current = (self.system_gestures.get("finger_up") or {}).get("mode", "single")
            self.system_gestures["finger_up"]["mode"] = "multi" if current == "single" else "single"
            self.settings["finger_up_mode"] = self.system_gestures["finger_up"]["mode"]
            self._save_system_gestures()
            self.geometry = self._build_geometry()
            return
        if kind == "test":
            self.sys_mode = False
            self.system_test_mode = True
            self.inputs.disable()
            self.status_message = t("sys.test_banner")
            return
        if kind == "finger":
            enabled = not sysg.is_system_enabled(
                self.system_gestures, row["side"],
                sysg.system_gesture_id("finger_up", row["side"], finger=row["finger"]),
            )
            self.system_gestures = sysg.set_system_enabled(
                self.system_gestures, "finger_up", row["side"], enabled=enabled, finger=row["finger"],
            )
        else:
            gid = sysg.system_gesture_id(kind, row.get("side"), row.get("direction"))
            enabled = not sysg.is_system_enabled(self.system_gestures, row.get("side"), gid)
            self.system_gestures = sysg.set_system_enabled(
                self.system_gestures, kind, row.get("side"), row.get("direction"), enabled,
            )
        self._save_system_gestures()
        builder = getattr(self, "_build_geometry", None)
        if callable(builder):
            self.geometry = builder()

    def _sys_row_ids(self, row):
        kind = row.get("kind")
        side = row.get("side")
        if kind == "finger":
            return side, sysg.system_gesture_id("finger_up", side, finger=row.get("finger"))
        return side, sysg.system_gesture_id(kind, side, row.get("direction"))

    def _sys_row_title(self, row):
        kind = row.get("kind")
        if kind == "pinch":
            return f"PINCH {str(row.get('side') or '').upper()}"
        if kind == "tilt":
            return f"TILT {str(row.get('direction') or '').upper()}"
        if kind == "finger":
            finger = str(row.get("finger") or "INDEX").upper()
            return f"{finger} UP"
        return ""

    def _sys_allowed_types(self, row):
        return (PRESS, HOLD, COMBINATION, MACRO)

    def _sys_begin_edit(self, row):
        side, gid = self._sys_row_ids(row)
        entry = self.mapping.get(side, gid) if gid else {}
        allowed = self._sys_allowed_types(row)
        entry_type = entry.get("type") if entry and entry.get("type") in allowed else allowed[0]
        inputs = list(entry.get("inputs") or ["A"])
        self.sys_edit = {"type": entry_type, "inputs": inputs or ["A"]}
        self.sys_combo_capture = False
        self.sys_phase = "assign"
        self.status_message = t("sys.assign_title")

    def _sys_cycle_finger(self, step=1):
        rows = self._sys_rows()
        row = rows[self.sys_index] if rows else {}
        if row.get("kind") != "finger":
            return
        fingers = list(geo.FINGER_UP_FINGERS)
        current = str(row.get("finger") or "INDEX").upper()
        if current not in fingers:
            current = "INDEX"
        new = fingers[(fingers.index(current) + int(step or 1)) % len(fingers)]
        if new == current:
            return
        side = row.get("side")
        old_gid = sysg.system_gesture_id("finger_up", side, finger=current)
        new_gid = sysg.system_gesture_id("finger_up", side, finger=new)
        mapped = self.mapping.get(side, old_gid)
        if mapped and mapped.get("type") not in ("", "NONE", None):
            self.mapping._store(side, new_gid, dict(mapped))
            self.mapping.disable(side, old_gid)
        enabled = sysg.is_system_enabled(self.system_gestures, side, old_gid)
        self.system_gestures = sysg.remove_finger_binding(self.system_gestures, side, current)
        self.system_gestures = sysg.set_system_enabled(
            self.system_gestures, "finger_up", side, enabled=enabled, finger=new,
        )
        self._save_system_gestures()
        self.geometry = self._build_geometry()
        for index, item in enumerate(self._sys_rows()):
            if item.get("kind") == "finger" and item.get("side") == side and item.get("finger") == new:
                self.sys_index = index
                break
        self._sys_begin_edit(self._sys_rows()[self.sys_index])
        self.status_message = f"{t('sys.finger')} : {t(f'sys.{new.lower()}')}"

    def _sys_assign_row(self, row, entry_type, action):
        kind = row.get("kind")
        if kind not in ("pinch", "tilt", "finger"):
            return
        side = row.get("side")
        if kind == "finger":
            gid = sysg.system_gesture_id("finger_up", side, finger=row.get("finger"))
            self.system_gestures = sysg.set_system_enabled(
                self.system_gestures, "finger_up", side, enabled=True, finger=row.get("finger"),
            )
        else:
            gid = sysg.system_gesture_id(kind, side, row.get("direction"))
            self.system_gestures = sysg.set_system_enabled(
                self.system_gestures, kind, side, row.get("direction"), True,
            )
        if gid:
            keys = action if isinstance(action, list) else [action]
            self.mapping.set_command(side, gid, entry_type, keys, 0 if entry_type == PRESS else None)
        self._save_system_gestures()

    def _sys_save_edit(self, row):
        edit = self.sys_edit if isinstance(self.sys_edit, dict) else {}
        entry_type = edit.get("type") or PRESS
        if entry_type not in self._sys_allowed_types(row):
            entry_type = self._sys_allowed_types(row)[0]
        inputs = list(edit.get("inputs") or ["A"])
        if entry_type == MACRO:
            self._sys_open_macro_editor(row)
            return
        if entry_type != COMBINATION:
            inputs = inputs[:1] or ["A"]
        self._sys_assign_row(row, entry_type, inputs)
        self.sys_phase = "list"
        self.status_message = f"{self._sys_row_title(row)} -> {'+'.join(inputs)} {action_label(entry_type)}"

    def _sys_open_macro_editor(self, row):
        side, gid = self._sys_row_ids(row)
        if not gid:
            return
        if row.get("kind") == "finger":
            self.system_gestures = sysg.set_system_enabled(
                self.system_gestures, "finger_up", side, enabled=True, finger=row.get("finger"),
            )
        else:
            self.system_gestures = sysg.set_system_enabled(
                self.system_gestures, row.get("kind"), side, row.get("direction"), True,
            )
        self._save_system_gestures()
        self.sys_mode = False
        self.sys_phase = "list"
        self.map_mode = True
        self.map_side = side
        names = self.map_gesture_names()
        self.map_index = names.index(gid) if gid in names else 0
        self.load_edit_buffers()
        self.map_phase = "macro"
        self.print_macro_editor()

    def draw_system_menu(self, frame):
        height, width = frame.shape[:2]
        self._reset_hits()
        mx = max(12, int(width * 0.02))
        cv2.rectangle(frame, (0, 0), (width, height), (18, 18, 18), -1)
        put(frame, t("sys.title"), (mx, 32), 0.7, (0, 220, 255), 2, width - 2 * mx)
        self._draw_button(frame, width - 120, 8, 108, 28, t("ui.back"), "sys", "back")
        if self.sys_phase in ("assign", "capture"):
            self._draw_system_assign(frame)
            return
        rows = self._sys_rows()
        if not rows:
            return
        self.sys_index = min(self.sys_index, len(rows) - 1)
        first = max(0, min(self.sys_index - 3, len(rows) - 8))
        for offset, row in enumerate(rows[first:first + 8]):
            index = first + offset
            selected = index == self.sys_index
            y = 70 + offset * 42
            color = (0, 255, 0) if selected else (210, 210, 210)
            put(frame, self._sys_row_label(row), (mx, y), 0.4, color, 1, width - 220)
            self._add_hit(mx, y - 16, width - 220, y + 8, "sys", "row", index)
            if row.get("kind") in ("pinch", "tilt", "finger"):
                side, gid = self._sys_row_ids(row)
                enabled = sysg.is_system_enabled(self.system_gestures, side, gid)
                self._draw_button(
                    frame, width - 210, y - 18, 56, 26,
                    t("ui.on") if enabled else t("ui.off"),
                    "sys", "toggle", index,
                    fill=(30, 110, 40) if enabled else (50, 40, 40),
                )
                self._draw_button(frame, width - 146, y - 18, 90, 26, t("ui.modify"), "sys", "edit", index)
        self._draw_button(frame, mx, height - 48, 160, 28, t("ui.test"), "sys", "test")
        put(frame, t("sys.help"), (mx, height - 14), 0.32, (180, 180, 180), 1, width - 2 * mx)

    def _draw_system_assign(self, frame):
        height, width = frame.shape[:2]
        mx = max(12, int(width * 0.02))
        rows = self._sys_rows()
        row = rows[self.sys_index] if rows else {}
        edit = self.sys_edit if isinstance(self.sys_edit, dict) else {}
        allowed = self._sys_allowed_types(row)
        entry_type = edit.get("type") if edit.get("type") in allowed else allowed[0]
        keys = "+".join(edit.get("inputs") or ["A"])
        if self.sys_phase == "capture":
            put(frame, t("sys.capture_title"), (mx, 100), 0.65, (0, 220, 255), 2, width - 2 * mx)
            put(frame, f"{t('hud.cmd_system')}: {self._sys_row_title(row)}", (mx, 140), 0.5, (255, 255, 255), 1, width - 2 * mx)
            put(frame, f"{t('sys.left') if row.get('side') == 'LEFT' else t('sys.right')}: {row.get('side')}", (mx, 168), 0.5, (255, 255, 255), 1, width - 2 * mx)
            put(frame, f"{t('sys.action_type')}: {action_label(entry_type)}", (mx, 196), 0.5, (255, 255, 255), 1, width - 2 * mx)
            put(frame, t("ui.press_to_assign"), (mx, 240), 0.55, (0, 220, 255), 1, width - 2 * mx)
            put(frame, t("ui.esc_cancel"), (mx, 280), 0.45, (180, 180, 180), 1, width - 2 * mx)
            self._draw_button(frame, mx, 310, 140, 30, t("ui.cancel"), "sys_cancel")
            if self.status_message:
                put(frame, self.status_message, (mx, 360), 0.5, (0, 220, 255), 1, width - 2 * mx)
            return
        put(frame, t("sys.assign_title"), (mx, 96), 0.65, (0, 220, 255), 2, width - 2 * mx)
        put(frame, f"{self._sys_row_title(row)}   {row.get('side')}", (mx, 132), 0.5, (255, 255, 255), 1, width - 2 * mx)
        y = 168
        if row.get("kind") == "finger":
            put(frame, t("sys.finger"), (mx, y), 0.5, (0, 220, 255), 1, width - 2 * mx)
            finger = str(row.get("finger") or "INDEX").upper()
            self._draw_button(frame, mx + 90, y - 18, 140, 28, t(f"sys.{finger.lower()}"), "sys", "finger", 1, fill=(40, 80, 40))
            self._draw_button(frame, mx + 238, y - 18, 36, 28, "<", "sys", "finger", -1)
            self._draw_button(frame, mx + 280, y - 18, 36, 28, ">", "sys", "finger", 1)
            y = 204
        put(frame, f"{t('sys.key_label')} : [{keys}]", (mx, y), 0.5, (255, 255, 255), 1, width - 2 * mx)
        put(frame, t("sys.type_label"), (mx, y + 32), 0.5, (0, 220, 255), 1, width - 2 * mx)
        x = mx
        for option in allowed:
            fill = (30, 110, 40) if option == entry_type else (52, 50, 48)
            self._draw_button(frame, x, y + 46, 130, 28, action_label(option), "sys", "type", option, fill=fill)
            x += 140
        by = y + 90
        self._draw_button(frame, mx, by, 160, 32, t("ui.assign"), "sys", "assign", fill=(20, 90, 140))
        self._draw_button(frame, mx + 172, by, 160, 32, t("ui.save"), "sys", "save", fill=(30, 110, 40))
        self._draw_button(frame, mx + 344, by, 160, 32, t("ui.back"), "sys", "back")
        put(frame, t("sys.assign_help"), (mx, height - 18), 0.36, (180, 180, 180), 1, width - 2 * mx)
        if self.status_message:
            put(frame, self.status_message, (mx, 310), 0.42, (0, 220, 255), 1, width - 2 * mx)

    def draw_system_test(self, frame):
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width, 220), (18, 18, 18), -1)
        put(frame, t("sys.test_title"), (12, 28), 0.7, (0, 220, 255), 2)
        put(frame, t("sys.test_banner"), (12, 54), 0.45, (0, 80, 255), 1)
        y = 82
        for side in SIDES:
            state = self._hud_system_state(side)
            put(frame, f"{side}", (12, y), 0.5, SIDE_TEXT_COLORS[side], 1)
            put(frame, t("debug.finger", value=state["finger"]), (90, y), 0.42, (255, 255, 255), 1)
            y += 20
            put(frame, f"PINCH {state['pinch']}   TILT {state['tilt']}", (90, y), 0.42, (210, 210, 210), 1)
            y += 28
        put(frame, t("sys.test_help"), (12, height - 18), 0.4, (180, 180, 180), 1)

    def _sys_begin_capture(self):
        """Assign button on the system gesture screen: wait for the next key or combination."""
        if self.sys_phase != "assign":
            return False
        edit = self.sys_edit if isinstance(self.sys_edit, dict) else {}
        self.sys_combo_capture = edit.get("type") == COMBINATION
        self.sys_phase = "capture"
        if self.sys_combo_capture:
            self.inputs.begin_combo_capture()
            self.status_message = t("status.hold_modifiers")
        else:
            self.inputs.begin_key_capture()
            self.status_message = t("sys.press_key")
        return True

    def handle_system_key(self, key, raw=0):
        rows = self._sys_rows()
        row = rows[self.sys_index] if rows else None
        if self.sys_phase == "capture":
            if key == 27:
                self.inputs.end_key_capture()
                self.sys_combo_capture = False
                self.sys_phase = "assign"
                self.status_message = t("status.capture_cancel")
            return
        if self.sys_phase == "assign":
            edit = self.sys_edit if isinstance(self.sys_edit, dict) else {}
            allowed = self._sys_allowed_types(row)
            if key == 27:
                self.sys_phase = "list"
                self.status_message = t("sys.cancel")
            elif key == ord("1") and PRESS in allowed:
                edit["type"] = PRESS
            elif key == ord("2") and HOLD in allowed:
                edit["type"] = HOLD
            elif key == ord("3") and COMBINATION in allowed:
                edit["type"] = COMBINATION
            elif key == ord("4") and MACRO in allowed:
                edit["type"] = MACRO
                self.sys_edit = edit
                if row:
                    self._sys_open_macro_editor(row)
                return
            elif key in (ord("s"), ord("S"), 13, 10) and row:
                self._sys_save_edit(row)
            elif is_right_arrow(raw) and row and row.get("kind") == "finger":
                self._sys_cycle_finger(1)
            elif is_left_arrow(raw) and row and row.get("kind") == "finger":
                self._sys_cycle_finger(-1)
            self.sys_edit = edit
            return
        if key == 27:
            self.sys_mode = False
            self.reset_state()
            return
        if not rows:
            return
        if is_right_arrow(raw):
            self.sys_index = (self.sys_index + 1) % len(rows)
        elif is_left_arrow(raw):
            self.sys_index = (self.sys_index - 1) % len(rows)
        elif key in (13, 10) or key in (ord("o"), ord("O")):
            self._sys_toggle_row(rows[self.sys_index])
        elif key in (ord("a"), ord("A"), ord("e"), ord("E")):
            if row and row.get("kind") in ("pinch", "tilt", "finger"):
                self._sys_begin_edit(row)
        elif key in (ord("x"), ord("X")):
            if row and row.get("kind") == "finger":
                self.system_gestures = sysg.remove_finger_binding(
                    self.system_gestures, row.get("side"), row.get("finger"),
                )
                side, gid = self._sys_row_ids(row)
                if gid:
                    self.mapping.disable(side, gid)
                self._save_system_gestures()
            elif row and row.get("kind") in ("pinch", "tilt"):
                self.system_gestures = sysg.set_system_enabled(
                    self.system_gestures, row.get("kind"), row.get("side"), row.get("direction"), False,
                )
                self._save_system_gestures()
        elif key in (ord("t"), ord("T")):
            self.sys_mode = False
            self.system_test_mode = True
            self.inputs.disable()
            self.status_message = t("sys.test_banner")

    # ---- Mapping (souris : Parametres de geste / Configurer) ----------------
    def enter_mapping_menu(self):
        self.help_mode = False
        self.lang_mode = False
        self.ensure_preview_visible()
        self.map_mode = True
        self.map_phase = "hand"
        self.reset_state()
        print("\n================================")
        print(f"            {t('mapping.title')}")
        print("================================")
        print(t("mapping.choose_left"))
        print(t("mapping.choose_right"))
        print(t("mapping.quit"))
        self.status_message = t("status.mapping_choose_hand")

    def exit_mapping_menu(self):
        if self.test_active:
            self.stop_macro_test(t("status.test_stopped"))
        self.map_mode = False
        self.map_phase = "hand"
        self.reset_state()
        self.status_message = t("status.mapping_closed")

    def map_gesture_names(self):
        """Recorded gestures that still exist, then geometry IDs. No empty Gxx."""
        learned = self.recorded_gesture_ids(self.map_side)
        specials = [
            name for name in side_gesture_names(self.map_side)
            if name not in GESTURE_NAMES and name not in learned and self._listable_preset(name)
        ]
        return learned + specials

    @staticmethod
    def _listable_preset(name):
        """TILT / FINGER UP variants plus PINCH. Never empty Gxx or neutrals."""
        core = str(name or "")
        if core.startswith("LEFT_"):
            core = core[5:]
        elif core.startswith("RIGHT_"):
            core = core[6:]
        if core.endswith("NEUTRAL"):
            return False
        return core == "PINCH" or core.startswith("TILT_") or core.startswith("FINGER_")

    def current_map_gesture(self):
        names = self.map_gesture_names()
        if not names:
            return ""
        return names[min(max(self.map_index, 0), len(names) - 1)]

    def load_edit_buffers(self):
        gesture = self.current_map_gesture()
        entry = self.mapping.get(self.map_side, gesture)
        self.edit_type = entry["type"] if entry["type"] in (HOLD, PRESS, COMBINATION) else HOLD
        self.edit_inputs = list(entry.get("inputs") or ["W"])
        if self.edit_type == COMBINATION:
            self.edit_modes = list(entry.get("modes") or [])
            while len(self.edit_modes) < len(self.edit_inputs):
                self.edit_modes.append(HOLD)
            self.edit_modes = self.edit_modes[:len(self.edit_inputs)]
        else:
            self.edit_modes = [HOLD] * len(self.edit_inputs)
        self.edit_slot = 0
        cooldown = self.mapping.get_cooldown(self.map_side, gesture)
        self.edit_cooldown = COOLDOWN_CHOICES.index(cooldown) if cooldown in COOLDOWN_CHOICES else 3
        self.macro_steps = self.mapping.get_steps(self.map_side, gesture) or [
            {"action": STEP_PRESS, "inputs": ["SPACE"]}
        ]
        self.macro_index = 0
        speed = self.mapping.get_speed(self.map_side, gesture)
        self.macro_speed = SPEED_CHOICES.index(speed) if speed in SPEED_CHOICES else SPEED_CHOICES.index(1.0)
        self.step_backup = None

    def print_edit_screen(self):
        gesture = self.current_map_gesture()
        entry = self.mapping.get(self.map_side, gesture)
        print(f"\n--- {side_label(self.map_side)} / {gesture} ---")
        print(t("mapping.current_type", type=action_label(entry["type"])))
        print(t("mapping.current_commands", commands=describe(entry)))
        print(t("mapping.options"))
        print(t("mapping.opt_simple"))
        print(t("mapping.opt_combo"))
        print(t("mapping.opt_macro"))
        print(t("mapping.opt_cancel") + "\n")

    def print_macro_editor(self):
        gesture = self.current_map_gesture()
        print(f"\n{side_label(self.map_side)}")
        print(f"{gesture}")
        print(t("mapping.macro"))
        print("------------------------")
        for index, step in enumerate(self.macro_steps):
            prefix = ">" if index == self.macro_index else " "
            print(f"{prefix} {index + 1}. {describe_step(step)}")
        print("------------------------")
        print(t("mapping.preview", text=preview(self.macro_steps)))
        print(t("mapping.speed_label", speed=SPEED_CHOICES[self.macro_speed]))
        print(t("mapping.macro_keys"))
        print(t("mapping.macro_edit"))
        print(t("mapping.macro_save") + "\n")

    def draw_mapping_menu(self, frame):
        height, width = frame.shape[:2]
        self._reset_hits()
        cv2.rectangle(frame, (0, 0), (width, height), (18, 18, 18), -1)
        gesture = self.current_map_gesture()
        entry = self.mapping.get(self.map_side, gesture)
        title_color = (0, 255, 255)

        if self.map_phase == "hand":
            put(frame, t("ui.key_settings"), (12, 40), 0.9, title_color, 2)
            self._draw_button(frame, 12, 90, 220, 36, t("ui.left"), "map", "hand", "LEFT", fill=(40, 80, 40))
            self._draw_button(frame, 12, 136, 220, 36, t("ui.right"), "map", "hand", "RIGHT", fill=(40, 60, 100))
            self._draw_button(frame, 12, 190, 220, 32, t("ui.back"), "map", "back")
            return

        header = f"{side_label(self.map_side)}"
        put(frame, header, (12, 32), 0.7, SIDE_TEXT_COLORS[self.map_side], 2)

        if self.map_phase == "list":
            self._draw_button(frame, width - 120, 8, 108, 28, t("ui.back"), "map", "back")
            names = self.map_gesture_names()
            first = max(0, min(self.map_index - 4, len(names) - 9))
            y = 64
            last_kind = None
            col_action = min(108, max(90, width // 6))
            col_type = min(330, max(240, width // 2))
            col_state = min(500, max(360, width - 110))
            for row, name in enumerate(names[first:first + 9]):
                kind = "preset" if name in geo.SPECIAL_NAMES else "user"
                if kind != last_kind:
                    header = t("mapping.preset_gestures") if kind == "preset" else t("mapping.user_gestures")
                    put(frame, header, (12, y), 0.42, (0, 220, 255), 1)
                    y += 18
                    last_kind = kind
                current = self.mapping.get(self.map_side, name)
                selected = (first + row) == self.map_index
                if selected:
                    color = (0, 255, 0)
                else:
                    color = (150, 210, 255) if name in geo.SPECIAL_NAMES else (210, 210, 210)
                gid, action, type_s, state = mapping_row_parts(name, current)
                put(frame, gid, (12, y), 0.45, color, 1)
                put(frame, action, (col_action, y), 0.45, color, 1)
                put(frame, type_s, (col_type, y), 0.45, color, 1)
                put(frame, state, (col_state, y), 0.45, color, 1)
                self._add_hit(8, y - 16, width - 8, y + 8, "map", "row", first + row)
                y += 26
            current_name = names[self.map_index] if names else ""
            shown = current_name
            if current_name in geo.SPECIAL_NAMES:
                shown = preset_display_name(current_name)
            put(frame, t("mapping.nav_hint", id=shown), (12, height - 40), 0.42, (180, 180, 180), 1)
            put(frame, t("mapping.list_help"), (12, height - 18), 0.38, (180, 180, 180), 1)
            return

        if self.map_phase == "rename":
            put(frame, t("mapping.rename_title"), (12, 120), 0.7, (0, 220, 255), 2)
            put(frame, self.rename_buffer or "_", (12, 170), 0.8, (255, 255, 255), 2)
            put(frame, t("mapping.rename_help"), (12, 220), 0.5, (180, 180, 180), 1)
            return

        put(frame, gesture, (12, 76), 0.9, (255, 255, 255), 2)

        if self.map_phase == "edit":
            put(frame, t("mapping.current_type", type=action_label(entry["type"])), (12, 116), 0.6, (0, 220, 255), 1)
            put(frame, t("mapping.current_commands", commands=describe(entry)), (12, 146), 0.6, (255, 255, 255), 1)
            options = (
                (t("mapping.opt_simple"), "simple"),
                (t("mapping.opt_combo"), "combo"),
                (t("mapping.opt_macro"), "macro"),
                (t("ui.back"), "back"),
            )
            for row, (text, action) in enumerate(options):
                self._draw_button(frame, 12, 176 + row * 36, 280, 30, text, "map", action)
            return

        if self.map_phase == "simple":
            put(frame, t("ui.key_settings"), (12, 112), 0.6, title_color, 1)
            put(frame, f"{t('ui.action')} : {'+'.join(self.edit_inputs)}", (12, 142), 0.5, (255, 255, 255), 1)
            put(frame, t("mapping.type_hold") + "   " + t("mapping.type_press"), (12, 172), 0.45, (0, 220, 255), 1)
            x = 12
            for value, label in ((HOLD, "HOLD"), (PRESS, "PRESS")):
                fill = (30, 110, 40) if self.edit_type == value else (52, 50, 48)
                self._draw_button(frame, x, 184, 130, 28, label, "map", "type", value, fill=fill)
                x += 140
            self._draw_button(frame, 12, 226, 200, 32, t("ui.assign"), "map", "assign", fill=(20, 90, 140))
            self._draw_button(frame, 224, 226, 160, 32, t("ui.save"), "map", "save", fill=(30, 110, 40))
            self._draw_button(frame, 396, 226, 160, 32, t("ui.back"), "map", "back")
            if self.edit_type == PRESS:
                put(frame, t("mapping.cooldown", ms=COOLDOWN_CHOICES[self.edit_cooldown]),
                    (12, height - 100), 0.5, (0, 220, 255), 1)
            put(frame, t("mapping.result", keys="+".join(self.edit_inputs)), (12, height - 72), 0.5, (0, 220, 255), 1)
            put(frame, t("mapping.nav_hint", id=gesture), (12, height - 44), 0.42, (180, 180, 180), 1)
            put(frame, t("mapping.simple_help"), (12, height - 18), 0.4, (180, 180, 180), 1)
            return

        if self.map_phase == "combo":
            put(frame, t("ui.key_settings"), (12, 112), 0.6, title_color, 1)
            put(frame, t("mapping.combo"), (12, 142), 0.5, (0, 220, 255), 1)
            modes = list(getattr(self, "edit_modes", []) or [])
            while len(modes) < len(self.edit_inputs):
                modes.append(HOLD)
            for slot, action in enumerate(self.edit_inputs):
                selected = slot == self.edit_slot
                color = (0, 255, 0) if selected else (255, 255, 255)
                prefix = ">" if selected else " "
                mode = modes[slot] if slot < len(modes) else HOLD
                put(frame, f"{prefix} {action}  [{mode}]", (12, 176 + slot * 24), 0.5, color, 1)
            self._draw_button(frame, 12, height - 140, 200, 32, t("ui.assign"), "map", "assign", fill=(20, 90, 140))
            self._draw_button(frame, 224, height - 140, 160, 32, t("ui.save"), "map", "save", fill=(30, 110, 40))
            self._draw_button(frame, 396, height - 140, 160, 32, t("ui.back"), "map", "back")
            put(frame, t("mapping.nav_hint", id=gesture), (12, height - 44), 0.42, (180, 180, 180), 1)
            put(frame, t("mapping.combo_help"), (12, height - 18), 0.4, (180, 180, 180), 1)
            return

        if self.map_phase == "capture":
            put(frame, t("sys.capture_title"), (12, 140), 0.7, (0, 220, 255), 2)
            put(frame, t("ui.press_to_assign"), (12, 180), 0.55, (255, 255, 255), 1)
            put(frame, t("ui.esc_cancel"), (12, 214), 0.5, (180, 180, 180), 1)
            put(frame, self.status_message[:78], (12, 250), 0.5, (0, 220, 255), 1)
            self._draw_button(frame, 12, 280, 160, 32, t("ui.cancel"), "map_cancel")
            return

        if self.map_phase == "macro":
            put(frame, t("mapping.macro"), (12, 108), 0.6, title_color, 1)
            if self.test_active:
                put(frame, t("mapping.test_banner"), (150, 108), 0.55, (0, 0, 255), 2)
            first = max(0, min(self.macro_index - 3, len(self.macro_steps) - 8))
            for row, step in enumerate(self.macro_steps[first:first + 8]):
                index = first + row
                selected = index == self.macro_index
                color = (0, 255, 0) if selected else (255, 255, 255)
                prefix = ">" if selected else " "
                put(frame, f"{prefix} {index + 1}. {describe_step(step)}", (12, 140 + row * 24), 0.52, color, 1)
            put(frame, preview(self.macro_steps)[:78], (12, height - 92), 0.42, (0, 220, 255), 1)
            put(frame, t("mapping.speed", speed=SPEED_CHOICES[self.macro_speed]), (12, height - 68), 0.45, (0, 220, 255), 1)
            put(frame, t("mapping.macro_help1"), (12, height - 28), 0.4, (180, 180, 180), 1)
            return

        if self.map_phase == "step":
            step = self.macro_steps[self.macro_index]
            put(frame, t("mapping.edit_step", index=self.macro_index + 1), (12, 112), 0.6, title_color, 1)
            put(frame, describe_step(step), (12, 158), 0.8, (255, 255, 255), 2)
            put(frame, t("mapping.step_action", action=action_label(step["action"])), (12, 202), 0.55, (0, 220, 255), 1)
            if step["action"] in (STEP_WAIT,):
                put(frame, t("mapping.step_wait", duration=step["duration"], step=WAIT_STEP_MS), (12, 236), 0.55, (255, 255, 255), 1)
            elif step["action"] == STEP_REPEAT_BEGIN:
                put(frame, t("mapping.step_repeat", count=step["count"]), (12, 236), 0.55, (255, 255, 255), 1)
            elif step["action"] != STEP_REPEAT_END:
                for slot, action in enumerate(step["inputs"]):
                    selected = slot == self.edit_slot
                    color = (0, 255, 0) if selected else (255, 255, 255)
                    prefix = ">" if selected else " "
                    put(frame, f"{prefix} {action}", (12, 236 + slot * 26), 0.55, color, 1)
            put(frame, t("mapping.step_help1"), (12, height - 44), 0.45, (180, 180, 180), 1)
            put(frame, t("mapping.step_help2"), (12, height - 18), 0.45, (180, 180, 180), 1)

    def _cycle_action(self, action, step):
        index = ACTIONS.index(action) if action in ACTIONS else 0
        return ACTIONS[(index + step) % len(ACTIONS)]

    def _shift_map_index(self, step):
        names = self.map_gesture_names()
        if not names:
            return
        self.map_index = (self.map_index + step) % len(names)

    def _save_simple_or_combo(self):
        gesture = self.current_map_gesture()
        cooldown = COOLDOWN_CHOICES[self.edit_cooldown]
        if self.mapping.set_command(
            self.map_side, gesture, self.edit_type, self.edit_inputs, cooldown,
            getattr(self, "edit_modes", None) if self.edit_type == COMBINATION else None,
        ):
            entry = self.mapping.get(self.map_side, gesture)
            if gesture in geo.SPECIAL_NAMES:
                self._enable_preset_gate(self.map_side, gesture)
            print(f"[{self.map_side}] {gesture} -> {entry['type']} [{', '.join(entry['inputs'])}]")
            self.status_message = f"{self.map_side}:{gesture} -> {describe(entry)} [{action_label(entry['type'])}]"
            self.map_phase = "list"
            return True
        self.status_message = self.mapping.last_error or t("status.command_refused")
        return False

    def _begin_map_capture(self):
        """Assign button on Key Settings: one key (simple) or a combination (combo)."""
        if self.map_phase not in ("simple", "combo"):
            return False
        self.capture_combo = self.map_phase == "combo"
        self.map_phase = "capture"
        if self.capture_combo:
            self.inputs.begin_combo_capture()
            self.status_message = t("status.hold_modifiers")
        else:
            self.inputs.begin_key_capture()
            self.status_message = t("status.press_key")
        return True

    def handle_mapping_key(self, key, raw=0):
        gesture = self.current_map_gesture()

        if self.map_phase == "hand":
            if key == 27:
                self.exit_mapping_menu()
            elif key in (ord("1"), ord("2")):
                self.map_side = "LEFT" if key == ord("1") else "RIGHT"
                self.map_phase = "list"
                self.map_index = 0
                self.mapping.display(self.map_side)
            return

        if self.map_phase == "list":
            if key == 27:
                self.exit_mapping_menu()
            elif is_left_arrow(raw):
                self._shift_map_index(-1)
            elif is_right_arrow(raw):
                self._shift_map_index(1)
            elif key in (ord("m"), ord("M")):
                self.map_phase = "hand"
            elif key in (ord("a"), ord("A")):
                gesture = self.current_map_gesture()
                self.mapping.set_enabled(self.map_side, gesture, True)
                self.status_message = t("mapping.state_on")
            elif key in (ord("d"), ord("D")):
                gesture = self.current_map_gesture()
                self.mapping.set_enabled(self.map_side, gesture, False)
                self.status_message = t("mapping.state_off")
            elif key in (ord("r"), ord("R")):
                gesture = self.current_map_gesture()
                if gesture_number(gesture) is None:
                    self.status_message = t("status.rename_only")
                else:
                    self.map_phase = "rename"
                    self.rename_buffer = self.engine.display_name(self.map_side, gesture) or ""
                    self.status_message = t("status.rename_prompt")
            elif key in (ord("s"), ord("S")):
                gesture = self.current_map_gesture()
                self.mapping.disable(self.map_side, gesture)
                self.status_message = t("status.disabled", side=self.map_side, gesture=gesture)
            elif key == ord(" "):
                self.mapping.save()
                self.status_message = t("menu.save")
            elif key in (13, 10):
                self.map_phase = "edit"
                self.load_edit_buffers()
                self.print_edit_screen()
            return

        if self.map_phase == "rename":
            gesture = self.current_map_gesture()
            if key == 27:
                self.map_phase = "list"
                self.status_message = t("status.rename_cancel")
            elif key in (13, 10):
                if self.engine.rename_gesture(self.map_side, gesture, self.rename_buffer):
                    self.status_message = f"{gesture} — {self.rename_buffer}"
                    self.map_phase = "list"
                else:
                    self.status_message = t("status.name_refused")
            elif key in (8, 127):
                self.rename_buffer = (self.rename_buffer or "")[:-1]
            elif 32 <= key < 127 and len(self.rename_buffer or "") < 32:
                self.rename_buffer = (self.rename_buffer or "") + chr(key)
            return

        if self.map_phase == "edit":
            if key == 27:
                self.map_phase = "list"
            elif key == ord("1"):
                self.edit_inputs = self.edit_inputs[:1]
                self.edit_slot = 0
                self.map_phase = "simple"
            elif key == ord("2"):
                self.map_phase = "combo"
                if len(self.edit_inputs) < 2:
                    self.edit_inputs.append("D")
                self.edit_modes = list(getattr(self, "edit_modes", []) or [])
                while len(self.edit_modes) < len(self.edit_inputs):
                    self.edit_modes.append(HOLD)
                self.edit_type = COMBINATION
            elif key == ord("3"):
                self.map_phase = "macro"
                self.print_macro_editor()
            return

        if self.map_phase == "capture":
            if key == 27:
                self.inputs.end_key_capture()
                self.map_phase = "combo" if len(self.edit_inputs) > 1 else "simple"
                self.status_message = t("status.capture_cancel")
            return

        if self.map_phase == "simple":
            if key == 27:
                self.map_phase = "edit"
            elif is_left_arrow(raw):
                self._shift_map_index(-1)
                self.load_edit_buffers()
                self.map_phase = "simple"
            elif is_right_arrow(raw):
                self._shift_map_index(1)
                self.load_edit_buffers()
                self.map_phase = "simple"
            elif key in (ord("1"), ord("m"), ord("M")):
                self.edit_type = HOLD
            elif key in (ord("2"), ord("p"), ord("P")):
                self.edit_type = PRESS
            elif key in (ord(" "), 13, 10):
                self.edit_type = HOLD if self.edit_type not in (HOLD, PRESS) else self.edit_type
                self._save_simple_or_combo()
            return

        if self.map_phase == "combo":
            modes = list(getattr(self, "edit_modes", []) or [])
            while len(modes) < len(self.edit_inputs):
                modes.append(HOLD)
            self.edit_modes = modes
            if key == 27:
                self.map_phase = "edit"
            elif is_left_arrow(raw):
                self._shift_map_index(-1)
                self.load_edit_buffers()
                self.map_phase = "combo"
            elif is_right_arrow(raw):
                self._shift_map_index(1)
                self.load_edit_buffers()
                self.map_phase = "combo"
            elif key in (ord("1"), ord("p"), ord("P")):
                self.edit_modes[self.edit_slot] = PRESS
            elif key in (ord("2"), ord("m"), ord("M")):
                self.edit_modes[self.edit_slot] = HOLD
            elif key in (ord("t"), ord("T")):
                self.edit_slot = (self.edit_slot + 1) % max(1, len(self.edit_inputs))
            elif key in (ord("a"), ord("A")):
                if len(self.edit_inputs) < MAX_INPUTS:
                    self.edit_inputs.append("W")
                    self.edit_modes.append(HOLD)
                    self.edit_slot = len(self.edit_inputs) - 1
            elif key in (ord("d"), ord("D")):
                if len(self.edit_inputs) > 1:
                    del self.edit_inputs[self.edit_slot]
                    if self.edit_slot < len(self.edit_modes):
                        del self.edit_modes[self.edit_slot]
                    self.edit_slot = min(self.edit_slot, len(self.edit_inputs) - 1)
            elif key in (ord(" "), 13, 10):
                self.edit_type = COMBINATION
                self._save_simple_or_combo()
            return

        if self.map_phase in ("simple", "combo"):
            return

        if self.map_phase == "macro":
            if key == 27:
                if self.test_active:
                    self.stop_macro_test(t("status.test_stopped"))
                else:
                    self.map_phase = "edit"
            elif is_right_arrow(raw):
                self.macro_index = (self.macro_index + 1) % max(1, len(self.macro_steps))
            elif is_left_arrow(raw):
                self.macro_index = (self.macro_index - 1) % max(1, len(self.macro_steps))
            elif key in (13, 10):
                self.step_backup = dict(self.macro_steps[self.macro_index])
                self.edit_slot = 0
                self.map_phase = "step"
            elif key in (ord("a"), ord("A")):
                self._insert_step({"action": STEP_PRESS, "inputs": ["SPACE"]})
            elif key in (ord("w"), ord("W")):
                self._insert_step({"action": STEP_WAIT, "duration": 150})
            elif key in (ord("m"), ord("M")):
                self._insert_step({"action": STEP_HOLD, "inputs": ["W"]})
            elif key in (ord("r"), ord("R")):
                self._insert_step({"action": STEP_RELEASE, "inputs": ["W"]})
            elif key in (ord("c"), ord("C")):
                self._insert_step({"action": STEP_COMBINATION, "inputs": ["W", "D"]})
            elif key in (ord("b"), ord("B")):
                self._insert_repeat_block()
            elif key in (ord("s"), ord("S")):
                if len(self.macro_steps) > 1:
                    del self.macro_steps[self.macro_index]
                    self.macro_index = min(self.macro_index, len(self.macro_steps) - 1)
            elif key in (ord("t"), ord("T")):
                self.start_macro_test()
            elif key == ord(" "):
                speed = SPEED_CHOICES[self.macro_speed]
                if self.mapping.set_macro(self.map_side, gesture, self.macro_steps, speed):
                    if gesture in geo.SPECIAL_NAMES:
                        self._enable_preset_gate(self.map_side, gesture)
                    print(f"[{self.map_side}] {gesture} -> {action_label(MACRO)} {len(self.macro_steps)} ({speed})")
                    print(t("mapping.preview", text=preview(self.macro_steps)))
                    self.status_message = self.mapping.last_warning or f"{self.map_side}:{gesture} -> {action_label(MACRO)}"
                    self.map_phase = "list"
                else:
                    self.status_message = self.mapping.last_error or t("status.macro_refused")
            return

        if self.map_phase == "step":
            step = self.macro_steps[self.macro_index]
            if key == 27:
                if self.step_backup is not None:
                    self.macro_steps[self.macro_index] = self.step_backup
                self.map_phase = "macro"
            elif is_right_arrow(raw):
                delta = 1
                index = (STEP_ACTIONS.index(step["action"]) + delta) % len(STEP_ACTIONS)
                self.macro_steps[self.macro_index] = self._new_step(STEP_ACTIONS[index], step)
                self.edit_slot = 0
            elif is_left_arrow(raw):
                delta = -1
                index = (STEP_ACTIONS.index(step["action"]) + delta) % len(STEP_ACTIONS)
                self.macro_steps[self.macro_index] = self._new_step(STEP_ACTIONS[index], step)
                self.edit_slot = 0
            elif key in (ord("j"), ord("J"), ord("k"), ord("K")):
                delta = 1 if key in (ord("k"), ord("K")) else -1
                if step["action"] == STEP_WAIT:
                    step["duration"] = max(WAIT_MIN_MS, min(WAIT_MAX_MS, step["duration"] + delta * WAIT_STEP_MS))
                elif step["action"] == STEP_REPEAT_BEGIN:
                    step["count"] = max(REPEAT_MIN, min(REPEAT_MAX, step["count"] + delta))
                elif "inputs" in step:
                    step["inputs"][self.edit_slot] = self._cycle_action(step["inputs"][self.edit_slot], delta)
            elif key in (ord("t"), ord("T")) and "inputs" in step:
                self.edit_slot = (self.edit_slot + 1) % len(step["inputs"])
            elif key in (ord("a"), ord("A")) and "inputs" in step:
                if len(step["inputs"]) < MAX_INPUTS:
                    step["inputs"].append("W")
                    self.edit_slot = len(step["inputs"]) - 1
            elif key in (ord("x"), ord("X")) and "inputs" in step:
                if len(step["inputs"]) > 1:
                    del step["inputs"][self.edit_slot]
                    self.edit_slot = min(self.edit_slot, len(step["inputs"]) - 1)
            elif key in (ord("s"), ord("S"), 13, 10):
                self.step_backup = None
                self.map_phase = "macro"
                self.print_macro_editor()

    def _insert_step(self, step):
        if len(self.macro_steps) >= MAX_STEPS:
            self.status_message = t("status.macro_max", count=MAX_STEPS)
            return False
        self.macro_steps.insert(self.macro_index + 1, step)
        self.macro_index += 1
        return True

    def _insert_repeat_block(self):
        """Insere un bloc REPEAT pret a l'emploi : 4 clics espaces de 120 ms."""
        block = [
            {"action": STEP_REPEAT_BEGIN, "count": 4},
            {"action": STEP_PRESS, "inputs": ["LEFT_MOUSE"]},
            {"action": STEP_WAIT, "duration": 120},
            {"action": STEP_REPEAT_END},
        ]
        if len(self.macro_steps) + len(block) > MAX_STEPS:
            self.status_message = t("status.macro_max", count=MAX_STEPS)
            return
        insert_at = self.macro_index + 1
        self.macro_steps[insert_at:insert_at] = block
        self.macro_index = insert_at + 1  # curseur sur le PRESS interne
        self.status_message = t("status.repeat_block")

    def _move_step(self, delta):
        target = self.macro_index + delta
        if 0 <= target < len(self.macro_steps):
            steps = self.macro_steps
            steps[self.macro_index], steps[target] = steps[target], steps[self.macro_index]
            self.macro_index = target

    @staticmethod
    def _new_step(action, previous):
        """Change le type d'etape en gardant ce qui peut l'etre."""
        if action == STEP_WAIT:
            return {"action": STEP_WAIT, "duration": previous.get("duration", 150)}
        if action == STEP_REPEAT_BEGIN:
            return {"action": STEP_REPEAT_BEGIN, "count": previous.get("count", REPEAT_MIN)}
        if action == STEP_REPEAT_END:
            return {"action": STEP_REPEAT_END}
        return {"action": action, "inputs": list(previous.get("inputs", ["SPACE"]))}

    # ---- Test de macro depuis l'editeur -------------------------------------
    def start_macro_test(self):
        """Execute la macro en cours d'edition, sans lancer le jeu."""
        if self.test_active:
            return
        try:
            steps = normalize_steps(self.macro_steps)
        except MappingError as error:
            self.status_message = t("status.macro_invalid", error=error)
            print(t("macro.test_invalid", error=error))
            return
        self.test_previous_enabled = self.inputs.enabled
        if not self.inputs.enabled:
            self.inputs.enable()
        gesture = self.current_map_gesture()
        print(t("macro.test_active"))
        if self.inputs.start_macro("TEST", steps, f"{self.map_side}:{gesture}", SPEED_CHOICES[self.macro_speed]):
            self.test_active = True
            self.status_message = t("status.test_active")
        else:
            self.finish_macro_test()

    def update_macro_test(self):
        if self.test_active and not self.inputs.is_macro_running("TEST"):
            self.finish_macro_test()

    def stop_macro_test(self, message=None):
        self.inputs.stop_macro("TEST")
        self.finish_macro_test()
        self.status_message = message if message is not None else t("status.test_stopped")

    def finish_macro_test(self):
        if not self.test_active:
            return
        self.test_active = False
        if not self.test_previous_enabled and self.inputs.enabled:
            self.inputs.disable()
        else:
            self.inputs.clear_source("MACRO:TEST")
        print(t("macro.test_done"))
        self.status_message = t("status.test_done")

    # ---- Calibration geometrique (seuils seulement) -------------------------
    def enter_geometry_calibration(self):
        """Mesure des poses reelles. N'ecrit jamais gestures.json."""
        self.ensure_preview_visible()
        self.geo_calib_mode = True
        self.geo_calib_phase = "neutral"
        self.geo_proposal = None
        self.geo_calib_samples = {name: [] for name in ("neutral", "pinch", "tilt_left", "tilt_right")}
        self.reset_state()
        self.status_message = t("status.geo_neutral")
        print(t("status.geo_intro"))

    def exit_geometry_calibration(self, message):
        self.geo_calib_mode = False
        self.geo_proposal = None
        self.reset_state()
        self.status_message = message

    def _capture_geometry_sample(self, result):
        phase = self.geo_calib_phase
        bucket = self.geo_calib_samples.get(phase)
        if bucket is None or len(bucket) >= 20 or not result.hand_landmarks:
            return
        hand = result.hand_landmarks[0]
        points = np.array([(lm.x, lm.y, lm.z) for lm in hand], dtype=np.float32)
        metrics = geo.measure_hand(points)
        bucket.append({
            "pinch_ratio": float(metrics.pinch_ratio),
            "dx": float(metrics.direction[0]),
            "dy": float(metrics.direction[1]),
        })

    def process_geometry_calibration(self, frame, result):
        if self.geo_calib_phase != "confirm":
            self._capture_geometry_sample(result)
        count = len(self.geo_calib_samples.get(self.geo_calib_phase, []))
        label = self.status_message if self.geo_calib_phase == "confirm" else f"{self.status_message} ({count}/8)"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (18, 18, 18), -1)
        put(frame, label[:78], (12, 24), 0.5, (0, 220, 255), 1)
        if self.geo_proposal:
            for row, (key, value) in enumerate(self.geo_proposal.items()):
                put(frame, f"{key}: {value}", (12, 58 + row * 18), 0.42, (255, 255, 255), 1)

    def handle_geometry_calibration_key(self, key):
        if key == 27:
            self.exit_geometry_calibration(t("status.geo_cancel"))
            return
        phases = ("neutral", "pinch", "tilt_left", "tilt_right")
        prompts = {
            "pinch": t("status.geo_pinch"),
            "tilt_left": t("status.geo_tilt_left"),
            "tilt_right": t("status.geo_tilt_right"),
        }
        if self.geo_calib_phase in phases and key == ord(" "):
            if len(self.geo_calib_samples[self.geo_calib_phase]) < 8:
                self.status_message = t("status.geo_short")
                return
            nxt = phases.index(self.geo_calib_phase) + 1
            if nxt >= len(phases):
                try:
                    self.geo_proposal = geo.suggest_geometry_thresholds(self.geo_calib_samples)
                except ValueError as error:
                    self.exit_geometry_calibration(str(error))
                    return
                self.geo_calib_phase = "confirm"
                self.status_message = t("status.geo_confirm")
            else:
                self.geo_calib_phase = phases[nxt]
                self.status_message = prompts[self.geo_calib_phase]
            return
        if self.geo_calib_phase == "confirm" and key in (13, 10):
            self.settings.update(self.geo_proposal or {})
            save_settings(self.settings)
            self.geometry = self._build_geometry()
            self.exit_geometry_calibration(t("status.geo_saved"))
            return

    # ---- Calibration --------------------------------------------------------
    def _largest_hand_for_side(self, result, side):
        """Ne capture QUE la main demandee : jamais un sample gauche dans RIGHT."""
        best_hand = None
        best_area = -1.0
        if not result.hand_landmarks:
            return None
        for index, hand in enumerate(result.hand_landmarks):
            if self.engine.get_hand_side(result, index) != side:
                continue
            xs = [lm.x for lm in hand]
            ys = [lm.y for lm in hand]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if area > best_area:
                best_area = area
                best_hand = hand
        return best_hand

    def enter_calibration(self):
        self._close_other_ui(keep=("calib",))
        self.help_mode = False
        self.lang_mode = False
        self.ensure_preview_visible()
        self.calib_mode = True
        self.calib_phase = "hand"
        self.calib_end_time = 0
        self.reset_state()
        print("\n--------------------------------")
        print(t("calib.header"))
        print("--------------------------------")
        print(t("calib.choose"))
        print(t("mapping.choose_left"))
        print(t("mapping.choose_right"))
        print(t("mapping.opt_cancel"))
        self.status_message = t("status.calib_choose_hand")

    def exit_calibration(self):
        self.engine.cancel_calibration()
        self.calib_mode = False
        self.calib_phase = "hand"
        self.reset_state()
        self.status_message = t("status.calib_cancel")
        print(t("status.calib_cancel"))

    def draw_record_panel(self, frame):
        """Dedicated full-frame form. One window, no leftover panel behind it."""
        height, width = frame.shape[:2]
        self._reset_hits()
        cv2.rectangle(frame, (0, 0), (width, height), (22, 20, 18), -1)
        put(frame, t("ui.record_gestures"), (20, 36), 0.7, (0, 220, 255), 2)
        put(frame, t("ui.name"), (20, 86), 0.5, (0, 220, 255), 1)
        box = (20, 98, max(220, width - 20), 136)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (48, 46, 44), -1)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (110, 108, 100), 1)
        put(frame, self.calib_name_buffer or "_", (28, 124), 0.55, (255, 255, 255), 1, width - 56)
        put(frame, t("ui.hand"), (20, 168), 0.5, (0, 220, 255), 1)
        self._draw_button(
            frame, 20, 180, 180, 32, t("ui.hand_left"), "calib", "side", "LEFT",
            fill=(40, 80, 40) if self.calib_side == "LEFT" else (48, 46, 44),
        )
        self._draw_button(
            frame, 210, 180, 180, 32, t("ui.hand_right"), "calib", "side", "RIGHT",
            fill=(40, 80, 40) if self.calib_side == "RIGHT" else (48, 46, 44),
        )
        samples = int(getattr(self, "samples_per_gesture", 30) or 30)
        put(frame, t("ui.samples_count", count=samples), (20, 240), 0.5, (230, 230, 230), 1)
        self._draw_button(frame, 20, 262, 300, 36, t("ui.start_recording"), "calib", "start", fill=(20, 90, 140))
        self._draw_button(frame, 20, 308, 300, 36, t("ui.add_samples"), "calib", "append", fill=(52, 50, 48))
        self._draw_button(frame, 20, height - 52, 140, 32, t("ui.cancel"), "calib", "back")
        if self.status_message:
            put(frame, self.status_message, (20, height - 70), 0.38, (0, 220, 255), 1)

    def process_calibration(self, frame, result):
        """Advance capture, then draw HUD. Callers may pass a fitted display frame."""
        self.advance_calibration(result)
        self.draw_calibration(frame, result)

    def advance_calibration(self, result):
        """Sample capture and phase timers. Independent of window size."""
        if self.calib_phase == "countdown":
            remaining = max(0.0, self.calib_end_time - time.monotonic())
            if remaining <= 0:
                self.calib_phase = "capture"
            return
        if self.calib_phase == "capture":
            gesture_name = self._calib_ids()[self.calib_index % len(self._calib_ids())]
            hand = self._largest_hand_for_side(result, self.calib_side)
            if hand is None:
                return
            self.engine.add_calibration_sample(self.engine.normalize_landmarks(hand))
            if self.engine.calibration_complete():
                if self.engine.finish_calibration():
                    recognizer = getattr(self, "recognition", None)
                    if recognizer is not None:
                        recognizer.refresh_models()
                    self.calib_phase = "done"
                    self.calib_end_time = time.monotonic() + 1.6
                    self.status_message = t("status.calib_saved", side=self.calib_side, gesture=gesture_name)
                else:
                    self.status_message = t("status.calib_few")
                    self.engine.cancel_calibration()
                    self.calib_phase = "menu"
            return
        if self.calib_phase == "done" and time.monotonic() >= self.calib_end_time:
            self.calib_phase = "menu"
            self.status_message = ""

    def draw_calibration(self, frame, result):
        """Classic on-camera calibration HUD. Drawn after the preview is fitted."""
        height, width = frame.shape[:2]
        if self.calib_phase == "create":
            return

        bar_h = min(220, max(140, height // 3))
        cv2.rectangle(frame, (0, 0), (width, bar_h), (18, 18, 18), -1)
        self._reset_hits()
        put(frame, t("calib.header"), (12, 26), 0.55, (0, 255, 255), 1)
        put(frame, t("calib.title"), (12, 50), 0.42, (180, 180, 180), 1)

        if self.calib_phase == "hand":
            put(frame, t("calib.choose"), (12, 84), 0.6, (255, 255, 255), 1)
            put(frame, t("mapping.choose_left"), (12, 118), 0.6, SIDE_TEXT_COLORS["LEFT"], 2)
            put(frame, t("mapping.choose_right"), (12, 150), 0.6, SIDE_TEXT_COLORS["RIGHT"], 2)
            put(frame, t("mapping.opt_cancel"), (12, 182), 0.45, (180, 180, 180), 1)
            return

        ids = self._calib_ids()
        if not ids:
            put(frame, t("status.calib_few"), (12, 84), 0.5, (0, 180, 255), 1)
            return
        gesture_name = ids[self.calib_index % len(ids)]
        count = self.engine.sample_count(self.calib_side, gesture_name)
        shown = self.engine.display_name(self.calib_side, gesture_name)
        title = editor.format_gesture_label(gesture_name, shown)
        put(frame, f"{side_label(self.calib_side)}", (12, 78), 0.6, SIDE_TEXT_COLORS[self.calib_side], 2)
        put(frame, f"{t('hud.gesture')}: {title}", (12, 108), 0.55, (255, 255, 255), 1)
        put(frame, t("calib.samples_stored", count=count), (12, 134), 0.45, (180, 220, 180), 1)
        spread = self.engine.pose_spread(self.calib_side, gesture_name)
        if spread is not None:
            put(frame, t("calib.quality", value=spread), (12, 156), 0.4, (180, 220, 180), 1)

        if self.calib_phase == "menu":
            put(frame, t("calib.nav_hint", id=title), (12, 180), 0.42, (200, 200, 200), 1)
            put(frame, t("calib.menu_help"), (12, 204), 0.38, (200, 200, 200), 1)
            put(frame, t("calib.store_hint", side=self.calib_side, gesture=gesture_name), (12, 226), 0.36, (180, 180, 180), 1)
        elif self.calib_phase == "mode":
            put(frame, t("calib.replace"), (12, 176), 0.5, (255, 255, 255), 1)
            put(frame, t("calib.append"), (12, 202), 0.5, (255, 255, 255), 1)
            put(frame, t("calib.cancel"), (12, bar_h - 16), 0.42, (200, 200, 200), 1)
        elif self.calib_phase == "countdown":
            remaining = max(0.0, self.calib_end_time - time.monotonic())
            seconds = int(remaining) + (1 if remaining > 0 else 0)
            put(frame, t("calib.countdown", seconds=seconds), (12, 176), 0.65, (0, 255, 255), 2)
            put(frame, t("calib.hold_pose", side=side_label(self.calib_side).lower()),
                (12, 208), 0.42, (200, 200, 200), 1)
        elif self.calib_phase == "capture":
            current, total = self.engine.calibration_progress()
            put(frame, t("calib.samples", current=current, total=total), (12, 176), 0.6, (0, 255, 0), 2)
            hand = self._largest_hand_for_side(result, self.calib_side)
            if hand is None:
                put(frame, t("calib.show_hand", side=side_label(self.calib_side).lower()), (12, 208), 0.45, (0, 180, 255), 1)
            else:
                put(frame, t("calib.hold_pose", side=side_label(self.calib_side).lower()), (12, 208), 0.42, (200, 200, 200), 1)
        elif self.calib_phase == "done":
            put(frame, t("calib.saved", side=self.calib_side, gesture=gesture_name), (12, 176), 0.5, (0, 255, 0), 2)

    def handle_calibration_key(self, key, raw=0):
        if key == 27:
            if self.calib_phase in ("mode", "capture", "countdown"):
                self.engine.cancel_calibration()
                self.calib_phase = "menu"
                self.status_message = t("status.calib_capture_cancel")
            else:
                self.exit_calibration()
            return

        if self.calib_phase == "create":
            if key in (13, 10):
                self._start_new_recorded_gesture(REPLACE)
            elif key in (8, 127):
                self.calib_name_buffer = getattr(self, "calib_name_buffer", "")[:-1]
            elif 32 <= key < 127 and len(getattr(self, "calib_name_buffer", "")) < 32:
                self.calib_name_buffer = getattr(self, "calib_name_buffer", "") + chr(key)
            return

        if self.calib_phase == "hand":
            if key in (ord("1"), ord("2")):
                self.calib_side = "LEFT" if key == ord("1") else "RIGHT"
                self.calib_phase = "menu"
                self.calib_index = 0
                print(t("calib.menu_keys", side=side_label(self.calib_side)))
            return

        if self.calib_phase == "menu":
            if key in (ord("m"), ord("M")):
                self.calib_phase = "hand"
            elif is_right_arrow(raw) or key in (ord(">"), ord(".")):
                ids = self._calib_ids()
                if ids:
                    self.calib_index = (self.calib_index + 1) % len(ids)
            elif is_left_arrow(raw) or key in (ord("<"), ord(",")):
                ids = self._calib_ids()
                if ids:
                    self.calib_index = (self.calib_index - 1) % len(ids)
            elif key in (ord("a"), ord("A")):
                created = self.engine.allocate_gesture(self.calib_side)
                if created:
                    ids = self._calib_ids()
                    self.calib_index = ids.index(created) if created in ids else 0
                    self.status_message = t("status.calib_new", id=created)
            elif key in (13, 10, ord(" ")):
                self.calib_phase = "mode"
                print(t("calib.mode_keys"))
            return

        if self.calib_phase == "mode" and key in (ord("1"), ord("2")):
            mode = REPLACE if key == ord("1") else APPEND
            if self._begin_calibration_capture(mode):
                gesture_name = self._calib_ids()[self.calib_index % len(self._calib_ids())]
                print(f"[{self.calib_side}] Capture {gesture_name} ({mode})...")
                if getattr(self, "log", None):
                    self.log.info("Calibration start %s:%s mode=%s", self.calib_side, gesture_name, mode)

    # ---- Boucle principale --------------------------------------------------
    def open_camera(self):
        """Une seule VideoCapture. Buffer minimal pour ne pas accumuler de retard."""
        index = getattr(self, "camera_index", 0)
        width = getattr(self, "camera_width", 1280)
        height = getattr(self, "camera_height", 720)
        strict = bool(getattr(self, "_camera_strict", False))
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if cap is None or not cap.isOpened():
            self.camera_actual_width = 0
            self.camera_actual_height = 0
            if strict:
                self._revert_camera_resolution()
                self._camera_reopen_previous = True
                self.status_message = t("ui.camera_unsupported", width=int(width), height=int(height))
            return cap
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        actual_w, actual_h = 0, 0
        attempts = ((int(width), int(height)),) if strict else tuple(camera_try_order(width, height))
        matched = False
        for try_w, try_h in attempts:
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, try_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, try_h)
            except Exception:
                continue
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if not accept_camera_size(actual_w, actual_h):
                continue
            if strict and not camera_size_matches(actual_w, actual_h, width, height):
                continue
            matched = True
            break
        if not accept_camera_size(actual_w, actual_h):
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.camera_actual_width = actual_w
        self.camera_actual_height = actual_h
        print(t("app.camera_requested", width=int(width), height=int(height)))
        print(t("app.camera_actual", width=actual_w, height=actual_h))
        if getattr(self, "log", None):
            self.log.info(
                "Camera requested %sx%s actual %sx%s",
                width, height, actual_w, actual_h,
            )
        if strict and not (matched and camera_size_matches(actual_w, actual_h, width, height)):
            self._revert_camera_resolution()
            self._camera_reopen_previous = True
            self.status_message = t("ui.camera_unsupported", width=int(width), height=int(height))
            return cap
        self._camera_strict = False
        if accept_camera_size(actual_w, actual_h):
            if camera_size_matches(actual_w, actual_h, width, height):
                stored_w, stored_h = int(width), int(height)
            else:
                stored_w, stored_h = nearest_camera_preset(actual_w, actual_h)
                self.status_message = t("ui.camera_fallback", width=actual_w, height=actual_h)
            self.camera_width = stored_w
            self.camera_height = stored_h
            settings = getattr(self, "settings", None)
            if isinstance(settings, dict) and (
                int(settings.get("camera_width") or 0) != stored_w
                or int(settings.get("camera_height") or 0) != stored_h
            ):
                settings["camera_width"] = stored_w
                settings["camera_height"] = stored_h
                save_settings(settings)
        return cap

    def run(self):
        cap = None
        miss_count = 0
        print(f"\n=== {WINDOW_TITLE} {APP_VERSION} ===")
        print(t("app.tagline"))
        print(t("app.startup_help") + "\n")
        print(t("app.close_hint"))
        print(t("app.user_data"), app_dir())
        rec = getattr(self, "recognition", None)
        requested = (self.runtime_settings() or {}).get("ml_backend", "knn")
        status = get_ml_backend_status(getattr(rec, "ml", None), requested=requested)
        print(format_ml_startup(status, requested=requested))
        self.log.info("Starting %s %s (cwd=%s, app_dir=%s)", WINDOW_TITLE, APP_VERSION, os.getcwd(), app_dir())
        self.log.info("Model path: %s", self.model_path)
        self.log.info("Gestures: %s", self.gestures_file)
        self.log.info("Mapping: %s", self.mapping_file)
        for side in SIDES:
            print(t("app.calibrated_count", side=side_label(side), count=self.engine.calibrated_count(side)))
        print(t("app.startup_hint") + "\n")

        if KEYBOARD_DIAGNOSTIC or os.environ.get("HANDCTRL_KEYTEST") == "1":
            self.inputs.keyboard_self_test("W")

        if not os.path.isfile(self.model_path):
            show_user_error(
                t("error.model_missing"),
                f"{t('error.model_hint', file=MODEL_FILE)}\n{self.model_path}",
            )
            return

        try:
            cap = self.open_camera()
            self.camera_ok = bool(cap is not None and cap.isOpened())
            if not self.camera_ok:
                show_user_error(
                    t("error.camera_unavailable"),
                    t("error.camera_hint"),
                )
                return
            self.log.info("Camera opened (index=%s)", getattr(self, "camera_index", 0))

            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self.model_path),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.inputs.start_emergency_listener(
                self.on_emergency_stop,
                self.on_preview_toggle,
            )
            with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
                self.log.info("MediaPipe HandLandmarker ready (2 hands, model %s)", os.path.basename(self.model_path))
                while self.running and cap is not None and cap.isOpened():
                    if getattr(self, "_camera_restart", False):
                        self._camera_restart = False
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = self.open_camera()
                        if getattr(self, "_camera_reopen_previous", False):
                            self._camera_reopen_previous = False
                            try:
                                if cap is not None:
                                    cap.release()
                            except Exception:
                                pass
                            cap = self.open_camera()
                        self.camera_ok = bool(cap is not None and cap.isOpened())
                        if not self.camera_ok:
                            self.status_message = t("hud.camera_error")
                            continue
                    frame_start = perf_counter()
                    cam_start = perf_counter()
                    ret, frame = cap.read()
                    self.measure("camera", perf_counter() - cam_start)
                    if not ret or frame is None or frame.size == 0:
                        miss_count += 1
                        self.camera_ok = False
                        self.status_message = t("hud.camera_interrupted")
                        if miss_count == 1:
                            self.log.warning("Camera frame missing")
                            self.inputs.stop_all_macros()
                            self.inputs.release_all()
                        if miss_count >= 15:
                            show_user_error(
                                t("error.camera_unavailable"),
                                t("error.camera_lost"),
                            )
                            break
                        cv2.waitKey(40)
                        continue
                    miss_count = 0
                    self.camera_ok = True

                    # Releases en attente puis avancement des macros. Jamais bloquant.
                    self.inputs.tick()
                    self.handle_emergency_flag()
                    self.handle_preview_toggle()

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    start = perf_counter()
                    result = landmarker.detect_for_video(mp_img, self.get_timestamp())
                    self.measure("mediapipe", perf_counter() - start)

                    if self.map_mode:
                        self.poll_key_capture()
                        self.update_macro_test()
                        overlay = "map"
                    elif self.sys_mode:
                        self.poll_key_capture()
                        overlay = "sys"
                    elif self.editor_mode:
                        overlay = "editor"
                    elif getattr(self, "settings_mode", False):
                        overlay = "settings"
                    elif getattr(self, "input_panel", False):
                        overlay = "input"
                    elif self.calib_mode:
                        if self.calib_phase != "create" and result.hand_landmarks:
                            for index, hand in enumerate(result.hand_landmarks):
                                side = self.engine.get_hand_side(result, index)
                                self._draw_landmarks(frame, hand, hand_point_color(side))
                        self.advance_calibration(result)
                        overlay = "calib"
                    elif self.geo_calib_mode:
                        self.process_geometry_calibration(frame, result)
                        overlay = None
                    else:
                        self.process_frame(frame, result, draw_overlay=False)
                        self._log_gesture_changes()
                        overlay = "sys_test" if getattr(self, "system_test_mode", False) else "main"

                    self._sync_game_osd()
                    if getattr(self, "preview_visible", True):
                        # Checked before fit_preview_frame, which would recreate a closed window.
                        if getattr(self, "_preview_open", False) and preview_was_closed(WINDOW_TITLE):
                            self.log.info("Preview window closed by the user")
                            self.running = False
                            break
                        overlay_main = overlay in ("main", "sys_test", "help")
                        frame = fit_preview_frame(WINDOW_TITLE, frame)
                        if overlay_main:
                            self.draw_ui(frame)
                            if overlay == "sys_test":
                                self.draw_system_test(frame)
                        elif overlay == "map":
                            self.draw_mapping_menu(frame)
                        elif overlay == "sys":
                            self.draw_system_menu(frame)
                        elif overlay == "editor":
                            self.draw_editor(frame)
                        elif overlay == "settings":
                            self.draw_gesture_settings(frame)
                        elif overlay == "input":
                            self.draw_input_panel(frame)
                        elif overlay == "calib":
                            if self.calib_phase == "create":
                                self.draw_record_panel(frame)
                            else:
                                self.draw_calibration(frame, result)
                        bind_preview_mouse(WINDOW_TITLE, self._on_preview_mouse)
                        show_preview_frame(WINDOW_TITLE, frame)
                        self._preview_open = True
                        if not getattr(self, "_preview_icon_applied", False):
                            self._preview_icon_applied = apply_window_icon(WINDOW_TITLE)
                    key_fn = getattr(cv2, "waitKeyEx", cv2.waitKey)
                    raw = key_fn(1)
                    if raw is None or int(raw) < 0:
                        key = 255
                        raw = -1
                    else:
                        raw = int(raw)
                        key = raw & 0xFF
                    self.measure("total", perf_counter() - frame_start)
                    self.update_perf_text()
                    self.inputs._preview_allowed = self._preview_key_allowed
                    # Les touches d'interface ne comptent que si la fenêtre HandTrack est devant.
                    # Les gestes, eux, ont déjà été traités plus haut, focus ou non.
                    if key not in (255, 0) and self.preview_visible and not interface_keys_enabled(own_window_is_foreground(WINDOW_TITLE)):
                        if not ((self.map_mode and self.map_phase == "capture") or (self.sys_mode and self.sys_phase == "capture")):
                            key = 255
                            raw = -1

                    ctx = ui_context.of(self)
                    if ui_context.is_text_context(ctx):
                        if self.map_mode:
                            self.handle_mapping_key(key, raw)
                        elif self.sys_mode:
                            self.handle_system_key(key, raw)
                        elif self.calib_mode:
                            self.handle_calibration_key(key, raw)
                    elif key == ord("q") or key == ord("Q"):
                        if ui_context.allows_quit(ctx):
                            self.running = False
                    elif self.map_mode:
                        self.handle_mapping_key(key, raw)
                    elif getattr(self, "safe_mode", False) and key == 27:
                        self.safe_mode = False
                        self.status_message = t("status.safe_off")
                    elif self.sys_mode:
                        self.handle_system_key(key, raw)
                    elif getattr(self, "system_test_mode", False) and key == 27:
                        self.system_test_mode = False
                        self.status_message = t("sys.test_off")
                    elif self.editor_mode:
                        self.handle_editor_key(key, raw)
                    elif getattr(self, "settings_mode", False) and key == 27:
                        self.settings_mode = False
                    elif getattr(self, "input_panel", False) and key == 27:
                        self.input_panel = False
                    elif self.calib_mode:
                        self.handle_calibration_key(key, raw)
                    elif self.geo_calib_mode:
                        self.handle_geometry_calibration_key(key)
                    elif getattr(self, "recording", False):
                        if key != 255:
                            self.status_message = t("status.recording_busy")
                    elif ctx in (ui_context.MAIN, ui_context.HELP, ui_context.LANGUAGE):
                        self.handle_idle_key(key, raw)
        except Exception as error:
            self.log.exception("Fatal error")
            show_user_error(t("app.unexpected_error"), str(error))
            self.status_message = str(error)
        finally:
            self.log.info(t("app.closing"))
            self.inputs.stop_all_macros()
            self.inputs.release_all()
            self.inputs.stop_emergency_listener()
            if getattr(self, "replay_handle", None) is not None:
                self.replay_handle.close()
                self.replay_handle = None
            if getattr(self, "game_osd", None) is not None:
                try:
                    self.game_osd.destroy()
                except Exception:
                    pass
                self.game_osd = None
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            self.log.info("Shutdown complete (inputs released, camera closed)")


def _cli_flags(argv):
    return {
        "debug": "--debug" in argv or os.environ.get("HANDCTRL_DEBUG") == "1",
        "debug_verbose": "--debug-verbose" in argv,
        "reset": "--reset-config" in argv,
        "diagnostic": "--diagnostic" in argv,
        "help": "--help" in argv or "-h" in argv,
        "replay": next((argv[index + 1] for index, item in enumerate(argv) if item == "--replay" and index + 1 < len(argv)), None),
        "benchmark": "--benchmark" in argv,
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    flags = _cli_flags(argv)
    settings = load_settings()
    apply_language(settings, interactive=False)
    if flags["help"]:
        launcher = "HandController.exe" if getattr(sys, "frozen", False) else "python main.py"
        print(f"{WINDOW_TITLE} {APP_VERSION}")
        print(t("cli.usage", launcher=launcher))
        print(t("cli.debug"))
        print(t("cli.debug_verbose"))
        print(t("cli.diagnostic"))
        print(t("cli.reset"))
        print(t("cli.replay"))
        print(t("cli.benchmark"))
        return 0

    if flags["diagnostic"]:
        from diagnostics import format_report
        ready, report = format_report(check_camera=True, camera_index=int(settings.get("camera_index", 0)))
        print(report)
        return 0 if ready else 1
    if flags["debug"] or flags.get("debug_verbose"):
        settings["debug"] = True
    setup_logging(bool(settings.get("debug")) or bool(flags.get("debug_verbose")))
    input_mod.VERBOSE = bool(flags.get("debug_verbose"))
    global VERBOSE_GESTURE_LOG
    VERBOSE_GESTURE_LOG = bool(flags.get("debug_verbose"))

    if flags["reset"]:
        return 0 if reset_user_configuration(settings.get("language")) else 1

    if flags["replay"]:
        return run_replay(flags["replay"], settings)
    if flags.get("benchmark"):
        return run_benchmark(settings)

    if language_configured(settings):
        apply_language(settings, interactive=False)
    else:
        chosen = _choose_language(settings)
        if chosen is None:
            return 0
        apply_language(settings, interactive=False)
    app = HandControllerApp(settings)
    app.debug_verbose = bool(flags.get("debug_verbose"))
    return _launch_controller(app)


def _choose_language(settings):
    """First launch: console prompt. No Qt dialog."""
    return apply_language(settings, interactive=True)


def _launch_controller(controller):
    """OpenCV camera loop. Recognition continues if this window is not focused."""
    controller.run()
    return 0


def _replay_detections(record):
    detections = []
    for hand in record.get("hands") or []:
        side = hand.get("hand_id")
        raw = hand.get("landmarks_raw")
        if side not in ("LEFT", "RIGHT") or not raw:
            continue
        detections.append({
            "hand": points_to_landmarks(raw),
            "side": side,
            "score": float(hand.get("confidence", 1.0)),
            "raw": UNKNOWN,
            "conf": 0.0,
        })
    return detections


def run_replay(path, settings):
    """Rejoue des landmarks. INPUT reste OFF : aucun clavier, aucune souris."""
    if not os.path.isfile(path):
        print(t("replay.missing", path=path))
        return 1
    app = HandControllerApp(settings)
    app.passive = True
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    for record in replay.read_frames(path):
        app.collect_detections = lambda result, current=record: _replay_detections(current)
        app.process_frame(blank, None)
        print(app.replay_frame, {side: app.curr_gestures[side] for side in SIDES})
        app.replay_frame += 1
    print(t("replay.done", enabled=app.inputs.enabled))
    return 0 if not app.inputs.enabled else 1


def run_benchmark(settings=None):
    """Leave-one-out ML comparison on gestures.json. Never loads the readonly backup dataset."""
    from classifier_benchmark import format_ml_benchmark, run_ml_benchmark
    report = run_ml_benchmark()
    print(format_ml_benchmark(report))
    return 0 if not report.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
