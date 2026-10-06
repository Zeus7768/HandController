"""Moteur de detection geometrique des 21 landmarks MediaPipe (NumPy).

Couche ADDITIVE. Elle ne remplace ni le k-NN de gesture_engine, ni la
stabilisation des gestes appris G01-G30 : les deux systemes tournent en
parallele et peuvent etre actifs en meme temps.

    MediaPipe
        v
    21 landmarks
        v
    geometry_engine (NumPy)          <- ce fichier
        v
    Special Gesture ID
        v
    gesture_mapping (mapping existant)
        v
    input_controller
        v
    clavier / souris

Ce module ne connait AUCUNE action : il ne sait pas ce qu'est une touche, un
clic ou une macro. Il n'importe que NumPy (verifie par un test) et retourne
uniquement des identifiants d'etat. C'est le mapping qui decide de l'action.

Organisation du fichier :
    1. Indices et accesseurs des landmarks
    2. Primitives geometriques generiques
    3. Configuration (tous les seuils au meme endroit)
    4. Identifiants de gestes speciaux
    5. Mesures par main
    6. Detecteurs (modulaires, un par famille de gestes)
    7. Moteur (les deux mains)

Pour ajouter un detecteur plus tard : ecrire une classe avec update()/reset(),
declarer ses identifiants dans SPECIAL_BY_SIDE, et l'enregistrer dans
HandGeometry._build_detectors. Rien d'autre a toucher dans le HandController.
"""
import numpy as np

# =========================================================================
# 1. Indices et accesseurs des landmarks (MediaPipe Hand Landmarker)
# =========================================================================
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

LANDMARK_COUNT = 21
FEATURE_SIZE = LANDMARK_COUNT * 3

LANDMARK_NAMES = {
    WRIST: "Wrist",
    THUMB_CMC: "Thumb CMC", THUMB_MCP: "Thumb MCP", THUMB_IP: "Thumb IP", THUMB_TIP: "Thumb Tip",
    INDEX_MCP: "Index MCP", INDEX_PIP: "Index PIP", INDEX_DIP: "Index DIP", INDEX_TIP: "Index Tip",
    MIDDLE_MCP: "Middle MCP", MIDDLE_PIP: "Middle PIP", MIDDLE_DIP: "Middle DIP", MIDDLE_TIP: "Middle Tip",
    RING_MCP: "Ring MCP", RING_PIP: "Ring PIP", RING_DIP: "Ring DIP", RING_TIP: "Ring Tip",
    PINKY_MCP: "Pinky MCP", PINKY_PIP: "Pinky PIP", PINKY_DIP: "Pinky DIP", PINKY_TIP: "Pinky Tip",
}

# Bases des doigts : la "ligne des articulations", stable quand les doigts bougent.
MCP_INDICES = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
FINGERTIP_INDICES = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGERS = ("THUMB", "INDEX", "MIDDLE", "RING", "PINKY")
FINGER_UP_FINGERS = ("INDEX", "MIDDLE", "RING", "PINKY", "THUMB")
FINGER_SEGMENT = {
    "THUMB": (THUMB_MCP, THUMB_TIP),
    "INDEX": (INDEX_MCP, INDEX_TIP),
    "MIDDLE": (MIDDLE_MCP, MIDDLE_TIP),
    "RING": (RING_MCP, RING_TIP),
    "PINKY": (PINKY_MCP, PINKY_TIP),
}
# tip, pip/ip, mcp — used by detect_finger_up(). Thumb uses IP instead of PIP.
FINGER_JOINTS = {
    "INDEX": (INDEX_TIP, INDEX_PIP, INDEX_MCP),
    "MIDDLE": (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
    "RING": (RING_TIP, RING_PIP, RING_MCP),
    "PINKY": (PINKY_TIP, PINKY_PIP, PINKY_MCP),
    "THUMB": (THUMB_TIP, THUMB_IP, THUMB_MCP),
}
# Proximal → distal joints for direction (image coords: +Y is down).
FINGER_CHAIN = {
    "INDEX": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "MIDDLE": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "RING": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "PINKY": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
    "THUMB": (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
}


def point(landmarks, index):
    """Un landmark par son indice, en tableau NumPy (x, y, z)."""
    return landmarks[index]


def wrist(landmarks):
    return landmarks[WRIST]


def thumb_tip(landmarks):
    return landmarks[THUMB_TIP]


def index_mcp(landmarks):
    return landmarks[INDEX_MCP]


def index_tip(landmarks):
    return landmarks[INDEX_TIP]


def middle_mcp(landmarks):
    return landmarks[MIDDLE_MCP]


def ring_mcp(landmarks):
    return landmarks[RING_MCP]


def pinky_mcp(landmarks):
    return landmarks[PINKY_MCP]


def landmark_points(hand):
    """(21, 3) float32 depuis une liste de landmarks MediaPipe.

    fromiter evite la liste de listes Python intermediaire a chaque frame.
    Retourne None si la main est incomplete.
    """
    if hand is None or len(hand) < LANDMARK_COUNT:
        return None
    flat = np.fromiter(
        (value for lm in hand for value in (lm.x, lm.y, lm.z)),
        dtype=np.float32,
        count=LANDMARK_COUNT * 3,
    )
    return flat.reshape(LANDMARK_COUNT, 3)


def points_from_features(features):
    """Reshape a 63-vector into 21x3 landmarks.

    k-NN features are now a palm-local frame. Geometry Engine must keep the
    original filtered image landmarks (stable_points / geometry_points).
    This helper still reshapes any 63-vector for tests and fallbacks.
    """
    if features is None:
        return None
    array = np.asarray(features, dtype=np.float32)
    if array.size != FEATURE_SIZE:
        return None
    return array.reshape(LANDMARK_COUNT, 3)


# =========================================================================
# 2. Primitives geometriques generiques (NumPy)
# =========================================================================
# Sous cette valeur un vecteur est considere comme nul : aucune division,
# aucun angle, pas de NaN qui se propagerait dans les detecteurs.
EPSILON = 1e-9


def vector_between_points(a, b):
    """Vecteur a -> b."""
    return np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)


def distance_between_points(a, b):
    """Distance euclidienne. ||a - b||"""
    return float(np.linalg.norm(vector_between_points(a, b)))


def vector_length(v):
    return float(np.linalg.norm(v))


def normalize_vector(v):
    """Vecteur unitaire. Un vecteur nul est renvoye tel quel (pas de NaN)."""
    array = np.asarray(v, dtype=np.float32)
    length = float(np.linalg.norm(array))
    if length < EPSILON:
        return np.zeros_like(array)
    return array / length


def angle_between_vectors(a, b):
    """Angle non signe entre deux vecteurs, en degres (0 a 180).

    Retourne 0.0 si l'un des deux vecteurs est nul.
    """
    unit_a = normalize_vector(a)
    unit_b = normalize_vector(b)
    if vector_length(unit_a) < EPSILON or vector_length(unit_b) < EPSILON:
        return 0.0
    dot = float(np.clip(np.dot(unit_a, unit_b), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def angle_2d(a, b):
    """Angle SIGNE de a vers b dans le plan image, en degres (-180 a 180).

    Signe positif = rotation horaire a l'ecran, car l'axe y des images
    descend. Retourne 0.0 si l'un des vecteurs est nul.
    """
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    if vector_length(va) < EPSILON or vector_length(vb) < EPSILON:
        return 0.0
    cross = float(va[0] * vb[1] - va[1] * vb[0])
    dot = float(va[0] * vb[0] + va[1] * vb[1])
    return float(np.degrees(np.arctan2(cross, dot)))


def vector_angle(v):
    """Orientation absolue d'un vecteur dans le plan image, en degres.

    atan2(dy, dx). 0 = vers la droite de l'image, -90 = vers le haut,
    +90 = vers le bas (l'axe y des images descend), 180 = vers la gauche.
    Retourne 0.0 pour un vecteur nul.
    """
    array = np.asarray(v, dtype=np.float32)
    if vector_length(array) < EPSILON:
        return 0.0
    return float(np.degrees(np.arctan2(float(array[1]), float(array[0]))))


def distance_from_landmarks(landmarks, index_a, index_b):
    """Distance entre deux landmarks designes par leurs indices."""
    return distance_between_points(landmarks[index_a], landmarks[index_b])


def hand_scale(landmarks):
    """Taille de reference de la main : poignet (0) -> base du majeur (9).

    Ce segment appartient a la paume, qui est RIGIDE : il ne change pas quand
    les doigts s'ouvrent ou se ferment, contrairement a une distance entre
    deux bouts de doigts. C'est aussi l'echelle utilisee par
    GestureEngine.normalize_landmarks, donc les deux couches mesurent la main
    de la meme facon.

    Le resultat est borne pour que hand_scale == 0 ne puisse jamais provoquer
    de division par zero.
    """
    return max(distance_from_landmarks(landmarks, WRIST, MIDDLE_MCP), MIN_HAND_SCALE)


def palm_width(landmarks):
    """Largeur de la paume : base de l'index (5) -> base de l'auriculaire (17).

    Deuxieme reference rigide, utile pour diagnostiquer une main de travers.
    """
    return distance_from_landmarks(landmarks, INDEX_MCP, PINKY_MCP)


def relative_length(landmarks, index_a, index_b):
    """Distance between two landmarks, divided by wrist→middle MCP."""
    return distance_from_landmarks(landmarks, index_a, index_b) / hand_scale(landmarks)


def palm_aspect(landmarks):
    """Palm width / hand_scale. Typical open hand sits near 0.7–1.4."""
    return palm_width(landmarks) / hand_scale(landmarks)


def finger_extension_ratio(landmarks, finger):
    """Tip→MCP (or thumb tip→MCP) / hand_scale. Long fingers ~0.8–1.2 when open."""
    finger = str(finger or "").upper()
    segment = FINGER_SEGMENT.get(finger)
    if segment is None or landmarks is None:
        return 0.0
    return relative_length(landmarks, segment[0], segment[1])


def finger_curl(landmarks, finger):
    """0 = stretched, 1 = fully tucked. Same reference as finger_extension_ratio."""
    typical = 0.55 if str(finger or "").upper() == "THUMB" else 1.0
    ratio = finger_extension_ratio(landmarks, finger)
    return float(max(0.0, min(1.0, 1.0 - ratio / max(typical, MIN_HAND_SCALE))))


def finger_arch(landmarks, finger):
    """tip-to-wrist / mcp-to-wrist. Raised fingers sit above 1.0; tucked near ≤1.0.

    Thumb uses CMC as the proximal base (different kinematics from the four long fingers).
    """
    finger = str(finger or "").upper()
    joints = FINGER_JOINTS.get(finger)
    if joints is None or landmarks is None:
        return 0.0
    tip_i, _mid_i, mcp_i = joints
    base_i = THUMB_CMC if finger == "THUMB" else mcp_i
    proximal = max(distance_from_landmarks(landmarks, WRIST, base_i), MIN_HAND_SCALE)
    return distance_from_landmarks(landmarks, WRIST, tip_i) / proximal


def finger_spread(landmarks):
    """Index tip → pinky tip / hand_scale. Open palm is large; a pinch/fist is small."""
    return relative_length(landmarks, INDEX_TIP, PINKY_TIP)


def thumb_span(landmarks):
    """Thumb tip → index MCP / hand_scale. Thumb-specific abduction, not a long-finger curl."""
    return relative_length(landmarks, THUMB_TIP, INDEX_MCP)


def relative_metrics(landmarks):
    """All geometry ratios used by detectors and the diagnostic tool. NumPy only."""
    scale = hand_scale(landmarks)
    pinch = float(np.linalg.norm(landmarks[THUMB_TIP][:2] - landmarks[INDEX_TIP][:2]))
    curls = {finger: finger_curl(landmarks, finger) for finger in FINGER_UP_FINGERS}
    arches = {finger: finger_arch(landmarks, finger) for finger in FINGER_UP_FINGERS}
    extensions = {finger: finger_extension_ratio(landmarks, finger) for finger in FINGER_UP_FINGERS}
    return {
        "scale": scale,
        "pinch_ratio": pinch / scale,
        "palm_aspect": palm_aspect(landmarks),
        "finger_spread": finger_spread(landmarks),
        "thumb_span": thumb_span(landmarks),
        "curl_index": curls["INDEX"],
        "curl_middle": curls["MIDDLE"],
        "curl_ring": curls["RING"],
        "curl_pinky": curls["PINKY"],
        "curl_thumb": curls["THUMB"],
        "arch_index": arches["INDEX"],
        "arch_middle": arches["MIDDLE"],
        "arch_ring": arches["RING"],
        "arch_pinky": arches["PINKY"],
        "arch_thumb": arches["THUMB"],
        "ext_index": extensions["INDEX"],
        "ext_middle": extensions["MIDDLE"],
        "ext_ring": extensions["RING"],
        "ext_pinky": extensions["PINKY"],
        "ext_thumb": extensions["THUMB"],
    }


# Sanity bounds (extreme only). Justified against synthetic PALM_POSE (~aspect 0.12)
# and MediaPipe image hands (typical aspect ~0.5–1.5, finger ext ~0.3–1.3).
# Below 0.08 the MCP row has collapsed; above 4.0 landmarks have exploded.
_PALM_ASPECT_MIN = 0.08
_PALM_ASPECT_MAX = 4.0
_FINGER_EXT_MAX = 4.0


def geometry_sane(landmarks):
    """True if palm/finger ratios could belong to a real hand. Conservative."""
    ok, _reasons = geometry_sanity(landmarks)
    return ok


def geometry_sanity(landmarks):
    """Return (ok, reasons). Does not decide ACCEPT/HOLD/REJECT by itself."""
    reasons = []
    if landmarks is None:
        return False, ["missing"]
    array = np.asarray(landmarks, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < LANDMARK_COUNT:
        return False, ["shape"]
    if not np.isfinite(array[:LANDMARK_COUNT, :3]).all():
        return False, ["nan"]
    scale = max(distance_from_landmarks(array, WRIST, MIDDLE_MCP), MIN_HAND_SCALE)
    if scale <= MIN_HAND_SCALE:
        reasons.append("scale")
    aspect = distance_from_landmarks(array, INDEX_MCP, PINKY_MCP) / scale
    if aspect < _PALM_ASPECT_MIN or aspect > _PALM_ASPECT_MAX:
        reasons.append("palm_aspect")
    for finger in FINGER_UP_FINGERS:
        segment = FINGER_SEGMENT[finger]
        extension = distance_from_landmarks(array, segment[0], segment[1]) / scale
        if extension > _FINGER_EXT_MAX:
            reasons.append("finger_" + finger.lower())
    return (not reasons), reasons


# =========================================================================
# 3. Configuration : tous les seuils au meme endroit
# =========================================================================
# Aucune de ces valeurs n'est definitive. Elles sont toutes exprimees en
# fraction de la taille de la main (voir hand_scale), donc elles restent
# valables quand la main se rapproche ou s'eloigne de la camera.

# Garde-fou de division.
MIN_HAND_SCALE = 1e-4

# ---- Pinch (pouce 4 / index 8) ------------------------------------------
PINCH_ON_THRESHOLD = 0.25       # ratio <= ON  -> pinch ON
PINCH_OFF_THRESHOLD = 0.32      # ratio >= OFF -> pinch OFF
# Entre les deux : l'etat precedent est conserve (hysteresis).

# z de MediaPipe est une profondeur relative bien moins precise que x et y.
# L'inclure rend le ratio nettement plus bruite, donc la mesure est faite en
# 2D image par defaut.
PINCH_USE_DEPTH = False

# ---- Orientation : fusion paume + index ---------------------------------
# Le vecteur de direction est la somme ponderee du vecteur de paume et du
# vecteur d'index (voir HandMetrics.direction). La paume domine : elle est
# rigide et toujours mesurable, alors que le vecteur d'index devient court et
# peu fiable quand le doigt est replie. L'index affine la direction quand il
# est tendu, ce qui rend l'inclinaison plus stable que la paume seule.
PALM_WEIGHT = 0.6
INDEX_WEIGHT = 0.4

# ---- Inclinaison (tilt), par direction ----------------------------------
# Seuils appliques aux composantes du vecteur de direction fusionne. Le
# vecteur est exprime en unites de taille de main : environ 1.0 pour une main
# ouverte et tendue, nettement moins pour une main refermee.
TILT_LEFT_THRESHOLD = 0.45
TILT_RIGHT_THRESHOLD = 0.45
TILT_UP_THRESHOLD = 0.80
TILT_DOWN_THRESHOLD = 0.55

# ---- Direction de l'index seul (methode de controle alternative) --------
INDEX_DIRECTION_THRESHOLD = 0.45

# ---- Stabilite ----------------------------------------------------------
# Marge d'hysteresis retiree du seuil d'entree pour obtenir le seuil de
# sortie : on entre a THRESHOLD, on ne revient au neutre qu'en dessous de
# THRESHOLD - DEAD_ZONE. Les petites variations de la main ne changent donc
# pas l'etat.
DEAD_ZONE = 0.12

# Frames consecutives exigees avant de valider un changement d'etat.
STABILITY_FRAMES = 2
# Le pinch doit rester reactif : une frame suffit.
PINCH_STABILITY_FRAMES = 1

# ---- Poing et paume ouverte (4 doigts, pas le pouce) --------------------
# Longueur bout->base / taille de la main. Court = doigt replie.
# Le poing exige que le PLUS LONG des quatre soit encore court : un index
# tendu (pointage) ne declenche donc pas. La paume exige que le PLUS COURT
# soit encore long. Les deux bandes ne se recouvrent pas.
FIST_ON_THRESHOLD = 0.45        # max(4) <= ON  -> poing
FIST_OFF_THRESHOLD = 0.62       # max(4) >= OFF -> plus un poing
OPEN_PALM_ON_THRESHOLD = 0.75   # min(4) >= ON  -> paume ouverte
OPEN_PALM_OFF_THRESHOLD = 0.58  # min(4) <= OFF -> plus une paume ouverte

# ---- Finger Up (doigt tendu, pas un sample k-NN) --------------------------
# Longueur bout->base / taille de main, plus un test d'alignement. Ce n'est
# PAS tip_y < pip_y : l'orientation de la paume est prise en compte.
FINGER_UP_ON_THRESHOLD = 0.55
FINGER_UP_OFF_THRESHOLD = 0.40
FINGER_UP_BEND_MAX = 45.0
FINGER_UP_PALM_ALIGN_MAX = 70.0
# Image-space UP: dy is negative. Hold margin is smaller to avoid flicker.
FINGER_UP_DIR_MARGIN = 0.12
FINGER_UP_DIR_HOLD_MARGIN = 0.02
FINGER_UP_MODE_SINGLE = "single"
FINGER_UP_MODE_MULTI = "multi"


# =========================================================================
# 4. Identifiants de gestes speciaux
# =========================================================================
# Volontairement distincts de G01-G30 : ce ne sont PAS des gestes appris, ils
# n'occupent aucun slot de calibration et ne sont JAMAIS renommes en G31/G32.
SIDES = ("LEFT", "RIGHT")

# Familles de detecteurs. Chaque famille a sa propre source de maintien dans
# input_controller, donc les familles n'interferent jamais entre elles.
PINCH = "PINCH"
TILT = "TILT"
INDEX = "INDEX"
FIST = "FIST"
PALM = "PALM"
FINGER_UP = "FINGER_UP"
FINGER_INDEX = "FINGER_INDEX"
FINGER_MIDDLE = "FINGER_MIDDLE"
FINGER_RING = "FINGER_RING"
FINGER_PINKY = "FINGER_PINKY"
FINGER_THUMB = "FINGER_THUMB"
FINGER_FAMILIES = (FINGER_INDEX, FINGER_MIDDLE, FINGER_RING, FINGER_PINKY, FINGER_THUMB)

RIGHT_PINCH = "RIGHT_PINCH"
LEFT_PINCH = "LEFT_PINCH"

LEFT_FIST = "LEFT_FIST"
RIGHT_FIST = "RIGHT_FIST"
LEFT_OPEN_PALM = "LEFT_OPEN_PALM"
RIGHT_OPEN_PALM = "RIGHT_OPEN_PALM"

LEFT_TILT_LEFT = "LEFT_TILT_LEFT"
LEFT_TILT_RIGHT = "LEFT_TILT_RIGHT"
LEFT_TILT_UP = "LEFT_TILT_UP"
LEFT_TILT_DOWN = "LEFT_TILT_DOWN"
LEFT_TILT_NEUTRAL = "LEFT_TILT_NEUTRAL"

RIGHT_TILT_LEFT = "RIGHT_TILT_LEFT"
RIGHT_TILT_RIGHT = "RIGHT_TILT_RIGHT"
RIGHT_TILT_UP = "RIGHT_TILT_UP"
RIGHT_TILT_DOWN = "RIGHT_TILT_DOWN"
RIGHT_TILT_NEUTRAL = "RIGHT_TILT_NEUTRAL"

LEFT_INDEX_LEFT = "LEFT_INDEX_LEFT"
LEFT_INDEX_RIGHT = "LEFT_INDEX_RIGHT"
LEFT_INDEX_UP = "LEFT_INDEX_UP"
LEFT_INDEX_DOWN = "LEFT_INDEX_DOWN"
LEFT_INDEX_NEUTRAL = "LEFT_INDEX_NEUTRAL"

RIGHT_INDEX_LEFT = "RIGHT_INDEX_LEFT"
RIGHT_INDEX_RIGHT = "RIGHT_INDEX_RIGHT"
RIGHT_INDEX_UP = "RIGHT_INDEX_UP"
RIGHT_INDEX_DOWN = "RIGHT_INDEX_DOWN"
RIGHT_INDEX_NEUTRAL = "RIGHT_INDEX_NEUTRAL"

LEFT_FINGER_INDEX = "LEFT_FINGER_INDEX"
LEFT_FINGER_MIDDLE = "LEFT_FINGER_MIDDLE"
LEFT_FINGER_RING = "LEFT_FINGER_RING"
LEFT_FINGER_PINKY = "LEFT_FINGER_PINKY"
LEFT_FINGER_THUMB = "LEFT_FINGER_THUMB"
RIGHT_FINGER_INDEX = "RIGHT_FINGER_INDEX"
RIGHT_FINGER_MIDDLE = "RIGHT_FINGER_MIDDLE"
RIGHT_FINGER_RING = "RIGHT_FINGER_RING"
RIGHT_FINGER_PINKY = "RIGHT_FINGER_PINKY"
RIGHT_FINGER_THUMB = "RIGHT_FINGER_THUMB"

# Directions internes d'un detecteur de direction. NEUTRAL existe toujours :
# c'est lui qui permet au systeme HOLD de relacher proprement les touches.
DIR_LEFT = "LEFT"
DIR_RIGHT = "RIGHT"
DIR_UP = "UP"
DIR_DOWN = "DOWN"
DIR_NEUTRAL = "NEUTRAL"
DIRECTIONS = (DIR_LEFT, DIR_RIGHT, DIR_UP, DIR_DOWN, DIR_NEUTRAL)

# Table direction -> identifiant, par main et par famille.
_DIRECTION_IDS = {
    ("LEFT", TILT): {
        DIR_LEFT: LEFT_TILT_LEFT, DIR_RIGHT: LEFT_TILT_RIGHT, DIR_UP: LEFT_TILT_UP,
        DIR_DOWN: LEFT_TILT_DOWN, DIR_NEUTRAL: LEFT_TILT_NEUTRAL,
    },
    ("RIGHT", TILT): {
        DIR_LEFT: RIGHT_TILT_LEFT, DIR_RIGHT: RIGHT_TILT_RIGHT, DIR_UP: RIGHT_TILT_UP,
        DIR_DOWN: RIGHT_TILT_DOWN, DIR_NEUTRAL: RIGHT_TILT_NEUTRAL,
    },
    ("LEFT", INDEX): {
        DIR_LEFT: LEFT_INDEX_LEFT, DIR_RIGHT: LEFT_INDEX_RIGHT, DIR_UP: LEFT_INDEX_UP,
        DIR_DOWN: LEFT_INDEX_DOWN, DIR_NEUTRAL: LEFT_INDEX_NEUTRAL,
    },
    ("RIGHT", INDEX): {
        DIR_LEFT: RIGHT_INDEX_LEFT, DIR_RIGHT: RIGHT_INDEX_RIGHT, DIR_UP: RIGHT_INDEX_UP,
        DIR_DOWN: RIGHT_INDEX_DOWN, DIR_NEUTRAL: RIGHT_INDEX_NEUTRAL,
    },
}

# Identifiants mappables par main. L'ordre fixe l'affichage dans Parametres de geste.
SPECIAL_BY_SIDE = {
    "LEFT": (
        LEFT_PINCH, LEFT_FIST, LEFT_OPEN_PALM,
        LEFT_TILT_LEFT, LEFT_TILT_RIGHT, LEFT_TILT_UP, LEFT_TILT_DOWN, LEFT_TILT_NEUTRAL,
        LEFT_INDEX_LEFT, LEFT_INDEX_RIGHT, LEFT_INDEX_UP, LEFT_INDEX_DOWN, LEFT_INDEX_NEUTRAL,
        LEFT_FINGER_INDEX, LEFT_FINGER_MIDDLE, LEFT_FINGER_RING, LEFT_FINGER_PINKY, LEFT_FINGER_THUMB,
    ),
    "RIGHT": (
        RIGHT_PINCH, RIGHT_FIST, RIGHT_OPEN_PALM,
        RIGHT_TILT_LEFT, RIGHT_TILT_RIGHT, RIGHT_TILT_UP, RIGHT_TILT_DOWN, RIGHT_TILT_NEUTRAL,
        RIGHT_INDEX_LEFT, RIGHT_INDEX_RIGHT, RIGHT_INDEX_UP, RIGHT_INDEX_DOWN, RIGHT_INDEX_NEUTRAL,
        RIGHT_FINGER_INDEX, RIGHT_FINGER_MIDDLE, RIGHT_FINGER_RING, RIGHT_FINGER_PINKY, RIGHT_FINGER_THUMB,
    ),
}

_PINCH_IDS = {"LEFT": LEFT_PINCH, "RIGHT": RIGHT_PINCH}
_FIST_IDS = {"LEFT": LEFT_FIST, "RIGHT": RIGHT_FIST}
_PALM_IDS = {"LEFT": LEFT_OPEN_PALM, "RIGHT": RIGHT_OPEN_PALM}
_FINGER_UP_IDS = {
    ("LEFT", "INDEX"): LEFT_FINGER_INDEX,
    ("LEFT", "MIDDLE"): LEFT_FINGER_MIDDLE,
    ("LEFT", "RING"): LEFT_FINGER_RING,
    ("LEFT", "PINKY"): LEFT_FINGER_PINKY,
    ("LEFT", "THUMB"): LEFT_FINGER_THUMB,
    ("RIGHT", "INDEX"): RIGHT_FINGER_INDEX,
    ("RIGHT", "MIDDLE"): RIGHT_FINGER_MIDDLE,
    ("RIGHT", "RING"): RIGHT_FINGER_RING,
    ("RIGHT", "PINKY"): RIGHT_FINGER_PINKY,
    ("RIGHT", "THUMB"): RIGHT_FINGER_THUMB,
}
_FINGER_FAMILY = {
    "INDEX": FINGER_INDEX,
    "MIDDLE": FINGER_MIDDLE,
    "RING": FINGER_RING,
    "PINKY": FINGER_PINKY,
    "THUMB": FINGER_THUMB,
}
_FINGER_PAIRS = (
    (INDEX_TIP, INDEX_MCP),
    (MIDDLE_TIP, MIDDLE_MCP),
    (RING_TIP, RING_MCP),
    (PINKY_TIP, PINKY_MCP),
)
SPECIAL_NAMES = tuple(name for side in SIDES for name in SPECIAL_BY_SIDE[side])

# Phases exposees par le detecteur de pinch. Le mapping n'en a pas besoin
# (PRESS se declenche sur START, HOLD suit l'etat), mais elles rendent le
# cycle explicite pour le debug et les tests.
PHASE_START = "PINCH_START"
PHASE_HOLD = "PINCH_HOLD"
PHASE_END = "PINCH_END"


def special_source(side, family):
    """Nom de source de maintien reserve a une famille de gestes d'une main.

    Distinct de "LEFT"/"RIGHT" (gestes appris) pour que input_controller
    compte les maintiens separement : une meme touche demandee par G01 et par
    un detecteur geometrique n'est relachee que par sa derniere source.
    """
    return f"GEO:{side}:{family}"


def side_of(gesture):
    """Main proprietaire d'un identifiant special, ou None."""
    for side in SIDES:
        if gesture in SPECIAL_BY_SIDE[side]:
            return side
    return None


# =========================================================================
# 5. Mesures par main
# =========================================================================
class HandMetrics:
    """Toutes les mesures geometriques d'une main pour une frame.

    scale           : taille de reference de la main (hand_scale)
    pinch_distance  : distance brute pouce-index, unite des landmarks
    pinch_ratio     : pinch_distance / scale, insensible a la distance camera
    palm_vector     : (p9 - p0) / scale, magnitude 1 par construction
    index_vector    : (p8 - p5) / scale, magnitude ~1 tendu, ~0.3 replie
    direction       : fusion ponderee paume + index (voir PALM_WEIGHT)

    Les trois angles (palm_angle, index_angle, agreement) ne servent qu'a
    l'affichage et au diagnostic : ils sont donc calcules A LA DEMANDE et non
    a chaque frame. Les detecteurs, eux, travaillent directement sur les
    vecteurs, ce qui evite deux atan2 et un arccos par main par frame.
    """

    __slots__ = ("scale", "pinch_distance", "pinch_ratio", "palm_vector",
                 "index_vector", "direction", "finger_ratios", "direction_vector")

    def __init__(self, scale, pinch_distance, pinch_ratio, palm_vector,
                 index_vector, direction, finger_ratios, direction_vector=None):
        self.scale = scale
        self.pinch_distance = pinch_distance
        self.pinch_ratio = pinch_ratio
        self.palm_vector = palm_vector
        self.index_vector = index_vector
        self.direction = direction
        self.finger_ratios = finger_ratios
        self.direction_vector = index_vector if direction_vector is None else direction_vector

    @property
    def palm_angle(self):
        """Orientation absolue de la paume, en degres (atan2)."""
        return vector_angle(self.palm_vector)

    @property
    def index_angle(self):
        """Orientation absolue de l'index, en degres (atan2)."""
        return vector_angle(self.index_vector)

    @property
    def direction_angle(self):
        """Orientation de la direction fusionnee, en degres."""
        return vector_angle(self.direction)

    @property
    def agreement(self):
        """Angle entre paume et index, en degres. 0 = parfaitement alignes."""
        return angle_between_vectors(self.palm_vector, self.index_vector)


def measure_hand(landmarks, palm_weight=PALM_WEIGHT, index_weight=INDEX_WEIGHT, finger="INDEX"):
    """Mesure une main en une seule passe.

    Les operations portent sur 21 points : deux soustractions 2D et une norme.
    Rien de volumineux n'est alloue, seulement trois vecteurs de deux
    composantes.

    Convention d'axes : x croit vers la DROITE de l'image affichee et y vers
    le BAS. Comme main.py retourne la frame (cv2.flip) avant d'appeler
    MediaPipe, "droite" ici correspond bien a la droite que l'utilisateur voit
    a l'ecran. L'identite de la main (LEFT/RIGHT) reste calculee par
    GestureEngine.get_hand_side, donc SWAP_HANDS continue de fonctionner
    exactement comme avant : ce module ne la recalcule jamais.
    """
    scale = hand_scale(landmarks)

    axes = 3 if PINCH_USE_DEPTH else 2
    pinch_distance = float(np.linalg.norm(
        landmarks[THUMB_TIP][:axes] - landmarks[INDEX_TIP][:axes]))

    # Vecteurs 2D exprimes en unites de taille de main. landmarks est deja un
    # tableau NumPy, donc la soustraction directe evite les conversions.
    palm_vector = (landmarks[MIDDLE_MCP][:2] - landmarks[WRIST][:2]) / scale
    index_vector = (landmarks[INDEX_TIP][:2] - landmarks[INDEX_MCP][:2]) / scale
    segment = FINGER_SEGMENT.get(finger, FINGER_SEGMENT["INDEX"])
    direction_vector = (landmarks[segment[1]][:2] - landmarks[segment[0]][:2]) / scale

    # Fusion : la paume donne une direction toujours disponible, l'index
    # l'affine quand il est tendu. Le vecteur resultant garde une MAGNITUDE
    # variable (il raccourcit quand la main se referme), ce qui fournit
    # naturellement la zone morte du detecteur de direction.
    direction = palm_weight * palm_vector + index_weight * index_vector
    finger_ratios = tuple(
        float(np.linalg.norm(landmarks[tip][:2] - landmarks[mcp][:2])) / scale
        for tip, mcp in _FINGER_PAIRS
    )

    return HandMetrics(
        scale=scale,
        pinch_distance=pinch_distance,
        pinch_ratio=pinch_distance / scale,
        palm_vector=palm_vector,
        index_vector=index_vector,
        direction=direction,
        finger_ratios=finger_ratios,
        direction_vector=direction_vector,
    )


def finger_direction_vector(landmarks, finger, scale=None):
    """Image-space finger direction. +X right, +Y down. Not palm-relative."""
    finger = str(finger or "").upper()
    chain = FINGER_CHAIN.get(finger)
    if chain is None or landmarks is None or len(landmarks) < LANDMARK_COUNT:
        return np.zeros(2, dtype=np.float32)
    if scale is None:
        scale = hand_scale(landmarks)
    scale = max(float(scale), MIN_HAND_SCALE)
    pts = [landmarks[index][:2] for index in chain]
    if finger == "THUMB":
        vec = 0.20 * (pts[1] - pts[0]) + 0.30 * (pts[2] - pts[1]) + 0.50 * (pts[3] - pts[2])
    else:
        vec = 0.25 * (pts[1] - pts[0]) + 0.25 * (pts[2] - pts[1]) + 0.50 * (pts[3] - pts[2])
    return np.asarray(vec, dtype=np.float32) / scale


def finger_image_direction(vector, already_active=False):
    """Dominant image direction of a finger vector, or NEUTRAL."""
    dx = float(vector[0])
    dy = float(vector[1])
    margin = FINGER_UP_DIR_HOLD_MARGIN if already_active else FINGER_UP_DIR_MARGIN
    if abs(dx) > abs(dy) + margin:
        if dx > 0:
            return DIR_RIGHT
        if dx < 0:
            return DIR_LEFT
        return DIR_NEUTRAL
    if -dy > abs(dx) - margin and -dy > 0:
        return DIR_UP
    if dy > abs(dx) - margin and dy > 0:
        return DIR_DOWN
    return DIR_NEUTRAL


def is_finger_extended(landmarks, finger, metrics=None, on_threshold=FINGER_UP_ON_THRESHOLD):
    """True if the finger is stretched (not curled), regardless of image direction."""
    finger = str(finger or "").upper()
    joints = FINGER_JOINTS.get(finger)
    chain = FINGER_CHAIN.get(finger)
    if joints is None or landmarks is None or len(landmarks) < LANDMARK_COUNT:
        return False
    tip_i, pip_i, mcp_i = joints
    scale = metrics.scale if metrics is not None else hand_scale(landmarks)
    tip = landmarks[tip_i][:2]
    pip = landmarks[pip_i][:2]
    mcp = landmarks[mcp_i][:2]
    wrist_pt = landmarks[WRIST][:2]
    needed = on_threshold
    if finger == "THUMB":
        needed = max(0.35, float(on_threshold) * 0.65)
    extension = float(np.linalg.norm(landmarks[tip_i][:3] - landmarks[mcp_i][:3])) / scale
    if extension < needed:
        return False
    if float(np.linalg.norm(tip - wrist_pt)) <= float(np.linalg.norm(mcp - wrist_pt)) + 0.05 * scale:
        return False
    if chain is not None:
        pts = [landmarks[index][:2] for index in chain]
        for start, mid, end in zip(pts, pts[1:], pts[2:]):
            if float(np.linalg.norm(mid - start)) <= 0.05 * scale:
                continue
            if angle_between_vectors(mid - start, end - mid) > FINGER_UP_BEND_MAX:
                return False
    else:
        pip_len = float(np.linalg.norm(pip - mcp))
        if pip_len > 0.05 * scale:
            if angle_between_vectors(pip - mcp, tip - pip) > FINGER_UP_BEND_MAX:
                return False
    return True


def detect_finger_up(landmarks, finger, metrics=None, on_threshold=FINGER_UP_ON_THRESHOLD,
                     already_active=False):
    """True if that finger is extended and points UP in image coordinates.

    +Y is down. A bent or downward / horizontal finger is not UP.
    This is not INDEX LEFT/RIGHT/UP/DOWN (the separate pointing family).
    """
    if not is_finger_extended(landmarks, finger, metrics, on_threshold):
        return False
    scale = metrics.scale if metrics is not None else hand_scale(landmarks)
    vector = finger_direction_vector(landmarks, finger, scale)
    return finger_image_direction(vector, already_active=already_active) == DIR_UP


def finger_up_id(side, finger):
    return _FINGER_UP_IDS.get((side, str(finger or "").upper()))


def finger_up_family(finger):
    return _FINGER_FAMILY.get(str(finger or "").upper())


# =========================================================================
# 6. Detecteurs
# =========================================================================
class PinchDetector:
    """Pouce + index : deux etats avec hysteresis, et un cycle explicite.

        OFF --(ratio <= ON)--> ON --(ratio >= OFF)--> OFF

    Entre les deux seuils l'etat courant est conserve. Un pincement tenu ne
    produit donc qu'UNE transition : le mapping recoit un evenement, jamais un
    PRESS par frame.

    phase expose le cycle demande : PINCH_START a la fermeture, PINCH_HOLD
    tant que le pincement dure, PINCH_END a l'ouverture, None au repos.
    """

    family = PINCH

    def __init__(self, side, on_threshold=PINCH_ON_THRESHOLD,
                 off_threshold=PINCH_OFF_THRESHOLD,
                 stability_frames=PINCH_STABILITY_FRAMES):
        if not 0.0 < on_threshold < off_threshold:
            raise ValueError("il faut 0 < on_threshold < off_threshold pour avoir une hysteresis")
        self.side = side
        self.gesture_id = _PINCH_IDS[side]
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.stability_frames = max(1, int(stability_frames))
        self.active = False
        self.phase = None
        self._streak = 0

    def update(self, metrics):
        ratio = metrics.pinch_ratio
        # Actif : on ne sort qu'au-dessus de OFF. Inactif : on n'entre qu'en
        # dessous de ON. La bande entre les deux ne change rien.
        wanted = ratio < self.off_threshold if self.active else ratio <= self.on_threshold

        changed = False
        if wanted == self.active:
            self._streak = 0
        else:
            self._streak += 1
            if self._streak >= self.stability_frames:
                self.active = wanted
                self._streak = 0
                changed = True

        if self.active:
            self.phase = PHASE_START if changed else PHASE_HOLD
        else:
            self.phase = PHASE_END if changed else None
        return self.gesture_id if self.active else None

    def reset(self):
        self.active = False
        self.phase = None
        self._streak = 0

    def state_text(self):
        return "ON" if self.active else "OFF"


class LevelDetector:
    """Seuil sur une grandeur, avec hysteresis. mode 'low' ou 'high'.

    low  : actif quand valeur <= ON, relache quand valeur >= OFF (ON < OFF).
    high : actif quand valeur >= ON, relache quand valeur <= OFF (OFF < ON).
    """

    def __init__(self, side, family, gesture_id, mode, on_threshold, off_threshold,
                 stability_frames=STABILITY_FRAMES):
        if mode == "low" and not on_threshold < off_threshold:
            raise ValueError("poing : il faut on < off")
        if mode == "high" and not off_threshold < on_threshold:
            raise ValueError("paume : il faut off < on")
        self.side = side
        self.family = family
        self.gesture_id = gesture_id
        self.mode = mode
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.stability_frames = max(1, int(stability_frames))
        self.active = False
        self._streak = 0

    def _wants(self, value):
        if self.mode == "low":
            return value < self.off_threshold if self.active else value <= self.on_threshold
        return value > self.off_threshold if self.active else value >= self.on_threshold

    def update(self, metrics):
        ratios = metrics.finger_ratios
        value = max(ratios) if self.mode == "low" else min(ratios)
        wanted = self._wants(value)
        if wanted == self.active:
            self._streak = 0
        else:
            self._streak += 1
            if self._streak >= self.stability_frames:
                self.active = wanted
                self._streak = 0
        return self.gesture_id if self.active else None

    def reset(self):
        self.active = False
        self._streak = 0

    def state_text(self):
        return "ON" if self.active else "OFF"


class DirectionDetector:
    """Direction 2D en quatre etats plus un neutre, avec hysteresis.

    La decision porte sur un vecteur exprime en unites de taille de main, donc
    les seuils ne dependent pas de la distance a la camera.

    Convention image (MediaPipe, frame deja retournee) :
        +X = droite, -X = gauche, +Y = bas, -Y = haut.

    Axe dominant : |dx| vs |dy| (egalite -> vertical), puis le signe, puis
    le seuil de cette direction. Une diagonale n'active pas l'axe faible.

    Deux garanties de stabilite :
      - hysteresis : on entre a THRESHOLD, on ne revient au neutre qu'en
        dessous de THRESHOLD - DEAD_ZONE ;
      - tout changement entre deux directions passe OBLIGATOIREMENT par
        NEUTRAL, donc l'action maintenue est relachee avant que la suivante ne
        soit appliquee. A et D ne peuvent jamais etre enfoncees ensemble a
        cause d'une transition.
    """

    def __init__(self, side, family, thresholds=None, dead_zone=DEAD_ZONE,
                 stability_frames=STABILITY_FRAMES):
        self.side = side
        self.family = family
        self.thresholds = dict(thresholds or {})
        for direction in (DIR_LEFT, DIR_RIGHT, DIR_UP, DIR_DOWN):
            if self.thresholds.get(direction) is None:
                raise ValueError(f"seuil manquant pour {direction}")
        self.dead_zone = float(dead_zone)
        self.stability_frames = max(1, int(stability_frames))
        self.ids = _DIRECTION_IDS[(side, family)]
        self.direction = DIR_NEUTRAL
        self._streak = 0

    def _threshold(self, direction):
        """Seuil d'entree, ou seuil de sortie (reduit) si deja dans cet etat."""
        base = self.thresholds[direction]
        if self.direction == direction:
            return max(base - self.dead_zone, 0.0)
        return base

    def _candidate(self, vector):
        dx = float(vector[0])
        dy = float(vector[1])
        if abs(dx) > abs(dy):
            if dx >= self._threshold(DIR_RIGHT):
                return DIR_RIGHT
            if -dx >= self._threshold(DIR_LEFT):
                return DIR_LEFT
            return DIR_NEUTRAL
        if -dy >= self._threshold(DIR_UP):
            return DIR_UP
        if dy >= self._threshold(DIR_DOWN):
            return DIR_DOWN
        return DIR_NEUTRAL

    def update_vector(self, vector):
        candidate = self._candidate(vector)
        # Deux directions opposees ne s'enchainent jamais directement : on
        # repasse par le neutre pour relacher avant d'activer.
        if candidate != self.direction and candidate != DIR_NEUTRAL and self.direction != DIR_NEUTRAL:
            candidate = DIR_NEUTRAL

        if candidate == self.direction:
            self._streak = 0
        else:
            self._streak += 1
            if self._streak >= self.stability_frames:
                self.direction = candidate
                self._streak = 0
        return self.ids[self.direction]

    def update(self, metrics):
        if self.family == TILT:
            vector = metrics.direction
        else:
            vector = getattr(metrics, "direction_vector", metrics.index_vector)
        return self.update_vector(vector)

    def reset(self):
        self.direction = DIR_NEUTRAL
        self._streak = 0

    def state_text(self):
        return self.direction


class FingerSlotDetector:
    """One raised finger. Shares detect_finger_up(); no k-NN samples."""

    def __init__(self, side, finger, on_threshold=FINGER_UP_ON_THRESHOLD,
                 off_threshold=FINGER_UP_OFF_THRESHOLD,
                 stability_frames=STABILITY_FRAMES):
        finger = str(finger or "INDEX").upper()
        if finger not in FINGER_UP_FINGERS:
            raise ValueError(f"doigt Finger Up inconnu : {finger}")
        self.side = side
        self.finger = finger
        self.family = finger_up_family(finger)
        self.gesture_id = finger_up_id(side, finger)
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.stability_frames = max(1, int(stability_frames))
        self.active = False
        self._streak = 0

    def update_wanted(self, wanted):
        if wanted == self.active:
            self._streak = 0
        else:
            self._streak += 1
            if self._streak >= self.stability_frames:
                self.active = wanted
                self._streak = 0
        return self.gesture_id if self.active else None

    def reset(self):
        self.active = False
        self._streak = 0

    def state_text(self):
        return "ON" if self.active else "OFF"


def _tilt_thresholds():
    return {
        DIR_LEFT: TILT_LEFT_THRESHOLD,
        DIR_RIGHT: TILT_RIGHT_THRESHOLD,
        DIR_UP: TILT_UP_THRESHOLD,
        DIR_DOWN: TILT_DOWN_THRESHOLD,
    }


def _index_thresholds():
    return {direction: INDEX_DIRECTION_THRESHOLD
            for direction in (DIR_LEFT, DIR_RIGHT, DIR_UP, DIR_DOWN)}


# =========================================================================
# 7. Moteur
# =========================================================================
class HandGeometry:
    """Etat geometrique d'UNE main : ses detecteurs et ses dernieres mesures."""

    def __init__(self, side, **overrides):
        self.side = side
        self.metrics = None
        self.finger = overrides.get("finger", "INDEX")
        if self.finger not in FINGERS:
            self.finger = "INDEX"
        mode = str(overrides.get("finger_up_mode", FINGER_UP_MODE_SINGLE)).strip().lower()
        self.finger_up_mode = mode if mode in (FINGER_UP_MODE_SINGLE, FINGER_UP_MODE_MULTI) else FINGER_UP_MODE_SINGLE
        self.palm_weight = float(overrides.get("palm_weight", PALM_WEIGHT))
        self.index_weight = float(overrides.get("index_weight", INDEX_WEIGHT))
        self.detectors = self._build_detectors(side, overrides)
        self.states = {family: None for family in self.detectors}

    @staticmethod
    def _build_detectors(side, overrides):
        """Detecteurs actifs pour cette main.

        Le pinch existe sur les deux mains, avec un identifiant par main.
        Poing et paume ouverte sont optionnels : desactives, ils n'existent
        pas et ne peuvent donc rien declencher.
        """
        detectors = {
            PINCH: PinchDetector(
                side,
                on_threshold=overrides.get("pinch_on", PINCH_ON_THRESHOLD),
                off_threshold=overrides.get("pinch_off", PINCH_OFF_THRESHOLD),
                stability_frames=overrides.get("pinch_stability_frames", PINCH_STABILITY_FRAMES),
            ),
        }
        stable = overrides.get("stability_frames", STABILITY_FRAMES)
        if overrides.get("fist_enabled", False):
            detectors[FIST] = LevelDetector(
                side, FIST, _FIST_IDS[side], "low",
                overrides.get("fist_on", FIST_ON_THRESHOLD),
                overrides.get("fist_off", FIST_OFF_THRESHOLD),
                stable,
            )
        if overrides.get("open_palm_enabled", False):
            detectors[PALM] = LevelDetector(
                side, PALM, _PALM_IDS[side], "high",
                overrides.get("open_palm_on", OPEN_PALM_ON_THRESHOLD),
                overrides.get("open_palm_off", OPEN_PALM_OFF_THRESHOLD),
                stable,
            )
        detectors[TILT] = DirectionDetector(
            side, TILT,
            thresholds=overrides.get("tilt_thresholds") or _tilt_thresholds(),
            dead_zone=overrides.get("dead_zone", DEAD_ZONE),
            stability_frames=stable,
        )
        detectors[INDEX] = DirectionDetector(
            side, INDEX,
            thresholds=overrides.get("index_thresholds") or _index_thresholds(),
            dead_zone=overrides.get("dead_zone", DEAD_ZONE),
            stability_frames=stable,
        )
        for finger in FINGER_UP_FINGERS:
            family = finger_up_family(finger)
            detectors[family] = FingerSlotDetector(
                side, finger,
                on_threshold=overrides.get("finger_up_on", FINGER_UP_ON_THRESHOLD),
                off_threshold=overrides.get("finger_up_off", FINGER_UP_OFF_THRESHOLD),
                stability_frames=stable,
            )
        return detectors

    def update(self, landmarks):
        self.metrics = measure_hand(landmarks, self.palm_weight, self.index_weight, self.finger)
        raw_up = {}
        for finger in FINGER_UP_FINGERS:
            slot = self.detectors[finger_up_family(finger)]
            threshold = slot.off_threshold if slot.active else slot.on_threshold
            raw_up[finger] = detect_finger_up(
                landmarks, finger, self.metrics, threshold, already_active=slot.active,
            )
        if self.finger_up_mode == FINGER_UP_MODE_SINGLE:
            raised = [finger for finger in FINGER_UP_FINGERS if finger != "THUMB" and raw_up[finger]]
            if len(raised) > 1:
                for finger in FINGER_UP_FINGERS:
                    if finger != "THUMB":
                        raw_up[finger] = False
        for family, detector in self.detectors.items():
            if isinstance(detector, FingerSlotDetector):
                self.states[family] = detector.update_wanted(raw_up[detector.finger])
            else:
                self.states[family] = detector.update(self.metrics)
        return self.states

    def reset(self):
        """Main perdue ou arret global : remet a zero CETTE main uniquement."""
        self.metrics = None
        for family, detector in self.detectors.items():
            detector.reset()
            self.states[family] = None

    def sources(self):
        return [special_source(self.side, family) for family in self.detectors]


class GeometryEngine:
    """Les deux mains, strictement independantes.

    Chaque main possede son propre etat : perdre la main droite ne touche pas
    l'inclinaison de la main gauche, et inversement. update() ne renvoie que
    des identifiants, il n'envoie jamais de commande.
    """

    def __init__(self, direction_fingers=None, **overrides):
        fingers = direction_fingers or {}
        self.hands = {
            side: HandGeometry(side, finger=fingers.get(side, "INDEX"), **overrides)
            for side in SIDES
        }

    def update(self, side, landmarks):
        """Etats {famille: identifiant ou None} pour cette main.

        landmarks : (21, 3) bruts ou features normalisees (voir
        points_from_features). Les deux conviennent, toutes les mesures sont
        relatives a la taille de la main.
        """
        hand = self.hands.get(side)
        if hand is None:
            return {}
        if landmarks is None or len(landmarks) < LANDMARK_COUNT:
            return self.hand_lost(side)
        return hand.update(landmarks)

    def hand_lost(self, side):
        """Main perdue : remet SES detecteurs a zero, jamais ceux de l'autre."""
        hand = self.hands.get(side)
        if hand is None:
            return {}
        hand.reset()
        return hand.states

    def reset(self):
        """Remise a zero globale (preview, INPUT OFF, entree/sortie de menu)."""
        for hand in self.hands.values():
            hand.reset()

    def states(self, side):
        hand = self.hands.get(side)
        return dict(hand.states) if hand else {}

    def metrics(self, side):
        hand = self.hands.get(side)
        return hand.metrics if hand else None

    def detector(self, side, family):
        hand = self.hands.get(side)
        return hand.detectors.get(family) if hand else None

    def families(self, side):
        """Familles de detecteurs actives pour cette main."""
        hand = self.hands.get(side)
        return tuple(hand.detectors) if hand else ()

    def sources(self, side):
        hand = self.hands.get(side)
        return hand.sources() if hand else []

    def active(self, side):
        """Identifiants reellement actifs pour cette main (neutres compris)."""
        return [gesture for gesture in self.states(side).values() if gesture]

    # ---- Debug ----------------------------------------------------------
    def debug_lines(self):
        """Valeurs numeriques et seuils. Uniquement pour le mode DEBUG."""
        lines = []
        for side in SIDES:
            hand = self.hands[side]
            metrics = hand.metrics
            if metrics is None:
                lines.append(f"{side:<5} main absente")
                continue
            tilt = hand.detectors[TILT]
            index = hand.detectors[INDEX]
            lines.append(
                f"{side:<5} orient {metrics.palm_angle:+7.1f}deg  index {metrics.index_angle:+7.1f}deg"
                f"  accord {metrics.agreement:5.1f}deg"
            )
            lines.append(
                f"      dir ({metrics.direction[0]:+.2f},{metrics.direction[1]:+.2f})"
                f" tilt {tilt.state_text():<7}"
                f" | idx ({metrics.index_vector[0]:+.2f},{metrics.index_vector[1]:+.2f})"
                f" {index.state_text()}"
            )
            pinch = hand.detectors.get(PINCH)
            if pinch is not None:
                lines.append(
                    f"      pinch {metrics.pinch_ratio:.3f} (d {metrics.pinch_distance:.3f}"
                    f" / main {metrics.scale:.3f}) {pinch.state_text()}"
                    f" phase {pinch.phase or '-'}"
                    f"  on {pinch.on_threshold:.2f} off {pinch.off_threshold:.2f}"
                )
            up = [
                detector.finger for detector in (
                    hand.detectors.get(family) for family in FINGER_FAMILIES
                )
                if detector is not None and detector.active
            ]
            lines.append(
                f"      finger {' '.join(up) if up else 'NONE'}  mode {hand.finger_up_mode}"
            )
            for family, label in ((FIST, "poing"), (PALM, "paume")):
                detector = hand.detectors.get(family)
                if detector is None or metrics.finger_ratios is None:
                    continue
                value = max(metrics.finger_ratios) if family == FIST else min(metrics.finger_ratios)
                lines.append(
                    f"      {label} {value:.2f} {detector.state_text()}"
                    f"  on {detector.on_threshold:.2f} off {detector.off_threshold:.2f}"
                )
        sample = self.hands["LEFT"].detectors[TILT]
        lines.append(
            f"seuils tilt L{sample.thresholds[DIR_LEFT]:.2f} R{sample.thresholds[DIR_RIGHT]:.2f}"
            f" U{sample.thresholds[DIR_UP]:.2f} D{sample.thresholds[DIR_DOWN]:.2f}"
            f" | zone morte {sample.dead_zone:.2f} | stab {sample.stability_frames}"
        )
        return lines


def _enabled_flag(value):
    """False par defaut. La chaine 'false' ne doit pas valoir True."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(value, (int, float)):
        return value != 0
    return False


def _median(values):
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if count == 0:
        return None
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _pair(low_name, high_name, low, high, fallback_low, fallback_high, low_first):
    """Garde une hysteresis valide, sinon les constantes d'usine."""
    try:
        low = float(low)
        high = float(high)
    except (TypeError, ValueError):
        return fallback_low, fallback_high
    ok = low < high if low_first else high < low
    if not ok:
        return fallback_low, fallback_high
    return low, high


def overrides_from_settings(settings):
    """Traduit settings.json vers les arguments de GeometryEngine.

    Un reglages absent ou incoherent retombe sur la constante du module.
    N'ouvre aucun fichier et ne touche pas aux gestes appris.
    """
    settings = settings if isinstance(settings, dict) else {}

    def number(key, fallback):
        try:
            return float(settings.get(key, fallback))
        except (TypeError, ValueError):
            return float(fallback)

    pinch_on, pinch_off = _pair(
        "on", "off",
        number("pinch_on_threshold", PINCH_ON_THRESHOLD),
        number("pinch_off_threshold", PINCH_OFF_THRESHOLD),
        PINCH_ON_THRESHOLD, PINCH_OFF_THRESHOLD, True,
    )
    fist_on, fist_off = _pair(
        "on", "off",
        number("fist_on_threshold", FIST_ON_THRESHOLD),
        number("fist_off_threshold", FIST_OFF_THRESHOLD),
        FIST_ON_THRESHOLD, FIST_OFF_THRESHOLD, True,
    )
    palm_off, palm_on = _pair(
        "off", "on",
        number("open_palm_off_threshold", OPEN_PALM_OFF_THRESHOLD),
        number("open_palm_on_threshold", OPEN_PALM_ON_THRESHOLD),
        OPEN_PALM_OFF_THRESHOLD, OPEN_PALM_ON_THRESHOLD, True,
    )
    try:
        stable = int(settings.get("geometry_stability_frames", STABILITY_FRAMES))
    except (TypeError, ValueError):
        stable = STABILITY_FRAMES
    stable = max(1, min(10, stable))
    index_threshold = number("index_direction_threshold", INDEX_DIRECTION_THRESHOLD)
    return {
        "pinch_on": pinch_on,
        "pinch_off": pinch_off,
        "tilt_thresholds": {
            DIR_LEFT: number("tilt_left_threshold", TILT_LEFT_THRESHOLD),
            DIR_RIGHT: number("tilt_right_threshold", TILT_RIGHT_THRESHOLD),
            DIR_UP: number("tilt_up_threshold", TILT_UP_THRESHOLD),
            DIR_DOWN: number("tilt_down_threshold", TILT_DOWN_THRESHOLD),
        },
        "index_thresholds": {direction: index_threshold for direction in (DIR_LEFT, DIR_RIGHT, DIR_UP, DIR_DOWN)},
        "dead_zone": number("dead_zone", DEAD_ZONE),
        "stability_frames": stable,
        "palm_weight": number("palm_weight", PALM_WEIGHT),
        "index_weight": number("index_weight", INDEX_WEIGHT),
        "fist_enabled": _enabled_flag(settings.get("fist_enabled", False)),
        "open_palm_enabled": _enabled_flag(settings.get("open_palm_enabled", False)),
        "fist_on": fist_on,
        "fist_off": fist_off,
        "open_palm_on": palm_on,
        "open_palm_off": palm_off,
        "finger_up_on": number("finger_up_on_threshold", FINGER_UP_ON_THRESHOLD),
        "finger_up_off": number("finger_up_off_threshold", FINGER_UP_OFF_THRESHOLD),
        "finger_up_mode": str(settings.get("finger_up_mode", FINGER_UP_MODE_SINGLE)).strip().lower(),
    }


def suggest_geometry_thresholds(observations):
    """Propose des seuils a partir de poses mesurees. Ne sauvegarde rien.

    observations : pose -> liste de dicts {pinch_ratio, dx, dy}.
    Poses requises : neutral, pinch, tilt_left, tilt_right.
    tilt_up / tilt_down sont optionnels. Si une pose ne se distingue pas
    du neutre, le seuil d'usine est conserve pour cet axe.
    """
    required = ("neutral", "pinch", "tilt_left", "tilt_right")
    for name in required:
        if not observations.get(name):
            raise ValueError(f"pose manquante : {name}")

    def column(pose, key):
        return [sample[key] for sample in observations[pose]]

    neutral_pinch = _median(column("neutral", "pinch_ratio"))
    closed_pinch = _median(column("pinch", "pinch_ratio"))
    proposed = {}
    if closed_pinch is not None and neutral_pinch - closed_pinch >= 0.08:
        gap = neutral_pinch - closed_pinch
        proposed["pinch_on_threshold"] = round(closed_pinch + 0.35 * gap, 3)
        proposed["pinch_off_threshold"] = round(closed_pinch + 0.65 * gap, 3)
    else:
        proposed["pinch_on_threshold"] = PINCH_ON_THRESHOLD
        proposed["pinch_off_threshold"] = PINCH_OFF_THRESHOLD

    def tilt_threshold(pose, sign, component, fallback):
        if not observations.get(pose):
            return fallback
        neutral = _median(abs(sample[component]) for sample in observations["neutral"])
        observed = _median(sign * sample[component] for sample in observations[pose])
        if observed is None or neutral is None or observed - neutral < 0.08:
            return fallback
        return round(max((neutral + observed) / 2.0, DEAD_ZONE + 0.05), 3)

    proposed["tilt_left_threshold"] = tilt_threshold("tilt_left", -1.0, "dx", TILT_LEFT_THRESHOLD)
    proposed["tilt_right_threshold"] = tilt_threshold("tilt_right", 1.0, "dx", TILT_RIGHT_THRESHOLD)
    if observations.get("tilt_up"):
        proposed["tilt_up_threshold"] = tilt_threshold("tilt_up", -1.0, "dy", TILT_UP_THRESHOLD)
    if observations.get("tilt_down"):
        proposed["tilt_down_threshold"] = tilt_threshold("tilt_down", 1.0, "dy", TILT_DOWN_THRESHOLD)
    return proposed
