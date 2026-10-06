"""HandController — single source of version, paths and tunable settings.

Runtime values that users may change live in settings.json next to the
program. Engine/mapping defaults stay in their own modules so existing
tests keep working if they construct GestureEngine without this file.
"""
import json
import logging
import os
import shutil
import sys
from logging.handlers import RotatingFileHandler

import geometry_engine as geo

APP_NAME = "HandController"
APP_VERSION = "1.0.0"

GESTURES_FILE = "gestures.json"
MAPPING_FILE = "mapping.json"
SETTINGS_FILE = "settings.json"
MODEL_FILE = "hand_landmarker.task"
LOG_DIR_NAME = "logs"
LOG_FILE_NAME = "handcontroller.log"

# Defaults. settings.json may override the user-facing subset.
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_RESOLUTIONS = (
    (640, 480),
    (1280, 720),
    (1920, 1080),
)


def camera_resolution_label(width, height):
    return f"{int(width)} \u00d7 {int(height)}"


def nearest_camera_preset(width, height):
    """Closest listed preset. Unknown sizes stay valid; UI still offers presets."""
    try:
        width = int(width)
        height = int(height)
    except (TypeError, ValueError):
        return CAMERA_RESOLUTIONS[1]
    exact = (width, height)
    if exact in CAMERA_RESOLUTIONS:
        return exact
    best = CAMERA_RESOLUTIONS[1]
    best_dist = None
    for preset in CAMERA_RESOLUTIONS:
        dist = abs(preset[0] - width) + abs(preset[1] - height)
        if best_dist is None or dist < best_dist:
            best = preset
            best_dist = dist
    return best


def accept_camera_size(actual_width, actual_height):
    """True if the driver returned a usable frame size."""
    try:
        return int(actual_width) >= 160 and int(actual_height) >= 120
    except (TypeError, ValueError):
        return False


def camera_size_matches(actual_width, actual_height, requested_width, requested_height, tolerance=48):
    """True if the driver size is close enough to the requested preset."""
    try:
        return (
            abs(int(actual_width) - int(requested_width)) <= int(tolerance)
            and abs(int(actual_height) - int(requested_height)) <= int(tolerance)
        )
    except (TypeError, ValueError):
        return False


def camera_try_order(width, height):
    """Requested size first, then remaining presets from closest to farthest."""
    requested = (int(width), int(height))
    ordered = []
    seen = set()
    for item in (requested,) + CAMERA_RESOLUTIONS:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
MAX_HANDS = 2
KNN_K = 5
CONFIDENCE_THRESHOLD = 0.55
CLASS_MARGIN = 0.12
STABLE_FRAMES = 2
HIGH_CONFIDENCE_THRESHOLD = 0.85
UNKNOWN_FRAMES = 2
SAMPLES_PER_GESTURE = 30
COUNTDOWN_SECONDS = 3
DEBUG = False
DISPLAY_PREVIEW = True
# Frames consecutives sans main avant de relacher SES holds. 0 = comportement
# historique (release immediat). 2 absorbe un micro-trou MediaPipe sans
# retarder un vrai retrait de la main.
HAND_LOSS_GRACE_FRAMES = 2
# Legacy names kept so older settings.json still load. They are not UI shortcuts.
EMERGENCY_KEY = "F11"
PREVIEW_KEY = "F10"
SMOOTHING_ENABLED = True
# Rest: more smoothing. Motion: beta still follows a real swipe.
SMOOTHING_MIN_CUTOFF = 1.2
SMOOTHING_BETA = 0.45
SMOOTHING_D_CUTOFF = 1.15

_USER_KEYS = (
    "camera_index",
    "debug",
    "confidence_threshold",
    "stable_frames",
    "knn_k",
    "knn_distance_margin",
    "countdown_seconds",
    "camera_width",
    "camera_height",
    "display_preview",
    "hand_loss_grace_frames",
    "pinch_on_threshold",
    "pinch_off_threshold",
    "tilt_left_threshold",
    "tilt_right_threshold",
    "tilt_up_threshold",
    "tilt_down_threshold",
    "dead_zone",
    "index_direction_threshold",
    "geometry_stability_frames",
    "palm_weight",
    "index_weight",
    "fist_on_threshold",
    "fist_off_threshold",
    "open_palm_on_threshold",
    "open_palm_off_threshold",
    "fist_enabled",
    "open_palm_enabled",
    "finger_up_on_threshold",
    "finger_up_off_threshold",
    "finger_up_mode",
    "emergency_key",
    "landmark_smoothing",
    "smoothing_min_cutoff",
    "smoothing_beta",
    "smoothing_d_cutoff",
    "direction_finger_left",
    "direction_finger_right",
    "gaming_mode",
    "language",  # "fr" or "en". Empty until the first-launch choice is saved.
    "system_gestures",
    "ml_backend",
    "ensemble_mode",
    "knn_weight",
    "svm_weight",
    "random_forest_weight",
    "show_landmarks",
    "hud_overlay",
)

_log = logging.getLogger("handcontroller")
_configured = False


def app_dir():
    """Directory that holds user data and, for a source run, the scripts.

    Frozen EXE: folder of HandController.exe (not the temp unpack dir).
    Source: folder of this file. Independent of the process working directory.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(name):
    """Bundled resource (model, localization). PyInstaller unpack dir first, then app_dir."""
    if getattr(sys, "frozen", False):
        bundled = os.path.join(getattr(sys, "_MEIPASS", ""), name)
        if os.path.isfile(bundled):
            return bundled
    return os.path.join(app_dir(), name)


def user_data_path(name):
    return os.path.join(app_dir(), name)


def log_dir():
    return os.path.join(app_dir(), LOG_DIR_NAME)


def default_settings():
    return {
        "camera_index": CAMERA_INDEX,
        "debug": DEBUG,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "stable_frames": STABLE_FRAMES,
        "knn_k": KNN_K,
        "knn_distance_margin": 0.04,
        "countdown_seconds": COUNTDOWN_SECONDS,
        "camera_width": CAMERA_WIDTH,
        "camera_height": CAMERA_HEIGHT,
        "display_preview": DISPLAY_PREVIEW,
        "hand_loss_grace_frames": HAND_LOSS_GRACE_FRAMES,
        "pinch_on_threshold": geo.PINCH_ON_THRESHOLD,
        "pinch_off_threshold": geo.PINCH_OFF_THRESHOLD,
        "tilt_left_threshold": geo.TILT_LEFT_THRESHOLD,
        "tilt_right_threshold": geo.TILT_RIGHT_THRESHOLD,
        "tilt_up_threshold": geo.TILT_UP_THRESHOLD,
        "tilt_down_threshold": geo.TILT_DOWN_THRESHOLD,
        "dead_zone": geo.DEAD_ZONE,
        "index_direction_threshold": geo.INDEX_DIRECTION_THRESHOLD,
        "geometry_stability_frames": geo.STABILITY_FRAMES,
        "palm_weight": geo.PALM_WEIGHT,
        "index_weight": geo.INDEX_WEIGHT,
        "fist_on_threshold": geo.FIST_ON_THRESHOLD,
        "fist_off_threshold": geo.FIST_OFF_THRESHOLD,
        "open_palm_on_threshold": geo.OPEN_PALM_ON_THRESHOLD,
        "open_palm_off_threshold": geo.OPEN_PALM_OFF_THRESHOLD,
        "fist_enabled": False,
        "open_palm_enabled": False,
        "finger_up_on_threshold": geo.FINGER_UP_ON_THRESHOLD,
        "finger_up_off_threshold": geo.FINGER_UP_OFF_THRESHOLD,
        "finger_up_mode": geo.FINGER_UP_MODE_SINGLE,
        "emergency_key": EMERGENCY_KEY,
        "landmark_smoothing": SMOOTHING_ENABLED,
        "smoothing_min_cutoff": SMOOTHING_MIN_CUTOFF,
        "smoothing_beta": SMOOTHING_BETA,
        "smoothing_d_cutoff": SMOOTHING_D_CUTOFF,
        "direction_finger_left": "INDEX",
        "direction_finger_right": "INDEX",
        "gaming_mode": False,
        "language": "",
        "system_gestures": {},
        "ml_backend": "knn",
        "ensemble_mode": "weighted",
        "knn_weight": 1.0,
        "svm_weight": 1.0,
        "random_forest_weight": 1.0,
        "show_landmarks": True,
        "hud_overlay": False,
    }


def backup_file(path):
    """Keep a single recent .bak next to path. No-op if the file is missing."""
    if not path or not os.path.isfile(path):
        return False
    bak = path + ".bak"
    try:
        shutil.copy2(path, bak)
        return True
    except OSError as error:
        _log.warning("Could not backup %s: %s", path, error)
        return False


def atomic_json_write(path, payload):
    """Write JSON via a temp file, then replace. A crash cannot leave a half file."""
    import json
    import tempfile
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, delete=False, suffix=".tmp")
    tmp_path = handle.name
    try:
        json.dump(payload, handle, indent=4, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(tmp_path, path)
    except Exception:
        handle.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _clamp_int(value, minimum, maximum, fallback):
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, number))


def _clamp_float(value, minimum, maximum, fallback):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number != number or number in (float("inf"), float("-inf")):
        return fallback
    return max(minimum, min(maximum, number))


def _as_bool(value, fallback):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        return fallback
    if isinstance(value, (int, float)):
        return value != 0
    return fallback


def validate_settings(raw):
    """Return a clean settings dict. Unknown keys are ignored."""
    defaults = default_settings()
    if not isinstance(raw, dict):
        return defaults
    out = dict(defaults)
    if "camera_index" in raw:
        out["camera_index"] = _clamp_int(raw["camera_index"], 0, 9, defaults["camera_index"])
    if "debug" in raw:
        out["debug"] = bool(raw["debug"])
    if "confidence_threshold" in raw:
        out["confidence_threshold"] = _clamp_float(
            raw["confidence_threshold"], 0.05, 0.95, defaults["confidence_threshold"]
        )
    if "stable_frames" in raw:
        out["stable_frames"] = _clamp_int(raw["stable_frames"], 1, 10, defaults["stable_frames"])
    if "knn_k" in raw:
        out["knn_k"] = _clamp_int(raw["knn_k"], 1, 15, defaults["knn_k"])
    if "knn_distance_margin" in raw:
        out["knn_distance_margin"] = _clamp_float(
            raw["knn_distance_margin"], 0.0, 5.0, defaults["knn_distance_margin"]
        )
    if "countdown_seconds" in raw:
        out["countdown_seconds"] = _clamp_int(
            raw["countdown_seconds"], 0, 10, defaults["countdown_seconds"]
        )
    if "camera_width" in raw:
        out["camera_width"] = _clamp_int(raw["camera_width"], 160, 1920, defaults["camera_width"])
    if "camera_height" in raw:
        out["camera_height"] = _clamp_int(raw["camera_height"], 120, 1080, defaults["camera_height"])
    if "camera" in raw and isinstance(raw["camera"], dict):
        nested = raw["camera"]
        if "width" in nested:
            out["camera_width"] = _clamp_int(nested["width"], 160, 1920, out["camera_width"])
        if "height" in nested:
            out["camera_height"] = _clamp_int(nested["height"], 120, 1080, out["camera_height"])
    if "display_preview" in raw:
        out["display_preview"] = bool(raw["display_preview"])
    if "hand_loss_grace_frames" in raw:
        out["hand_loss_grace_frames"] = _clamp_int(
            raw["hand_loss_grace_frames"], 0, 10, defaults["hand_loss_grace_frames"]
        )
    for key, low, high in (
        ("pinch_on_threshold", 0.02, 0.9),
        ("pinch_off_threshold", 0.04, 0.98),
        ("tilt_left_threshold", 0.05, 1.5),
        ("tilt_right_threshold", 0.05, 1.5),
        ("tilt_up_threshold", 0.05, 1.5),
        ("tilt_down_threshold", 0.05, 1.5),
        ("dead_zone", 0.0, 0.8),
        ("index_direction_threshold", 0.05, 1.5),
        ("palm_weight", 0.0, 1.0),
        ("index_weight", 0.0, 1.0),
        ("fist_on_threshold", 0.05, 0.9),
        ("fist_off_threshold", 0.1, 1.2),
        ("open_palm_on_threshold", 0.2, 1.5),
        ("open_palm_off_threshold", 0.05, 1.2),
    ):
        if key in raw:
            out[key] = _clamp_float(raw[key], low, high, defaults[key])
    if not out["pinch_on_threshold"] < out["pinch_off_threshold"]:
        out["pinch_on_threshold"] = defaults["pinch_on_threshold"]
        out["pinch_off_threshold"] = defaults["pinch_off_threshold"]
    if not out["fist_on_threshold"] < out["fist_off_threshold"]:
        out["fist_on_threshold"] = defaults["fist_on_threshold"]
        out["fist_off_threshold"] = defaults["fist_off_threshold"]
    if not out["open_palm_off_threshold"] < out["open_palm_on_threshold"]:
        out["open_palm_on_threshold"] = defaults["open_palm_on_threshold"]
        out["open_palm_off_threshold"] = defaults["open_palm_off_threshold"]
    if "geometry_stability_frames" in raw:
        out["geometry_stability_frames"] = _clamp_int(
            raw["geometry_stability_frames"], 1, 10, defaults["geometry_stability_frames"]
        )
    if "fist_enabled" in raw:
        out["fist_enabled"] = _as_bool(raw["fist_enabled"], defaults["fist_enabled"])
    if "open_palm_enabled" in raw:
        out["open_palm_enabled"] = _as_bool(raw["open_palm_enabled"], defaults["open_palm_enabled"])
    if "finger_up_on_threshold" in raw:
        out["finger_up_on_threshold"] = _clamp_float(
            raw["finger_up_on_threshold"], 0.05, 2.0, defaults["finger_up_on_threshold"]
        )
    if "finger_up_off_threshold" in raw:
        out["finger_up_off_threshold"] = _clamp_float(
            raw["finger_up_off_threshold"], 0.05, 2.0, defaults["finger_up_off_threshold"]
        )
    if "finger_up_mode" in raw:
        mode = str(raw["finger_up_mode"]).strip().lower()
        out["finger_up_mode"] = mode if mode in ("single", "multi") else defaults["finger_up_mode"]
    if "emergency_key" in raw:
        out["emergency_key"] = str(raw["emergency_key"]).strip().upper() or defaults.get("emergency_key", "F11")
    if "landmark_smoothing" in raw:
        out["landmark_smoothing"] = _as_bool(raw["landmark_smoothing"], defaults["landmark_smoothing"])
    if "smoothing_min_cutoff" in raw:
        out["smoothing_min_cutoff"] = _clamp_float(
            raw["smoothing_min_cutoff"], 0.1, 10.0, defaults["smoothing_min_cutoff"]
        )
    if "smoothing_beta" in raw:
        out["smoothing_beta"] = _clamp_float(raw["smoothing_beta"], 0.0, 10.0, defaults["smoothing_beta"])
    if "smoothing_d_cutoff" in raw:
        out["smoothing_d_cutoff"] = _clamp_float(
            raw["smoothing_d_cutoff"], 0.1, 10.0, defaults["smoothing_d_cutoff"]
        )
    fingers = {"THUMB", "INDEX", "MIDDLE", "RING", "PINKY"}
    for key in ("direction_finger_left", "direction_finger_right"):
        if key in raw:
            name = str(raw[key]).strip().upper()
            out[key] = name if name in fingers else defaults[key]
    if "gaming_mode" in raw:
        out["gaming_mode"] = _as_bool(raw["gaming_mode"], defaults["gaming_mode"])
    if "ml_backend" in raw:
        backend = str(raw["ml_backend"]).strip().lower()
        out["ml_backend"] = backend if backend in ("knn", "svm", "rf", "random_forest", "ensemble") else "knn"
    if "ensemble_mode" in raw:
        mode = str(raw["ensemble_mode"]).strip().lower().replace("_voting", "")
        out["ensemble_mode"] = mode if mode in ("hard", "soft", "weighted") else "weighted"
    for key in ("knn_weight", "svm_weight", "random_forest_weight"):
        if key in raw:
            out[key] = _clamp_float(raw[key], 0.0, 5.0, defaults[key])
    if "show_landmarks" in raw:
        out["show_landmarks"] = _as_bool(raw["show_landmarks"], defaults["show_landmarks"])
    if "hud_overlay" in raw:
        out["hud_overlay"] = _as_bool(raw["hud_overlay"], defaults["hud_overlay"])
    if "language" in raw:
        code = str(raw["language"]).strip().lower()
        out["language"] = code if code in ("fr", "en") else ""
    from system_gestures import normalize_system_gestures
    out["system_gestures"] = normalize_system_gestures(raw.get("system_gestures") if "system_gestures" in raw else out.get("system_gestures"))
    return out


def load_settings(path=None):
    """Load settings.json. Corrupt files are backed up; defaults are then used."""
    settings_path = path or user_data_path(SETTINGS_FILE)
    if not os.path.isfile(settings_path):
        data = default_settings()
        save_settings(data, settings_path)
        return data
    try:
        with open(settings_path, "r", encoding="utf-8-sig") as file:
            loaded = json.load(file)
    except json.JSONDecodeError as error:
        _log.error("settings.json is invalid: %s", error)
        backup_file(settings_path)
        data = default_settings()
        save_settings(data, settings_path)
        return data
    except OSError as error:
        _log.error("Could not read settings.json: %s", error)
        return default_settings()
    return validate_settings(loaded)


def save_settings(data, path=None):
    settings_path = path or user_data_path(SETTINGS_FILE)
    payload = validate_settings(data)
    try:
        atomic_json_write(settings_path, payload)
        return True
    except OSError as error:
        _log.error("Could not save settings.json: %s", error)
        return False


def setup_logging(debug=False):
    """One rotating log file. No webcam frames, no per-frame spam."""
    global _configured
    logger = logging.getLogger("handcontroller")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    if _configured:
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        return logger
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    try:
        os.makedirs(log_dir(), exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir(), LOG_FILE_NAME),
            maxBytes=512 * 1024,
            backupCount=1,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as error:
        logger.addHandler(logging.StreamHandler(sys.stderr))
        logger.error("Could not create log file: %s", error)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if debug else logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False
    _configured = True
    return logger


def get_logger():
    return logging.getLogger("handcontroller")
