"""Enable/disable gates for PINCH, TILT and FINGER UP.

Stored in settings.json. Not a game-profile manager.
"""
import json
import os
import shutil

from config import GESTURES_FILE, MAPPING_FILE, atomic_json_write, backup_file, get_logger

DIRECTIONS = ("left", "right", "up", "down")
FINGER_UP_FINGERS = ("index", "middle", "ring", "pinky", "thumb")
_INDEX_UP_IDS = {"LEFT": "LEFT_INDEX_UP", "RIGHT": "RIGHT_INDEX_UP"}
_FINGER_INDEX_IDS = {"LEFT": "LEFT_FINGER_INDEX", "RIGHT": "RIGHT_FINGER_INDEX"}

# Former per-profile knobs. They now live in the unique settings.json.
LEGACY_PROFILE_SETTING_KEYS = (
    "direction_finger_left",
    "direction_finger_right",
    "confidence_threshold",
    "stable_frames",
    "knn_k",
    "knn_distance_margin",
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
)


def default_system_gestures():
    hand = {direction: {"enabled": False} for direction in DIRECTIONS}
    return {
        "pinch": {"left": {"enabled": False}, "right": {"enabled": False}},
        "tilt": {"left": dict(hand), "right": {direction: {"enabled": False} for direction in DIRECTIONS}},
        "finger_up": {"mode": "single", "bindings": []},
    }


def _bool(value, fallback=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(value, (int, float)):
        return value != 0
    return fallback


def _copy_dirs(raw, fallback=False):
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    for direction in DIRECTIONS:
        block = raw.get(direction)
        if not isinstance(block, dict):
            block = {"enabled": _bool(block, fallback)} if block is not None else {"enabled": fallback}
        else:
            block = {"enabled": _bool(block.get("enabled"), fallback)}
        out[direction] = block
    return out


def normalize_system_gestures(raw):
    base = default_system_gestures()
    raw = raw if isinstance(raw, dict) else {}
    pinch = raw.get("pinch") if isinstance(raw.get("pinch"), dict) else {}
    for side in ("left", "right"):
        block = pinch.get(side)
        if isinstance(block, dict):
            base["pinch"][side]["enabled"] = _bool(block.get("enabled"))
        elif block is not None:
            base["pinch"][side]["enabled"] = _bool(block)
    tilt = raw.get("tilt") if isinstance(raw.get("tilt"), dict) else {}
    for side in ("left", "right"):
        base["tilt"][side] = _copy_dirs(tilt.get(side) if isinstance(tilt.get(side), dict) else tilt)
    finger = raw.get("finger_up") if isinstance(raw.get("finger_up"), dict) else {}
    mode = str(finger.get("mode") or "single").strip().lower()
    base["finger_up"]["mode"] = mode if mode in ("single", "multi") else "single"
    bindings = []
    seen = set()
    for item in finger.get("bindings") or []:
        if not isinstance(item, dict):
            continue
        hand = str(item.get("hand") or "").strip().lower()
        finger_name = str(item.get("finger") or "").strip().lower()
        if hand not in ("left", "right") or finger_name not in FINGER_UP_FINGERS:
            continue
        key = (hand, finger_name)
        if key in seen:
            continue
        seen.add(key)
        bindings.append({
            "hand": hand,
            "finger": finger_name,
            "enabled": _bool(item.get("enabled"), True),
        })
    base["finger_up"]["bindings"] = bindings
    return base


def _mapping_active(entry):
    if not isinstance(entry, dict):
        return False
    if entry.get("enabled") is False:
        return False
    entry_type = str(entry.get("type") or "").upper()
    return entry_type not in ("", "NONE")


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("unreadable json")
    return data


def migrate_system_gestures(config, mapping_path):
    """Fill missing gates from mapping.json. Never wipes old mappings."""
    config = normalize_system_gestures(config)
    mapping_data = {}
    if mapping_path and os.path.isfile(mapping_path):
        try:
            mapping_data = _read_json(mapping_path)
        except (OSError, ValueError, json.JSONDecodeError):
            mapping_data = {}
    if not isinstance(mapping_data, dict):
        mapping_data = {}

    def side_table(side):
        table = mapping_data.get(side)
        return table if isinstance(table, dict) else {}

    for side in ("LEFT", "RIGHT"):
        table = side_table(side)
        key = side.lower()
        pinch_id = f"{side}_PINCH"
        if _mapping_active(table.get(pinch_id)):
            config["pinch"][key]["enabled"] = True
        for direction in DIRECTIONS:
            tilt_id = f"{side}_TILT_{direction.upper()}"
            if _mapping_active(table.get(tilt_id)):
                config["tilt"][key][direction]["enabled"] = True
        old_id = _INDEX_UP_IDS[side]
        new_id = _FINGER_INDEX_IDS[side]
        if _mapping_active(table.get(old_id)):
            already = any(
                item["hand"] == key and item["finger"] == "index"
                for item in config["finger_up"]["bindings"]
            )
            if not already:
                config["finger_up"]["bindings"].append({
                    "hand": key,
                    "finger": "index",
                    "enabled": True,
                })
            if not _mapping_active(table.get(new_id)):
                table[new_id] = dict(table[old_id])
                try:
                    payload = dict(mapping_data)
                    payload[side] = table
                    atomic_json_write(mapping_path, payload)
                except OSError:
                    pass
    return config


def enable_mapped_preset(config, side, gesture):
    """Turn on the geometry gate that matches a mapped special ID."""
    config = normalize_system_gestures(config)
    side = str(side or "").strip().upper()
    gesture = str(gesture or "").strip().upper()
    if gesture == f"{side}_PINCH":
        return set_system_enabled(config, "pinch", side, enabled=True)
    for direction in DIRECTIONS:
        if gesture == f"{side}_TILT_{direction.upper()}":
            return set_system_enabled(config, "tilt", side, direction, True)
    prefix = f"{side}_FINGER_"
    if gesture.startswith(prefix):
        finger = gesture[len(prefix):].lower()
        return set_system_enabled(config, "finger_up", side, enabled=True, finger=finger)
    return config


def is_system_enabled(config, side, gesture):
    """Gate for PINCH / TILT / FINGER UP. Other specials stay mapping-only."""
    if not config:
        return True
    config = normalize_system_gestures(config)
    side = str(side or "").strip().upper()
    gesture = str(gesture or "").strip().upper()
    key = side.lower()
    if key not in ("left", "right"):
        return True
    if gesture == f"{side}_PINCH":
        return bool(config["pinch"][key]["enabled"])
    for direction in DIRECTIONS:
        if gesture == f"{side}_TILT_{direction.upper()}":
            return bool(config["tilt"][key][direction]["enabled"])
    prefix = f"{side}_FINGER_"
    if gesture.startswith(prefix):
        finger = gesture[len(prefix):].lower()
        for item in config["finger_up"]["bindings"]:
            if item["hand"] == key and item["finger"] == finger:
                return bool(item["enabled"])
        return False
    return True


def set_system_enabled(config, kind, side, direction=None, enabled=True, finger=None):
    config = normalize_system_gestures(config)
    side = str(side or "").strip().lower()
    if kind == "pinch" and side in config["pinch"]:
        config["pinch"][side]["enabled"] = bool(enabled)
    elif kind == "tilt" and side in config["tilt"] and direction in DIRECTIONS:
        config["tilt"][side][direction]["enabled"] = bool(enabled)
    elif kind == "finger_up" and side in ("left", "right"):
        finger = str(finger or "").strip().lower()
        if finger in FINGER_UP_FINGERS:
            found = False
            for item in config["finger_up"]["bindings"]:
                if item["hand"] == side and item["finger"] == finger:
                    item["enabled"] = bool(enabled)
                    found = True
                    break
            if not found and enabled:
                config["finger_up"]["bindings"].append({
                    "hand": side,
                    "finger": finger,
                    "enabled": True,
                })
    return config


def add_finger_binding(config, side, finger, enabled=True):
    config = normalize_system_gestures(config)
    side = str(side or "").strip().lower()
    finger = str(finger or "").strip().lower()
    if side not in ("left", "right") or finger not in FINGER_UP_FINGERS:
        return config
    for item in config["finger_up"]["bindings"]:
        if item["hand"] == side and item["finger"] == finger:
            item["enabled"] = bool(enabled)
            return config
    config["finger_up"]["bindings"].append({
        "hand": side,
        "finger": finger,
        "enabled": bool(enabled),
    })
    return config


def remove_finger_binding(config, side, finger):
    config = normalize_system_gestures(config)
    side = str(side or "").strip().lower()
    finger = str(finger or "").strip().lower()
    config["finger_up"]["bindings"] = [
        item for item in config["finger_up"]["bindings"]
        if not (item["hand"] == side and item["finger"] == finger)
    ]
    return config


def system_gesture_id(kind, side, direction=None, finger=None):
    side = str(side or "").strip().upper()
    if kind == "pinch":
        return f"{side}_PINCH"
    if kind == "tilt":
        return f"{side}_TILT_{str(direction or '').strip().upper()}"
    if kind == "finger_up":
        return f"{side}_FINGER_{str(finger or '').strip().upper()}"
    return None


def _list_legacy_profiles(profiles_dir):
    if not os.path.isdir(profiles_dir):
        return []
    found = []
    for name in sorted(os.listdir(profiles_dir)):
        folder = os.path.join(profiles_dir, name)
        if os.path.isfile(os.path.join(folder, "profile.json")):
            found.append(name)
    return found


def _copy_if_file(source, dest):
    if not source or not dest or not os.path.isfile(source):
        return False
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    if os.path.isfile(dest):
        backup_file(dest)
    shutil.copy2(source, dest)
    return True


def migrate_legacy_profiles(app_directory, settings):
    """Move the active profile's files into the unique config.

    Never deletes profiles/. Never opens gestures_backup.json.
    Extra game folders are copied to profiles_archive/ once.
    """
    settings = dict(settings or {})
    log = get_logger()
    profiles_dir = os.path.join(app_directory, "profiles")
    ids = _list_legacy_profiles(profiles_dir)
    if not ids:
        settings["system_gestures"] = normalize_system_gestures(settings.get("system_gestures"))
        settings.pop("active_profile", None)
        return settings

    archive = os.path.join(app_directory, "profiles_archive")
    if not os.path.isdir(archive):
        try:
            shutil.copytree(profiles_dir, archive, dirs_exist_ok=True)
            log.info("Archived legacy profiles to %s", archive)
        except OSError as error:
            log.warning("Could not archive profiles/: %s", error)

    requested = str(settings.get("active_profile") or "default").strip() or "default"
    chosen = requested if requested in ids else ("default" if "default" in ids else ids[0])
    folder = os.path.join(profiles_dir, chosen)
    gestures_src = os.path.join(folder, "gestures.json")
    mapping_src = os.path.join(folder, "mapping.json")
    gestures_dst = os.path.join(app_directory, GESTURES_FILE)
    mapping_dst = os.path.join(app_directory, MAPPING_FILE)
    copied_g = _copy_if_file(gestures_src, gestures_dst)
    copied_m = _copy_if_file(mapping_src, mapping_dst)
    if copied_g or copied_m:
        log.info("Migrated unique config from profile %s (gestures=%s mapping=%s)", chosen, copied_g, copied_m)
    if len(ids) > 1:
        log.warning(
            "Several legacy profiles were found (%s). Runtime uses %s; others stay in profiles_archive/",
            ", ".join(ids), chosen,
        )

    info = {}
    profile_json = os.path.join(folder, "profile.json")
    if os.path.isfile(profile_json):
        try:
            info = _read_json(profile_json)
        except (OSError, ValueError, json.JSONDecodeError):
            info = {}
    overlay = info.get("settings") if isinstance(info.get("settings"), dict) else {}
    for key in LEGACY_PROFILE_SETTING_KEYS:
        if key in overlay:
            settings[key] = overlay[key]
    mapping_path = mapping_dst if os.path.isfile(mapping_dst) else mapping_src
    if settings.get("system_gestures"):
        settings["system_gestures"] = normalize_system_gestures(settings.get("system_gestures"))
    else:
        settings["system_gestures"] = migrate_system_gestures(info.get("system_gestures"), mapping_path)
    settings.pop("active_profile", None)
    return settings
