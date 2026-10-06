"""Label helpers for tests and debug. Geometry names are support features, not the primary HUD gesture."""
from gesture_editor import format_gesture_label
from gesture_engine import UNKNOWN


def special_label(gesture):
    """Display name for one active detector. Neutrals and index-direction IDs are omitted."""
    raw = str(gesture or "").strip()
    if not raw or raw == UNKNOWN:
        return None
    name = raw.upper()
    if name.endswith("_NEUTRAL") or name.endswith(" NEUTRAL"):
        return None
    side = ""
    if name.startswith("LEFT_"):
        side = "LEFT"
        name = name[5:]
    elif name.startswith("RIGHT_"):
        side = "RIGHT"
        name = name[6:]
    if name in ("INDEX UP", "MIDDLE UP", "RING UP", "PINKY UP"):
        return name
    if name in ("PINCH LEFT", "PINCH RIGHT"):
        return name
    if name.startswith("TILT "):
        return name
    if name == "PINCH":
        if side == "LEFT":
            return "PINCH LEFT"
        if side == "RIGHT":
            return "PINCH RIGHT"
        return "PINCH"
    if name.startswith("TILT_"):
        return "TILT " + name[5:].replace("_", " ")
    if name.startswith("FINGER_"):
        return name[7:].replace("_", " ") + " UP"
    if name.startswith("INDEX_"):
        return None
    if name in ("FIST", "OPEN PALM", "OPEN_PALM"):
        return name.replace("_", " ")
    return name.replace("_", " ")


def active_gesture_labels(app, side, now=None):
    """Active mapped k-NN pose plus enabled geometry families."""
    labels = []
    seen = set()

    def add(text):
        name = str(text or "").strip()
        if not name or name == UNKNOWN:
            return
        if name not in seen:
            seen.add(name)
            labels.append(name)

    gesture = (getattr(app, "curr_gestures", None) or {}).get(side, UNKNOWN)
    if gesture and gesture != UNKNOWN:
        mapping = getattr(app, "mapping", None)
        mapped = True
        if mapping is not None:
            entry = mapping.get(side, gesture)
            if isinstance(entry, dict):
                kind = str(entry.get("type") or "").upper()
                if entry.get("enabled") is False or kind in ("", "NONE"):
                    mapped = False
        if mapped:
            shown = ""
            engine = getattr(app, "engine", None)
            if engine is not None and hasattr(engine, "display_name"):
                shown = engine.display_name(side, gesture)
            add(format_gesture_label(gesture, shown))
    specials = (getattr(app, "curr_specials", None) or {}).get(side) or {}
    allowed = getattr(app, "_system_allowed", None)
    for raw in specials.values():
        if allowed is not None and not allowed(side, raw):
            continue
        labeled = special_label(raw)
        if labeled:
            add(labeled)
    return labels


def _hand_present(app, side):
    track = (getattr(app, "track_state", None) or {}).get(side, "OK")
    if track == "GRACE":
        return True
    if track == "RELEASED":
        return False
    return bool((getattr(app, "detected", None) or {}).get(side))


def hand_overlay_state(app, side, now=None):
    """Camera card: (present, list of active labels)."""
    if not _hand_present(app, side):
        return False, []
    return True, active_gesture_labels(app, side, now)


_PRESET_I18N = {
    "INDEX UP": "overlay.index_up",
    "MIDDLE UP": "overlay.middle_up",
    "RING UP": "overlay.ring_up",
    "PINKY UP": "overlay.pinky_up",
    "THUMB UP": "overlay.thumb_up",
    "PINCH": "overlay.pinch",
    "PINCH LEFT": "overlay.pinch_left",
    "PINCH RIGHT": "overlay.pinch_right",
    "TILT UP": "overlay.tilt_up",
    "TILT DOWN": "overlay.tilt_down",
    "TILT LEFT": "overlay.tilt_left",
    "TILT RIGHT": "overlay.tilt_right",
    "FIST": "overlay.fist",
    "OPEN PALM": "overlay.open_palm",
}


# Predefined geometry gestures, in display order. Keys are engine IDs without
# the LEFT_/RIGHT_ prefix. Index-direction and NEUTRAL states are not gestures.
PRESET_LABELS = (
    ("PINCH", "PINCH"),
    ("TILT_UP", "TILT UP"),
    ("TILT_DOWN", "TILT DOWN"),
    ("TILT_LEFT", "TILT LEFT"),
    ("TILT_RIGHT", "TILT RIGHT"),
    ("FINGER_INDEX", "FINGER UP INDEX"),
    ("FINGER_MIDDLE", "FINGER UP MIDDLE"),
    ("FINGER_RING", "FINGER UP RING"),
    ("FINGER_PINKY", "FINGER UP PINKY"),
    ("FINGER_THUMB", "FINGER UP THUMB"),
    ("FIST", "FIST"),
    ("OPEN_PALM", "OPEN PALM"),
)
_PRESET_BY_CORE = dict(PRESET_LABELS)
_PRESET_RANK = {core: index for index, (core, _label) in enumerate(PRESET_LABELS)}
_LEGACY_LABELS = {
    "PINCH LEFT": "PINCH", "PINCH RIGHT": "PINCH",
    "INDEX UP": "FINGER_INDEX", "MIDDLE UP": "FINGER_MIDDLE", "RING UP": "FINGER_RING",
    "PINKY UP": "FINGER_PINKY", "THUMB UP": "FINGER_THUMB", "OPEN PALM": "OPEN_PALM",
}
_FAMILY_CORES = {
    "PINCH": ("PINCH",),
    "TILT": ("TILT_UP", "TILT_DOWN", "TILT_LEFT", "TILT_RIGHT"),
    "FINGER_INDEX": ("FINGER_INDEX",),
    "FINGER_MIDDLE": ("FINGER_MIDDLE",),
    "FINGER_RING": ("FINGER_RING",),
    "FINGER_PINKY": ("FINGER_PINKY",),
    "FINGER_THUMB": ("FINGER_THUMB",),
    "FIST": ("FIST",),
    "PALM": ("OPEN_PALM",),
}


def preset_core(gesture):
    """'LEFT_TILT_UP' -> 'TILT_UP'. None for neutrals, index directions, Gxx and unknown IDs."""
    name = str(gesture or "").strip().upper()
    if not name or name == UNKNOWN:
        return None
    if name.startswith("LEFT_"):
        name = name[5:]
    elif name.startswith("RIGHT_"):
        name = name[6:]
    name = _LEGACY_LABELS.get(name, name).replace(" ", "_")
    return name if name in _PRESET_BY_CORE else None


def preset_display_label(gesture):
    """Canonical on-screen name (PINCH, TILT LEFT, FINGER UP INDEX...). Never translated."""
    core = preset_core(gesture)
    return _PRESET_BY_CORE.get(core) if core else None


def active_preset_labels(app, side):
    """Predefined gestures the detectors report right now for this hand, in fixed order.

    Detector hysteresis and stability frames already debounce these states.
    Disabled gates are skipped, like the input path does.
    """
    if not _hand_present(app, side):
        return []
    specials = (getattr(app, "curr_specials", None) or {}).get(side) or {}
    allowed = getattr(app, "_system_allowed", None)
    cores = set()
    for raw in specials.values():
        core = preset_core(raw)
        if core is None:
            continue
        if allowed is not None and not allowed(side, raw):
            continue
        cores.add(core)
    return [_PRESET_BY_CORE[core] for core in sorted(cores, key=_PRESET_RANK.get)]


def preset_catalog(app, side):
    """Predefined gestures that exist in the running engine for this hand.

    Rows: {"id", "label", "enabled", "active"}. FIST / OPEN PALM only appear
    when their detectors are built (settings), so nothing is invented.
    """
    side = str(side or "").upper()
    geometry = getattr(app, "geometry", None)
    families = ()
    if geometry is not None and hasattr(geometry, "families"):
        try:
            families = tuple(geometry.families(side))
        except Exception:
            families = ()
    if families:
        cores = [core for family in families for core in _FAMILY_CORES.get(str(family), ())]
    else:
        import geometry_engine as geo
        cores = [preset_core(gid) for gid in geo.SPECIAL_BY_SIDE.get(side, ())]
        cores = [core for core in cores if core and core not in ("FIST", "OPEN_PALM")]
    allowed = getattr(app, "_system_allowed", None)
    active = set(active_preset_labels(app, side))
    rows = []
    for core in sorted(set(cores), key=_PRESET_RANK.get):
        gid = f"{side}_{core}"
        enabled = True
        if allowed is not None:
            try:
                enabled = bool(allowed(side, gid))
            except Exception:
                enabled = True
        label = _PRESET_BY_CORE[core]
        rows.append({"id": gid, "label": label, "enabled": enabled, "active": label in active})
    return rows


def recorded_gesture_label(app, side):
    """Stable k-NN pose for this hand ('G01' or 'G01   custom name'), or ''."""
    if not _hand_present(app, side):
        return ""
    gesture = str((getattr(app, "curr_gestures", None) or {}).get(side, UNKNOWN) or UNKNOWN)
    if gesture == UNKNOWN:
        return ""
    shown = ""
    engine = getattr(app, "engine", None)
    if engine is not None and hasattr(engine, "display_name"):
        try:
            shown = engine.display_name(side, gesture)
        except Exception:
            shown = ""
    return localize_display_label(format_gesture_label(gesture, shown))


def localize_display_label(text):
    """'G01 — Index gauche' -> 'G01   Index gauche'. Preset names follow the language.
    User gesture names are never translated."""
    from localization import t

    raw = str(text or "").strip()
    if not raw:
        return ""
    key = _PRESET_I18N.get(raw.upper().replace("_", " "))
    if key:
        return t(key)
    return raw.replace(" — ", "   ")


def osd_side_view(app, side):
    """External OSD block for one hand: title, present/absent, detected gesture(s)."""
    from localization import t

    present = _hand_present(app, side)
    parts = []
    if present:
        recorded = recorded_gesture_label(app, side)
        if recorded:
            parts.append(recorded)
        parts.extend(active_preset_labels(app, side))
    return {
        "side": side,
        "title": t("overlay.left") if side == "LEFT" else t("overlay.right"),
        "gesture": " + ".join(parts) if parts else (t("overlay.none") if present else ""),
        "state": t("overlay.present") if present else t("overlay.absent"),
        "present": present,
    }


def osd_view(app):
    return [osd_side_view(app, side) for side in ("LEFT", "RIGHT")]


def osd_paint_lines(app):
    """Rows for game_osd.GameOSD.update: (text, tone) with tone white|cyan|dim."""
    rows = []
    for index, view in enumerate(osd_view(app)):
        if index:
            rows.append(("", "dim"))
        rows.append((view["title"], "cyan"))
        rows.append((view["state"], "dim"))
        if view["gesture"]:
            rows.append((view["gesture"], "white"))
    return rows


def status_side_view(app, side):
    """Compact local status (OpenCV window): title, present/absent, gesture line.

    gesture line: 'G01   Conf: 0.94' (k-NN), 'PINCH' (predefined only),
    'G01 + PINCH   Conf: 0.94' (both) or 'Aucun   Conf: 0.00'.
    """
    from localization import t

    present = _hand_present(app, side)
    gesture = (getattr(app, "curr_gestures", None) or {}).get(side, UNKNOWN) if present else UNKNOWN
    gesture = str(gesture or UNKNOWN)
    try:
        conf = float((getattr(app, "confidences", None) or {}).get(side, 0.0) or 0.0) if present else 0.0
    except (TypeError, ValueError):
        conf = 0.0
    presets = active_preset_labels(app, side) if present else []
    conf_text = t("state.conf", value=f"{max(0.0, min(1.0, conf)):.2f}")
    names = ([gesture] if gesture != UNKNOWN else []) + presets
    if gesture != UNKNOWN:
        line = f"{' + '.join(names)}   {conf_text}"
    elif presets:
        line = " + ".join(presets)
    else:
        line = f"{t('state.none')}   {conf_text}"
    return {
        "side": side,
        "title": t("state.left") if side == "LEFT" else t("state.right"),
        "state": t("state.present") if present else t("state.absent"),
        "present": present,
        "gesture": " + ".join(names) if names else t("state.none"),
        "presets": presets,
        "conf": conf_text,
        "line": line,
    }


def osd_state_from_app(app, now=None):
    """Label snapshot for tests and debug. Does not create a window."""
    def status(side):
        present, labels = hand_overlay_state(app, side, now)
        if not present:
            return "none"
        if not labels:
            return "none"
        return " + ".join(labels)

    return {
        "enabled": False,
        "left": status("LEFT"),
        "right": status("RIGHT"),
        "left_labels": hand_overlay_state(app, "LEFT", now)[1] if _hand_present(app, "LEFT") else [],
        "right_labels": hand_overlay_state(app, "RIGHT", now)[1] if _hand_present(app, "RIGHT") else [],
        "show_left_hand": True,
        "show_right_hand": True,
    }
