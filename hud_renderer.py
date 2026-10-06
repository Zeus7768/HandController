"""Classic OpenCV HUD. Display only — never runs MediaPipe or sends input."""
import cv2
import numpy as np

from gesture_engine import SIDES


SIDE_TEXT_COLORS = {"LEFT": (245, 245, 245), "RIGHT": (120, 180, 255)}
HUD_WHITE = (255, 255, 255)
HUD_DIM = (180, 180, 180)
HUD_DEBUG = (150, 210, 255)
HUD_OK = (0, 255, 0)


def draw_hand_landmarks(frame, hand, color=(150, 150, 150)):
    """MediaPipe points: left white, right black. Circles only."""
    if frame is None or hand is None:
        return
    height, width = frame.shape[:2]
    outline = (255, 255, 255) if color == (0, 0, 0) else (0, 0, 0)
    rows = hand
    if hasattr(hand, "shape"):
        rows = np.asarray(hand)
    for lm in rows:
        if hasattr(lm, "x"):
            center = (int(lm.x * width), int(lm.y * height))
        else:
            center = (int(float(lm[0]) * width), int(float(lm[1]) * height))
        cv2.circle(frame, center, 4, outline, 1)
        cv2.circle(frame, center, 3, color, -1)


def preset_display_name(gesture):
    """LEFT_TILT_UP -> TILT_UP, LEFT_FINGER_INDEX -> FINGER_UP_INDEX."""
    name = str(gesture or "").strip().upper()
    if name.startswith("LEFT_"):
        name = name[5:]
    elif name.startswith("RIGHT_"):
        name = name[6:]
    if name.startswith("FINGER_"):
        return "FINGER_UP_" + name[7:]
    return name


def mapping_row_parts(name, entry):
    """Aligned OpenCV columns: ID, key/action, type, enabled state."""
    from localization import t

    entry = entry or {}
    kind = str(entry.get("type") or "NONE").upper()
    enabled = entry.get("enabled", True) is not False
    raw_name = str(name or "")
    if len(raw_name) >= 3 and raw_name[0].upper() == "G" and raw_name[1:].isdigit():
        gid = "G" + raw_name[1:]
    else:
        gid = preset_display_name(raw_name)
    if kind == "MACRO":
        action = t("mapping.macro_cell")
        type_s = t("mapping.kind_macro")
        state = t("mapping.state_on") if enabled else t("mapping.state_off")
    elif kind in ("", "NONE") or not list(entry.get("inputs") or []):
        action = t("mapping.empty_cell")
        type_s = t("mapping.kind_none")
        state = t("mapping.state_off")
    else:
        action = "+".join(entry.get("inputs") or [])
        type_map = {
            "HOLD": t("mapping.kind_hold"),
            "PRESS": t("mapping.kind_press"),
            "COMBINATION": t("mapping.kind_combo"),
        }
        type_s = type_map.get(kind, f"[{kind}]")
        state = t("mapping.state_on") if enabled else t("mapping.state_off")
    return gid, action, type_s, state


def status_scale(width, height, compact=False):
    """Small Hershey scale that follows the preview size (0.32 .. 0.50)."""
    base = min(float(width), float(height) * 1.5) / 1700.0
    if compact:
        base *= 0.9
    return max(0.32, min(0.50, base))


def status_rows(app):
    """Compact local status rows: (text, color). Never mentions version or camera."""
    from localization import t
    from overlay_labels import status_side_view

    rows = []
    for side in SIDES:
        view = status_side_view(app, side)
        rows.append((view["title"], SIDE_TEXT_COLORS.get(side, HUD_WHITE)))
        rows.append((view["state"], HUD_OK if view["present"] else HUD_DIM))
        rows.append((view["line"], HUD_WHITE if view["present"] else HUD_DIM))
    inputs = getattr(app, "inputs", None)
    enabled = bool(getattr(inputs, "enabled", False))
    rows.append((t("state.input", state="ON" if enabled else "OFF"), HUD_OK if enabled else HUD_DIM))
    active = False
    probe = getattr(app, "game_osd_active", None)
    if callable(probe):
        try:
            active = bool(probe())
        except Exception:
            active = False
    state = t("state.overlay_ok") if active else t("state.overlay_off")
    rows.append((t("state.screen_overlay", state=state), HUD_OK if active else HUD_DIM))
    return rows


def draw_preview(app, frame, put, compact=False):
    """Compact status block, top-left, semi-transparent. Always drawn on the main view."""
    if frame is None or frame.size == 0:
        return 0
    rows = status_rows(app)
    height, width = frame.shape[:2]
    scale = status_scale(width, height, compact)
    line_h = max(11, int(round(38 * scale)))
    gap = max(3, line_h // 3)
    pad = max(3, line_h // 4)
    x0, y0 = 6, 6
    widest = 0
    for text, _color in rows:
        tw = cv2.getTextSize(_ascii(text), cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        widest = max(widest, tw)
    box_w = min(width - 2 * x0, widest + 2 * pad + 4)
    # Blank gap after each hand block (3 rows) and before the footer rows.
    box_h = pad * 2 + line_h * len(rows) + gap * 2
    box_h = min(box_h, height - 2 * y0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)
    y = y0 + pad + int(line_h * 0.8)
    for index, (text, color) in enumerate(rows):
        if index in (3, 6):
            y += gap
        put(frame, text, (x0 + pad + 2, y), scale, color, 1, box_w - pad)
        y += line_h
    bottom = y0 + box_h
    app._status_box_bottom = bottom
    return bottom


def _ascii(text):
    import unicodedata

    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def draw_debug_info(app, frame, put, compact=False):
    """Debug-only dump. Independent of the Afterburner OSD."""
    if not getattr(app, "debug", False):
        return
    if frame is None or frame.size == 0:
        return
    lines = []
    if hasattr(app, "debug_hud_lines"):
        lines = list(app.debug_hud_lines() or [])
    if not lines:
        return
    height, width = frame.shape[:2]
    y = max(38, int(getattr(app, "_status_box_bottom", 0) or 0) + 18)
    scale = 0.36 if compact else 0.40
    for text in lines[:28]:
        put(frame, str(text), (10, y), scale, HUD_DEBUG, 1, width - 20)
        y += 14 if compact else 16
        if y > height - 24:
            break
