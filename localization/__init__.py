"""French / English UI strings. Technical IDs and mapping values stay unchanged.

Catalogs live in this package (fr.json, en.json) so a frozen EXE can ship
them next to the model via resource_path().
"""
import json
import os
import sys
import unicodedata

from config import APP_NAME, resource_path, save_settings

SUPPORTED = ("fr", "en")
DEFAULT_LANGUAGE = "fr"
CATALOG_DIR = "localization"

_catalogs = {}
_language = DEFAULT_LANGUAGE
_loaded = False


def catalog_path(lang):
    name = f"{lang}.json"
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if os.path.isfile(here):
        return here
    path = resource_path(os.path.join(CATALOG_DIR, name))
    if os.path.isfile(path):
        return path
    alt = resource_path(f"{CATALOG_DIR}/{name}")
    if os.path.isfile(alt):
        return alt
    return here


def _flatten(node, prefix=""):
    flat = {}
    if not isinstance(node, dict):
        return flat
    for key, value in node.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, full))
        else:
            flat[full] = "" if value is None else str(value)
    return flat


def _read_catalog(lang):
    path = catalog_path(lang)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return _flatten(raw)


def available_languages():
    """Language codes that have a real catalog file in localization/."""
    folder = os.path.dirname(os.path.abspath(__file__))
    found = []
    try:
        names = os.listdir(folder)
    except OSError:
        names = []
    for name in sorted(names):
        if not name.lower().endswith(".json"):
            continue
        code = name[:-5].lower()
        path = os.path.join(folder, name)
        if code and os.path.isfile(path):
            found.append(code)
    return found or list(SUPPORTED)


def language_display_name(code):
    """Native name from that catalog (Francais, English). Falls back to the code."""
    table = _read_catalog(code) if code else {}
    return table.get(f"lang.{code}") or str(code or "").upper()


def language_options():
    return [(code, language_display_name(code)) for code in available_languages()]


def language_prompt_state(selected=None, open_list=False):
    langs = available_languages()
    if selected not in langs:
        selected = DEFAULT_LANGUAGE if DEFAULT_LANGUAGE in langs else (langs[0] if langs else "fr")
    return {"selected": selected, "open": bool(open_list), "langs": list(langs)}


def language_prompt_click(state, action, payload=None):
    """Drive the installer-style language prompt without starting the camera."""
    state = dict(state or language_prompt_state())
    langs = list(state.get("langs") or available_languages())
    state["langs"] = langs
    if action == "toggle":
        state["open"] = not bool(state.get("open"))
        return state
    if action == "pick" and payload in langs:
        state["selected"] = payload
        state["open"] = False
        return state
    if action == "ok":
        chosen = state.get("selected") if state.get("selected") in langs else DEFAULT_LANGUAGE
        return chosen
    if action == "cancel":
        return None
    return state


INPUT_OPTIONS = ("on", "off")


def input_option_label(value):
    return t("ui.enable") if value == "on" else t("ui.disable")


def input_prompt_state(selected=None, open_list=False):
    if selected not in INPUT_OPTIONS:
        selected = "off"
    return {"selected": selected, "open": bool(open_list), "options": list(INPUT_OPTIONS)}


def input_prompt_click(state, action, payload=None):
    """Drive the Input enable/disable dialog without starting the camera."""
    state = dict(state or input_prompt_state())
    if action == "toggle":
        state["open"] = not bool(state.get("open"))
        return state
    if action == "pick" and payload in INPUT_OPTIONS:
        state["selected"] = payload
        state["open"] = False
        return state
    if action == "ok":
        return "on" if state.get("selected") == "on" else "off"
    if action == "cancel":
        return None
    return state


def load_catalogs(force=False):
    global _catalogs, _loaded
    if _loaded and not force:
        return _catalogs
    _catalogs = {lang: _read_catalog(lang) for lang in available_languages()}
    _loaded = True
    return _catalogs


def normalize_language(value):
    if not isinstance(value, str):
        return None
    code = value.strip().lower()
    return code if code in available_languages() else None


def language_configured(settings):
    return normalize_language((settings or {}).get("language")) is not None


def current_language():
    return _language


def set_language(lang):
    """Switch the active catalog. Unknown codes fall back to French."""
    global _language
    load_catalogs()
    _language = normalize_language(lang) or DEFAULT_LANGUAGE
    return _language


def catalog_keys(lang=None):
    load_catalogs()
    return set(_catalogs.get(lang or _language, {}))


def all_catalog_keys():
    load_catalogs()
    keys = set()
    for table in _catalogs.values():
        keys.update(table)
    return keys


def t(msgid, **kwargs):
    """Look up a UI string. Missing keys fall back to French, then to the key."""
    load_catalogs()
    text = _catalogs.get(_language, {}).get(msgid)
    if text is None:
        text = _catalogs.get(DEFAULT_LANGUAGE, {}).get(msgid, msgid)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            pass
    return text


def fold_ascii(text):
    """Hershey fonts cannot draw accents. Used only for OpenCV overlays."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def action_label(name):
    """Display name for PRESS / HOLD / … Internal values stay unchanged."""
    if not name:
        return ""
    key = f"action.{str(name).strip().lower()}"
    load_catalogs()
    if key in _catalogs.get(_language, {}) or key in _catalogs.get(DEFAULT_LANGUAGE, {}):
        return t(key)
    return str(name)


def dynamic_label(gesture_id, custom=None):
    """Kept for older tests; FLICK/SWIPE names are no longer catalogued."""
    if custom:
        return custom
    return str(gesture_id or "")


def apply_language(settings, interactive=False):
    """Activate the saved language, or ask once if none is stored.

    interactive Cancel returns None and does not start the camera.
    """
    settings = settings if isinstance(settings, dict) else {}
    code = normalize_language(settings.get("language"))
    if code is None:
        if interactive:
            code = prompt_language()
            if code is None:
                return None
            settings["language"] = code
            save_settings(settings)
        else:
            code = DEFAULT_LANGUAGE
    else:
        settings["language"] = code
    set_language(code)
    return code


def persist_language(settings, code, path=None):
    """Save a valid catalog code to settings.json and activate it."""
    settings = settings if isinstance(settings, dict) else {}
    chosen = normalize_language(code) or DEFAULT_LANGUAGE
    settings["language"] = chosen
    if path is None:
        save_settings(settings)
    else:
        save_settings(settings, path)
    set_language(chosen)
    return chosen


def prompt_language():
    """First-launch installer-style picker. Camera is not opened."""
    chosen = _prompt_language_opencv()
    if chosen is False:
        return _prompt_language_stdin()
    return chosen


def _prompt_language_opencv():
    try:
        import cv2
        import numpy as np
    except Exception:
        return False
    title = APP_NAME
    width, height = 460, 280
    state = language_prompt_state()
    hits = []

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONUP:
            return
        for x0, y0, x1, y1, action, payload in list(hits):
            if x0 <= x < x1 and y0 <= y < y1:
                result = language_prompt_click(state, action, payload)
                if result is None or isinstance(result, str):
                    state["_done"] = result
                else:
                    state.update(result)
                return

    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, width, height)
        _center_small_window(title, width, height)
        cv2.setMouseCallback(title, on_mouse)
        while True:
            try:
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    return None
            except cv2.error:
                return None
            if "_done" in state:
                return state["_done"]
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (36, 34, 32)
            hits[:] = _draw_language_prompt(frame, state)
            cv2.imshow(title, frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                return None
            if key in (13, 10):
                return language_prompt_click(state, "ok")
    except Exception:
        return False
    finally:
        try:
            cv2.destroyWindow(title)
        except Exception:
            pass


def _center_small_window(title, width, height):
    """Center a dialog. Does not steal focus or pin the window."""
    try:
        import ctypes
        sw = int(ctypes.windll.user32.GetSystemMetrics(0))
        sh = int(ctypes.windll.user32.GetSystemMetrics(1))
        import cv2
        cv2.moveWindow(title, max(0, (sw - width) // 2), max(0, (sh - height) // 2))
    except Exception:
        pass


def _draw_language_prompt(frame, state):
    import cv2
    height, width = frame.shape[:2]
    hits = []
    set_language(state.get("selected") or DEFAULT_LANGUAGE)
    cv2.putText(frame, APP_NAME, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.putText(frame, fold_ascii(t("lang.choose")), (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1)
    box = (20, 104, width - 20, 140)
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (52, 50, 48), -1)
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (120, 118, 110), 1)
    label = language_display_name(state.get("selected"))
    cv2.putText(frame, fold_ascii(label), (28, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, "v", (width - 42, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    hits.append((box[0], box[1], box[2], box[3], "toggle", None))
    y = 148
    if state.get("open"):
        for code, name in language_options():
            selected = code == state.get("selected")
            cv2.rectangle(frame, (20, y), (width - 20, y + 28), (40, 90, 40) if selected else (48, 46, 44), -1)
            cv2.putText(frame, fold_ascii(name), (28, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            hits.append((20, y, width - 20, y + 28, "pick", code))
            y += 30
    cv2.rectangle(frame, (20, height - 48), (130, height - 16), (50, 50, 50), -1)
    cv2.putText(frame, fold_ascii(t("lang.cancel") if t("lang.cancel") != "lang.cancel" else t("ui.cancel")), (28, height - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    hits.append((20, height - 48, 130, height - 16, "cancel", None))
    cv2.rectangle(frame, (width - 110, height - 48), (width - 20, height - 16), (30, 110, 50), -1)
    cv2.putText(frame, "OK", (width - 88, height - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    hits.append((width - 110, height - 48, width - 20, height - 16, "ok", None))
    return hits


def _prompt_language_stdin():
    if not sys.stdin or not hasattr(sys.stdin, "isatty") or not sys.stdin.isatty():
        return DEFAULT_LANGUAGE
    options = language_options()
    try:
        print(t("lang.choose"))
        for index, (code, name) in enumerate(options, start=1):
            print(f"[{index}] {name}")
        choice = input("[OK/cancel] ").strip().lower()
        if choice in ("cancel", "annuler", "n", "q"):
            return None
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(options):
                return options[index][0]
        return DEFAULT_LANGUAGE
    except EOFError:
        return DEFAULT_LANGUAGE


set_language(DEFAULT_LANGUAGE)
