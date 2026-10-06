"""Per-screen UI shortcut table. A key only fires if this context allows it."""

MAIN = "MAIN"
CALIBRATION = "CALIBRATION"
GESTURE_LIST = "GESTURE_LIST"
COMMAND_SETTINGS = "COMMAND_SETTINGS"
COMBINATION_SETTINGS = "COMBINATION_SETTINGS"
MACRO = "MACRO"
RENAME = "RENAME"
LANGUAGE = "LANGUAGE"
HELP = "HELP"
EDITOR = "EDITOR"
SYSTEM = "SYSTEM"
CAPTURE = "CAPTURE"

TEXT_CONTEXTS = (RENAME, CAPTURE)

# Keys that this screen actually handles. A key not listed here must not fire
# a leftover global action from another menu.
SHORTCUTS = {
    MAIN: {
        "q": "quit", "c": "calibration", "s": "overlay", "p": "mapping",
        "e": "editor", "l": "language", "h": "help", "g": "input",
        "i": "input_panel", "esc": "close_sub",
    },
    CALIBRATION: {"esc": "back", "m": "hand", "enter": "calibrate", "left": "prev", "right": "next"},
    GESTURE_LIST: {
        "esc": "back", "m": "hand", "a": "enable", "d": "disable", "enter": "edit",
        "r": "rename", "s": "delete", "space": "save", "left": "prev", "right": "next",
    },
    COMMAND_SETTINGS: {
        "esc": "back", "1": "hold", "2": "press", "enter": "edit",
        "space": "save", "left": "prev", "right": "next",
    },
    COMBINATION_SETTINGS: {
        "esc": "back", "a": "add", "d": "remove", "t": "slot",
        "1": "press", "2": "hold", "space": "save", "left": "prev", "right": "next",
    },
    MACRO: {
        "esc": "back", "a": "press", "w": "wait", "m": "hold", "r": "release",
        "c": "combination", "b": "repeat", "enter": "edit", "s": "delete",
        "t": "test", "space": "save", "left": "prev", "right": "next",
    },
    RENAME: {"enter": "confirm", "esc": "cancel", "backspace": "delete"},
    LANGUAGE: {"1": "fr", "2": "en", "esc": "back"},
    HELP: {"esc": "back"},
    EDITOR: {"esc": "back"},
    SYSTEM: {"esc": "back"},
    CAPTURE: {"esc": "cancel"},
}


def of(app):
    """Current UI screen. Mapping/calibration take priority over MAIN."""
    if getattr(app, "map_mode", False):
        phase = getattr(app, "map_phase", "")
        if phase == "rename":
            return RENAME
        if phase == "capture":
            return CAPTURE
        if phase == "combo":
            return COMBINATION_SETTINGS
        if phase == "simple":
            return COMMAND_SETTINGS
        if phase in ("macro", "step"):
            return MACRO
        return GESTURE_LIST
    if getattr(app, "sys_mode", False):
        if getattr(app, "sys_phase", "") == "capture":
            return CAPTURE
        return SYSTEM
    if getattr(app, "calib_mode", False):
        if getattr(app, "calib_phase", "") == "create":
            return RENAME
        return CALIBRATION
    if getattr(app, "lang_mode", False):
        return LANGUAGE
    if getattr(app, "help_mode", False):
        return HELP
    if getattr(app, "editor_mode", False):
        return EDITOR
    return MAIN


def allows_quit(ctx):
    return ctx == MAIN


def allows_calibration(ctx):
    return ctx == MAIN


def allows_overlay_toggle(ctx):
    return ctx == MAIN


def is_text_context(ctx):
    return ctx in TEXT_CONTEXTS


def allows(ctx, action):
    return action in SHORTCUTS.get(ctx, {})
