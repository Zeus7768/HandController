import json
import logging
import os

from config import atomic_json_write, backup_file
from gesture_engine import gesture_number
from geometry_engine import (
    LEFT_TILT_LEFT, LEFT_TILT_RIGHT, RIGHT_PINCH, SPECIAL_BY_SIDE, SPECIAL_NAMES,
)
from localization import action_label, t

MAPPING_VERSION = 6

SIDES = ("LEFT", "RIGHT")
GESTURE_NAMES = [f"G{i:02d}" for i in range(1, 31)]

# Types de mapping
HOLD = "HOLD"
PRESS = "PRESS"
COMBINATION = "COMBINATION"
MACRO = "MACRO"
NONE = "NONE"
ENTRY_TYPES = (HOLD, PRESS, COMBINATION, MACRO, NONE)
# COMBINATION se comporte comme HOLD : maintenu tant que le geste est reconnu.
MAINTAINED_TYPES = (HOLD, COMBINATION)

# Actions disponibles dans une macro
STEP_PRESS = "PRESS"
STEP_HOLD = "HOLD"
STEP_RELEASE = "RELEASE"
STEP_WAIT = "WAIT"
STEP_COMBINATION = "COMBINATION"
STEP_REPEAT_BEGIN = "REPEAT_BEGIN"
STEP_REPEAT_END = "REPEAT_END"
STEP_ACTIONS = (STEP_PRESS, STEP_COMBINATION, STEP_HOLD, STEP_RELEASE, STEP_WAIT,
                STEP_REPEAT_BEGIN, STEP_REPEAT_END)
INPUT_STEPS = (STEP_PRESS, STEP_COMBINATION, STEP_HOLD, STEP_RELEASE)

MOUSE_ACTIONS = ["LEFT_MOUSE", "RIGHT_MOUSE", "MIDDLE_MOUSE", "X1_MOUSE", "X2_MOUSE"]
NAMED_KEYS = [
    "SPACE", "SHIFT", "CTRL", "ALT", "TAB", "ENTER", "ESC", "BACKSPACE",
    "UP", "DOWN", "LEFT", "RIGHT",
    "HOME", "END", "PAGE_UP", "PAGE_DOWN", "INSERT", "DELETE",
    "CAPS_LOCK", "PAUSE", "NUM_LOCK", "SCROLL_LOCK",
] + [f"F{index}" for index in range(1, 13)]
KEYBOARD_ACTIONS = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [str(d) for d in range(10)]
    + NAMED_KEYS
)
ACTIONS = KEYBOARD_ACTIONS + MOUSE_ACTIONS

ALIASES = {
    "MOUSE_LEFT": "LEFT_MOUSE", "MOUSE_RIGHT": "RIGHT_MOUSE", "MOUSE_MIDDLE": "MIDDLE_MOUSE",
    "LEFT_CLICK": "LEFT_MOUSE", "RIGHT_CLICK": "RIGHT_MOUSE", "MIDDLE_CLICK": "MIDDLE_MOUSE",
    "MOUSE_X1": "X1_MOUSE", "MOUSE_X2": "X2_MOUSE",
    "ESCAPE": "ESC", "RETURN": "ENTER", "CONTROL": "CTRL",
    "PAGEUP": "PAGE_UP", "PAGEDOWN": "PAGE_DOWN", "PGUP": "PAGE_UP", "PGDN": "PAGE_DOWN",
    "DEL": "DELETE", "INS": "INSERT", "CAPS": "CAPS_LOCK",
}
LEGACY_TYPES = {
    "HOLD": HOLD, "PRESS": PRESS, "CLICK": PRESS, "MACRO": MACRO, "NONE": NONE,
    "COMBINATION": COMBINATION, "COMBINAISON": COMBINATION,
}

# Garde-fous
MAX_STEPS = 24
MAX_INPUTS = 4
WAIT_MIN_MS = 0
WAIT_MAX_MS = 5000
WAIT_STEP_MS = 50
REPEAT_MIN = 2
REPEAT_MAX = 20
MAX_REPEAT_DEPTH = 2
DEFAULT_COOLDOWN_MS = 150
COOLDOWN_CHOICES = [0, 50, 100, 150, 200, 300, 500, 1000]
SPEED_MIN = 0.25
SPEED_MAX = 4.0
DEFAULT_SPEED = 1.0
SPEED_CHOICES = [0.5, 1.0, 1.5, 2.0]

_DEFAULT_ACTIONS = [
    ("W", HOLD), ("S", HOLD), ("A", HOLD), ("D", HOLD), ("SPACE", HOLD), ("SHIFT", HOLD),
    ("E", PRESS), ("F", PRESS), ("LEFT_MOUSE", PRESS), ("RIGHT_MOUSE", PRESS),
    ("R", PRESS), ("TAB", PRESS), ("C", PRESS), ("ENTER", PRESS), ("Q", PRESS),
    ("1", PRESS), ("2", PRESS), ("3", PRESS), ("4", PRESS), ("5", PRESS),
    ("6", PRESS), ("7", PRESS), ("8", PRESS), ("9", PRESS), ("0", PRESS),
    ("UP", PRESS), ("DOWN", PRESS), ("LEFT", PRESS), ("RIGHT", PRESS), ("MIDDLE_MOUSE", PRESS),
]

# Gestes speciaux (detecteurs geometriques de geometry_engine). Ils vivent
# dans la table de LEUR main, a cote de G01-G30, et utilisent exactement le
# meme format d'entree : rien n'est code en dur dans les detecteurs, ces
# valeurs ne sont que des defauts modifiables dans Parametres de geste.
#
# Seuls les trois gestes de reference sont actifs par defaut. Tous les autres
# (inclinaisons haut/bas, main droite, direction de l'index, etats neutres)
# sont livres DESACTIVES : brancher d'office une vingtaine de touches
# enverrait des commandes que l'utilisateur n'a pas demandees. Parametres de
# geste les active sans raccourci clavier.
_DEFAULT_SPECIAL_ACTIONS = {
    RIGHT_PINCH: ("SPACE", PRESS),
    LEFT_TILT_LEFT: ("A", HOLD),
    LEFT_TILT_RIGHT: ("D", HOLD),
}


def side_gesture_names(side):
    """Noms mappables d'une main : les 30 gestes appris puis ses gestes speciaux.

    Les speciaux restent des identifiants distincts (RIGHT_PINCH,
    LEFT_TILT_*) : ils ne sont jamais renommes en G31/G32 et n'occupent aucun
    slot de calibration.
    """
    return GESTURE_NAMES + list(SPECIAL_BY_SIDE.get(normalize_side(side), ()))


class MappingError(ValueError):
    """Erreur de validation lisible par l'utilisateur."""


def default_entry(gesture):
    if gesture in SPECIAL_NAMES:
        if gesture not in _DEFAULT_SPECIAL_ACTIONS:
            return {"type": NONE, "inputs": []}
        action, entry_type = _DEFAULT_SPECIAL_ACTIONS[gesture]
    elif gesture_number(gesture) is not None and gesture not in GESTURE_NAMES:
        return {"type": NONE, "inputs": []}
    else:
        action, entry_type = _DEFAULT_ACTIONS[GESTURE_NAMES.index(gesture)]
    entry = {"type": entry_type, "inputs": [action]}
    if entry_type == PRESS:
        entry["cooldown_ms"] = DEFAULT_COOLDOWN_MS
    return entry


def default_side(side):
    return {gesture: default_entry(gesture) for gesture in side_gesture_names(side)}


def normalize_side(value):
    if not isinstance(value, str):
        return None
    upper = value.strip().upper()
    return upper if upper in SIDES else None


def normalize_action(value):
    """Valide une commande. Une commande inconnue n'est jamais envoyee."""
    if not isinstance(value, str):
        return None
    upper = value.strip().upper()
    upper = ALIASES.get(upper, upper)
    return upper if upper in ACTIONS else None


def normalize_inputs(values, label="inputs"):
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)) or not values:
        raise MappingError(t("mapping.error.empty_inputs", label=label))
    cleaned = []
    for value in values:
        action = normalize_action(value)
        if action is None:
            raise MappingError(t("mapping.error.unknown_command", label=label, value=repr(value)))
        if action not in cleaned:
            cleaned.append(action)
    if len(cleaned) > MAX_INPUTS:
        raise MappingError(t("mapping.error.max_inputs", label=label, max=MAX_INPUTS))
    return cleaned


def normalize_type(value):
    if not isinstance(value, str):
        return None
    return LEGACY_TYPES.get(value.strip().upper())


def normalize_cooldown(value):
    if value is None:
        return DEFAULT_COOLDOWN_MS
    try:
        cooldown = int(value)
    except (TypeError, ValueError):
        raise MappingError(t("mapping.error.cooldown_invalid", value=repr(value)))
    if cooldown < 0 or cooldown > 10000:
        raise MappingError(t("mapping.error.cooldown_range", value=cooldown))
    return cooldown


def normalize_speed(value):
    if value is None:
        return DEFAULT_SPEED
    try:
        speed = float(value)
    except (TypeError, ValueError):
        raise MappingError(t("mapping.error.speed_invalid", value=repr(value)))
    if speed < SPEED_MIN or speed > SPEED_MAX:
        raise MappingError(t("mapping.error.speed_range", min=SPEED_MIN, max=SPEED_MAX, value=speed))
    return speed


def normalize_step(step, position=0):
    if not isinstance(step, dict):
        raise MappingError(t("mapping.error.step_struct", position=position))
    raw = step.get("action")
    action = raw.strip().upper() if isinstance(raw, str) else None
    if action not in STEP_ACTIONS:
        raise MappingError(t("mapping.error.step_action", position=position, value=repr(raw)))

    if action == STEP_WAIT:
        try:
            duration = int(step.get("duration", 0))
        except (TypeError, ValueError):
            raise MappingError(t("mapping.error.step_duration", position=position))
        if duration < WAIT_MIN_MS:
            raise MappingError(t("mapping.error.wait_negative", position=position, duration=duration))
        if duration > WAIT_MAX_MS:
            raise MappingError(t("mapping.error.wait_long", position=position, duration=duration, max=WAIT_MAX_MS))
        return {"action": STEP_WAIT, "duration": duration}

    if action == STEP_REPEAT_BEGIN:
        try:
            count = int(step.get("count", REPEAT_MIN))
        except (TypeError, ValueError):
            raise MappingError(t("mapping.error.repeat_invalid", position=position))
        if count < REPEAT_MIN:
            raise MappingError(t("mapping.error.repeat_min", position=position, min=REPEAT_MIN))
        if count > REPEAT_MAX:
            raise MappingError(t("mapping.error.repeat_max", position=position, count=count, max=REPEAT_MAX))
        return {"action": STEP_REPEAT_BEGIN, "count": count}

    if action == STEP_REPEAT_END:
        return {"action": STEP_REPEAT_END}

    inputs = normalize_inputs(step.get("inputs", step.get("input")), f"etape {position}")
    return {"action": action, "inputs": inputs}


def normalize_steps(steps):
    """Valide la liste et l'equilibrage des REPEAT."""
    if not isinstance(steps, list) or not steps:
        raise MappingError(t("mapping.error.empty_macro"))
    if len(steps) > MAX_STEPS:
        raise MappingError(t("mapping.error.macro_long", count=len(steps), max=MAX_STEPS))
    cleaned = [normalize_step(step, index + 1) for index, step in enumerate(steps)]

    depth = 0
    for position, step in enumerate(cleaned, 1):
        if step["action"] == STEP_REPEAT_BEGIN:
            depth += 1
            if depth > MAX_REPEAT_DEPTH:
                raise MappingError(t("mapping.error.repeat_depth", position=position, max=MAX_REPEAT_DEPTH))
        elif step["action"] == STEP_REPEAT_END:
            depth -= 1
            if depth < 0:
                raise MappingError(t("mapping.error.repeat_end", position=position))
    if depth != 0:
        raise MappingError(t("mapping.error.repeat_open"))
    if all(step["action"] in (STEP_REPEAT_BEGIN, STEP_REPEAT_END) for step in cleaned):
        raise MappingError(t("mapping.error.macro_useless"))
    return cleaned


def orphan_releases(steps):
    """RELEASE d'une commande jamais mise en HOLD plus haut : sans effet, donc signale."""
    held = set()
    orphans = []
    for step in steps:
        if step["action"] == STEP_HOLD:
            held.update(step["inputs"])
        elif step["action"] == STEP_RELEASE:
            for action in step["inputs"]:
                if action not in held and action not in orphans:
                    orphans.append(action)
    return orphans


def describe(entry):
    """Texte court pour les menus : 'W+D', 'MACRO 5 etapes', 'DESACTIVE'."""
    if not entry:
        return "?"
    if entry["type"] == MACRO:
        return t("mapping.macro_steps", count=len(entry.get("steps", [])))
    if entry["type"] == NONE:
        return action_label(NONE)
    if entry["type"] == COMBINATION:
        inputs = list(entry.get("inputs") or [])
        modes = combination_modes(entry)
        if any(mode == PRESS for mode in modes):
            parts = [f"{key}:{mode[0]}" for key, mode in zip(inputs, modes)]
            return "+".join(parts) if parts else "+"
        return "+".join(inputs) if inputs else "+"
    return "+".join(entry.get("inputs", []))


def normalize_modes(raw_modes, inputs):
    """One PRESS/HOLD per combination key. Missing slots default to HOLD."""
    inputs = list(inputs or [])
    raw = list(raw_modes or [])
    modes = []
    for index, _key in enumerate(inputs):
        item = raw[index] if index < len(raw) else HOLD
        modes.append(PRESS if str(item).strip().upper() == PRESS else HOLD)
    return modes


def combination_modes(entry):
    return normalize_modes((entry or {}).get("modes"), (entry or {}).get("inputs") or [])


def combination_hold_inputs(entry):
    inputs = list((entry or {}).get("inputs") or [])
    return [key for key, mode in zip(inputs, combination_modes(entry)) if mode == HOLD]


def combination_press_inputs(entry):
    inputs = list((entry or {}).get("inputs") or [])
    return [key for key, mode in zip(inputs, combination_modes(entry)) if mode == PRESS]


def mapping_hold_inputs(entry):
    """Keys that stay down while the gesture is recognized."""
    if not entry or entry.get("enabled", True) is False:
        return []
    kind = entry.get("type")
    if kind == HOLD:
        return list(entry.get("inputs") or [])
    if kind == COMBINATION:
        return combination_hold_inputs(entry)
    return []


def describe_step(step):
    action = step["action"]
    label = action_label(action)
    if action == STEP_WAIT:
        return t("macro.wait_ms", label=label, duration=step["duration"])
    if action == STEP_REPEAT_BEGIN:
        return t("macro.repeat_begin", label=label, count=step["count"])
    if action == STEP_REPEAT_END:
        return label
    return f"{label} {'+'.join(step['inputs'])}"


def preview(steps):
    """Apercu ASCII : SPACE --150ms--> LEFT_MOUSE --150ms--> LEFT_MOUSE"""
    if not steps:
        return ""
    parts = []
    for step in steps:
        action = step["action"]
        if action == STEP_WAIT:
            parts.append(f"--{step['duration']}ms-->")
        elif action == STEP_REPEAT_BEGIN:
            parts.append(f"[x{step['count']}")
        elif action == STEP_REPEAT_END:
            parts.append("]")
        elif action == STEP_HOLD:
            parts.append(f"(+{'+'.join(step['inputs'])})")
        elif action == STEP_RELEASE:
            parts.append(f"(-{'+'.join(step['inputs'])})")
        else:
            parts.append("+".join(step["inputs"]))
    return " ".join(parts)


def _keep_enabled(raw, entry):
    if isinstance(raw, dict) and raw.get("enabled") is False:
        entry["enabled"] = False
    return entry


class GestureMapping:
    """Seul pont (LEFT|RIGHT):G01-G30 -> commandes clavier/souris.

    Format v4 : deux tables independantes, une par main.
    LEFT:G01 et RIGHT:G01 sont deux entrees distinctes.
    """

    def __init__(self, mapping_file="mapping.json"):
        self.mapping_file = mapping_file
        self.last_error = ""
        self.last_warning = ""
        self.mapping = self.load()

    # ---- Validation ---------------------------------------------------------
    def build_entry(self, raw):
        """Accepte v4, v3, v2, v1 et la forme plate. Leve MappingError si invalide."""
        if isinstance(raw, str):
            # Forme plate {"G01": "W"} : aucun type fourni.
            # Migration sure -> PRESS, aucune touche ne peut rester enfoncee.
            return {"type": PRESS, "inputs": normalize_inputs(raw), "cooldown_ms": DEFAULT_COOLDOWN_MS}
        if not isinstance(raw, dict):
            raise MappingError(t("mapping.error.invalid_struct"))

        entry_type = normalize_type(raw.get("type"))
        if entry_type is None:
            raise MappingError(t("mapping.error.unknown_type", value=repr(raw.get("type"))))

        if entry_type == MACRO:
            entry = {"type": MACRO, "steps": normalize_steps(raw.get("steps"))}
            speed = normalize_speed(raw.get("speed"))
            if speed != DEFAULT_SPEED:
                entry["speed"] = speed
            return _keep_enabled(raw, entry)
        if entry_type == NONE:
            return _keep_enabled(raw, {"type": NONE, "inputs": []})

        # v4/v3 "inputs", v2 "action", v1 "input"
        raw_inputs = raw.get("inputs")
        if raw_inputs is None:
            raw_inputs = raw.get("action")
        if raw_inputs is None:
            raw_inputs = raw.get("input")
        entry = {"type": entry_type, "inputs": normalize_inputs(raw_inputs)}
        if entry_type == PRESS:
            entry["cooldown_ms"] = normalize_cooldown(raw.get("cooldown_ms"))
        if entry_type == COMBINATION:
            entry["modes"] = normalize_modes(raw.get("modes"), entry["inputs"])
        return _keep_enabled(raw, entry)

    def _clean_side(self, raw, label):
        names = list(side_gesture_names(label))
        cleaned = {}
        extras = []
        if isinstance(raw, dict):
            for gesture, entry in raw.items():
                learned = gesture_number(gesture) is not None
                if gesture not in names and not learned:
                    print(t("mapping.unknown_gesture", side=label, gesture=gesture))
                    continue
                try:
                    cleaned[gesture] = self.build_entry(entry)
                except MappingError as error:
                    print(t("mapping.invalid_ignored", side=label, gesture=gesture, error=error))
                    continue
                if gesture not in names:
                    extras.append(gesture)
        for gesture in names:
            cleaned.setdefault(gesture, default_entry(gesture))
        ordered = names + extras
        return {gesture: cleaned[gesture] for gesture in ordered if gesture in cleaned}

    def _acceptable(self, side, gesture):
        return gesture in side_gesture_names(side) or gesture_number(gesture) is not None

    # ---- Chargement / sauvegarde -------------------------------------------
    def load(self):
        if not os.path.exists(self.mapping_file):
            self.mapping = {side: default_side(side) for side in SIDES}
            self.save()
            return self.mapping

        try:
            with open(self.mapping_file, "r", encoding="utf-8-sig") as file:
                loaded = json.load(file)
        except json.JSONDecodeError as error:
            print(t("mapping.invalid_file"))
            logging.getLogger("handcontroller").error("mapping.json is invalid: %s", error)
            backup_file(self.mapping_file)
            fallback = {side: default_side(side) for side in SIDES}
            self.mapping = fallback
            self.save()
            return fallback
        except OSError as error:
            print(t("mapping.unread"))
            logging.getLogger("handcontroller").error("Unable to read mapping.json: %s", error)
            backup_file(self.mapping_file)
            return {side: default_side(side) for side in SIDES}

        if not isinstance(loaded, dict):
            loaded = {}

        if any(normalize_side(key) for key in loaded):
            self.mapping = {side: self._clean_side(loaded.get(side), side) for side in SIDES}
            migrated = int(loaded.get("version", 0)) != MAPPING_VERSION
        else:
            # Ancien format sans information de main. On n'invente aucune
            # association : le meme mapping est applique aux DEUX mains a
            # l'identique, ce qui reproduit exactement le comportement
            # precedent. Utiliser Parametres de geste pour les differencier.
            print("=" * 60)
            print(t("mapping.legacy"))
            print("=" * 60)
            source = {key: value for key, value in loaded.items() if key != "version"}
            self.mapping = {side: self._clean_side(source, side) for side in SIDES}
            migrated = True

        if migrated:
            print(t("mapping.migrated", version=MAPPING_VERSION))
            self.save()
        return self.mapping

    def save(self):
        log = logging.getLogger("handcontroller")
        backup_file(self.mapping_file)
        try:
            directory = os.path.dirname(os.path.abspath(self.mapping_file))
            if directory:
                os.makedirs(directory, exist_ok=True)
            payload = {"version": MAPPING_VERSION}
            payload.update({side: self.mapping[side] for side in SIDES})
            atomic_json_write(self.mapping_file, payload)
            return True
        except OSError as error:
            print(t("mapping.unwritable"))
            log.error("Unable to save mapping.json: %s", error)
            return False

    # ---- Lecture -----------------------------------------------------------
    def get(self, side, gesture):
        side = normalize_side(side)
        if side is None or not self._acceptable(side, gesture):
            return None
        return self.mapping[side].get(gesture) or (default_entry(gesture) if gesture_number(gesture) else None)

    def get_type(self, side, gesture):
        entry = self.get(side, gesture)
        return entry["type"] if entry else None

    def get_inputs(self, side, gesture):
        entry = self.get(side, gesture)
        return list(entry.get("inputs", [])) if entry else []

    def get_steps(self, side, gesture):
        entry = self.get(side, gesture)
        return [dict(step) for step in entry.get("steps", [])] if entry else []

    def get_cooldown(self, side, gesture):
        entry = self.get(side, gesture)
        return int(entry.get("cooldown_ms", DEFAULT_COOLDOWN_MS)) if entry else DEFAULT_COOLDOWN_MS

    def get_speed(self, side, gesture):
        entry = self.get(side, gesture)
        return float(entry.get("speed", DEFAULT_SPEED)) if entry else DEFAULT_SPEED

    def describe(self, side, gesture):
        return describe(self.get(side, gesture))

    # ---- Ecriture ----------------------------------------------------------
    def _store(self, side, gesture, entry):
        side = normalize_side(side)
        if side is None or not self._acceptable(side, gesture):
            self.last_error = t("mapping.unknown_target", side=side, gesture=gesture)
            print(t("mapping.refused", detail=self.last_error))
            return False
        self.mapping[side][gesture] = entry
        self.last_error = ""
        self.save()
        return True

    def set_command(self, side, gesture, entry_type, inputs, cooldown_ms=None, modes=None):
        """HOLD, PRESS ou COMBINATION. inputs = ['W'] ou ['W', 'D']."""
        try:
            clean_type = normalize_type(entry_type)
            if clean_type not in (HOLD, PRESS, COMBINATION):
                raise MappingError(t("mapping.error.invalid_type", value=repr(entry_type)))
            entry = {"type": clean_type, "inputs": normalize_inputs(inputs)}
            if clean_type == PRESS:
                entry["cooldown_ms"] = normalize_cooldown(cooldown_ms)
            if clean_type == COMBINATION:
                entry["modes"] = normalize_modes(modes, entry["inputs"])
        except MappingError as error:
            self.last_error = str(error)
            print(t("mapping.refused", detail=f"{side}:{gesture} -> {error}"))
            return False
        return self._store(side, gesture, entry)

    def set_enabled(self, side, gesture, enabled):
        """Keep the mapping, only flip the enabled flag."""
        entry = self.get(side, gesture)
        if entry is None:
            return False
        stored = dict(entry)
        if enabled:
            stored.pop("enabled", None)
        else:
            stored["enabled"] = False
        return self._store(side, gesture, stored)

    def set_type(self, side, gesture, entry_type):
        """Change le type sans toucher aux commandes."""
        entry = self.get(side, gesture)
        if entry is None or entry["type"] == MACRO:
            return False
        return self.set_command(side, gesture, entry_type, entry.get("inputs"),
                                entry.get("cooldown_ms"))

    def set_cooldown(self, side, gesture, cooldown_ms):
        entry = self.get(side, gesture)
        if entry is None or entry["type"] != PRESS:
            return False
        return self.set_command(side, gesture, PRESS, entry["inputs"], cooldown_ms)

    def set_macro(self, side, gesture, steps, speed=None):
        try:
            clean_steps = normalize_steps(steps)
            entry = {"type": MACRO, "steps": clean_steps}
            clean_speed = normalize_speed(speed)
            if clean_speed != DEFAULT_SPEED:
                entry["speed"] = clean_speed
        except MappingError as error:
            self.last_error = str(error)
            print(t("mapping.macro_refused", side=side, gesture=gesture, error=error))
            return False
        orphans = orphan_releases(clean_steps)
        self.last_warning = t("mapping.error.orphan_release", keys=", ".join(orphans)) if orphans else ""
        if orphans:
            print(t("mapping.macro_warn", side=side, gesture=gesture, warning=self.last_warning))
        return self._store(side, gesture, entry)

    def disable(self, side, gesture):
        return self._store(side, gesture, {"type": NONE, "inputs": []})

    def reset(self, side, gesture):
        if gesture not in side_gesture_names(side):
            return False
        return self._store(side, gesture, default_entry(gesture))

    def reset_all(self):
        """Restore every LEFT/RIGHT entry to the built-in defaults. Backs up first."""
        self.mapping = {side: default_side(side) for side in SIDES}
        self.last_error = ""
        self.last_warning = ""
        return self.save()

    # ---- Affichage ---------------------------------------------------------
    def display(self, side=None):
        sides = [normalize_side(side)] if normalize_side(side) else list(SIDES)
        print("\n================================")
        print(f"            {t('mapping.title')}")
        print("================================")
        for current in sides:
            label = t("hand.left") if current == "LEFT" else t("hand.right")
            print(f"\n--- {label} ({current}) ---")
            for gesture in side_gesture_names(current):
                if gesture in SPECIAL_NAMES and gesture == SPECIAL_BY_SIDE[current][0]:
                    print(t("mapping.specials"))
                entry = self.mapping[current][gesture]
                extra = ""
                if entry["type"] == PRESS:
                    extra = f"  cooldown {entry.get('cooldown_ms', DEFAULT_COOLDOWN_MS)} ms"
                elif entry["type"] == MACRO:
                    extra = "  " + preview(entry["steps"])
                print(f"[{gesture}] {describe(entry):<24} [{action_label(entry['type'])}]{extra}")
        print("================================\n")
