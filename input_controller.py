import threading
import time
import traceback
from pynput.keyboard import Controller as KeyboardController, Key, KeyCode, Listener as KeyboardListener
from pynput.mouse import Controller as MouseController, Button

from localization import t

# Overridden from main after settings.json is loaded. Stay quiet in release.
VERBOSE = False

# Duree minimale d'un PRESS. Le press part immediatement ; le release est fait
# par tick() sur une frame suivante. Aucun sleep, aucun thread : la boucle
# webcam n'est jamais bloquee. Mettre 0.0 pour un press/release atomique.
PRESS_MIN_DURATION = 0.04

# Garde-fou : nombre maximal d'etapes executees pour un seul lancement de macro.
MAX_MACRO_STEPS = 2000

KEY_OBJECTS = {
    "SPACE": Key.space, "ENTER": Key.enter, "TAB": Key.tab,
    "SHIFT": Key.shift, "CTRL": Key.ctrl, "ALT": Key.alt,
    "ESC": Key.esc, "BACKSPACE": Key.backspace,
    "UP": Key.up, "DOWN": Key.down, "LEFT": Key.left, "RIGHT": Key.right,
    "HOME": Key.home, "END": Key.end, "PAGE_UP": Key.page_up, "PAGE_DOWN": Key.page_down,
    "INSERT": Key.insert, "DELETE": Key.delete,
    "CAPS_LOCK": Key.caps_lock, "PAUSE": Key.pause,
    "NUM_LOCK": Key.num_lock, "SCROLL_LOCK": Key.scroll_lock,
}
for _index in range(1, 13):
    KEY_OBJECTS[f"F{_index}"] = getattr(Key, f"f{_index}")
MOUSE_OBJECTS = {
    "LEFT_MOUSE": Button.left,
    "RIGHT_MOUSE": Button.right,
    "MIDDLE_MOUSE": Button.middle,
}
# X1/X2 n'existent pas sur toutes les plateformes : on ne les expose que si pynput les fournit.
for _name, _attr in (("X1_MOUSE", "x1"), ("X2_MOUSE", "x2")):
    _button = getattr(Button, _attr, None)
    if _button is not None:
        MOUSE_OBJECTS[_name] = _button

ALIASES = {
    "MOUSE_LEFT": "LEFT_MOUSE", "MOUSE_RIGHT": "RIGHT_MOUSE", "MOUSE_MIDDLE": "MIDDLE_MOUSE",
    "LEFT_CLICK": "LEFT_MOUSE", "RIGHT_CLICK": "RIGHT_MOUSE", "MIDDLE_CLICK": "MIDDLE_MOUSE",
    "MOUSE_X1": "X1_MOUSE", "MOUSE_X2": "X2_MOUSE",
    "ESCAPE": "ESC", "RETURN": "ENTER", "CONTROL": "CTRL",
    "PAGEUP": "PAGE_UP", "PAGEDOWN": "PAGE_DOWN", "PGUP": "PAGE_UP", "PGDN": "PAGE_DOWN",
    "DEL": "DELETE", "INS": "INSERT", "CAPS": "CAPS_LOCK",
}

MODIFIERS = ("CTRL", "SHIFT", "ALT")
KEYBOARD = "key"
MOUSE = "mouse"

STEP_PRESS = "PRESS"
STEP_HOLD = "HOLD"
STEP_RELEASE = "RELEASE"
STEP_WAIT = "WAIT"
STEP_COMBINATION = "COMBINATION"
STEP_REPEAT_BEGIN = "REPEAT_BEGIN"
STEP_REPEAT_END = "REPEAT_END"


def macro_source(owner):
    return f"MACRO:{owner}"


class InputController:
    """Emulation clavier/souris externe. Ne connait ni les gestes ni le jeu.

    Les maintiens sont comptabilises PAR SOURCE (LEFT, RIGHT, MACRO:LEFT, ...).
    Une touche n'est relachee que lorsque plus aucune source ne la demande.
    """

    def __init__(self):
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self.hold_sources = {}              # source -> set((kind, target))
        self.currently_pressed = set()      # union clavier reellement enfoncee
        self.held_buttons = set()           # union souris reellement enfoncee
        self.ephemeral_pressed = set()      # cibles PRESS en attente de release
        self.pending_releases = []          # [(targets_in_release_order, deadline)]
        self.macros = {}                    # owner -> machine a etats
        self.enabled = False
        self.lock = threading.RLock()
        self._listener = None
        self._on_emergency = None
        self._on_preview_toggle = None
        self._capturing = False
        self._captured_name = None
        self._combo_capture = False
        self._held_names = set()
        self._captured_combo = None
        self._preview_allowed = None
        self.special_keys = KEY_OBJECTS
        self.mouse_buttons = MOUSE_OBJECTS

    # ---- Journalisation -----------------------------------------------------
    @staticmethod
    def _log(message):
        if VERBOSE:
            print("[INPUT]", message)

    @staticmethod
    def _error(message):
        """Aucune exception silencieuse : trace complete."""
        print("[INPUT][ERREUR]", message)
        traceback.print_exc()

    @staticmethod
    def _name(target):
        if isinstance(target, str):
            return target.upper()
        for name, obj in MOUSE_OBJECTS.items():
            if target == obj:
                return name
        for name, obj in KEY_OBJECTS.items():
            if target == obj:
                return name
        return getattr(target, "name", str(target)).upper()

    # ---- Resolution ---------------------------------------------------------
    def resolve(self, action):
        """'W' -> (key, 'w') | 'LEFT_MOUSE' -> (mouse, Button.left) | inconnu -> (None, None)."""
        if isinstance(action, Button):
            return MOUSE, action
        if not isinstance(action, str):
            return KEYBOARD, action
        raw = action.strip()
        if not raw:
            return None, None
        upper = ALIASES.get(raw.upper(), raw.upper())
        if upper in MOUSE_OBJECTS:
            return MOUSE, MOUSE_OBJECTS[upper]
        if upper in KEY_OBJECTS:
            return KEYBOARD, KEY_OBJECTS[upper]
        if len(raw) == 1:
            return KEYBOARD, raw.lower()
        return None, None

    def resolve_all(self, actions):
        """Retourne la liste des cibles, ou None si une commande est inconnue."""
        if isinstance(actions, str):
            actions = [actions]
        targets = []
        for action in actions:
            kind, target = self.resolve(action)
            if kind is None:
                self._log(f"commande inconnue ignoree : {action!r}")
                return None
            targets.append((kind, target))
        return targets

    # ---- Etat global --------------------------------------------------------
    def enable(self):
        with self.lock:
            self.enabled = True
        print(t("input.on"))

    def disable(self):
        with self.lock:
            self.enabled = False
        self.stop_all_macros()
        self.release_all()
        print(t("input.off"))

    def toggle(self):
        if self.enabled:
            self.disable()
        else:
            self.enable()

    # ---- Envoi bas niveau ---------------------------------------------------
    def _press_target(self, kind, target):
        """System-wide synthetic input (pynput → Win32 SendInput).

        Not bound to the OpenCV preview. The foreground application receives
        the event. This controller never targets a game window, never forces
        the game to the front, and never injects into another process.
        """
        try:
            if kind == MOUSE:
                self.mouse.press(target)
            else:
                self.keyboard.press(target)
            return True
        except Exception:
            self._error(f"press {self._name(target)}")
            return False

    def _release_target(self, kind, target):
        try:
            if kind == MOUSE:
                self.mouse.release(target)
            else:
                self.keyboard.release(target)
            return True
        except Exception:
            self._error(f"release {self._name(target)}")
            return False

    # ---- HOLD par source ----------------------------------------------------
    def set_source_hold(self, source, actions):
        """Declare ce que CETTE source maintient. Recalcule l'union ensuite.

        Une touche demandee par LEFT et par RIGHT reste enfoncee tant que
        l'une des deux la demande encore.
        """
        targets = set()
        for action in actions or ():
            kind, target = self.resolve(action)
            if kind is None:
                self._log(f"hold ignore : commande inconnue {action!r}")
                continue
            targets.add((kind, target))
        with self.lock:
            if not self.enabled:
                if self.hold_sources:
                    self.hold_sources.clear()
                    self._apply_union()
                return
            if self.hold_sources.get(source, set()) == targets:
                return  # rien n'a change pour cette source
            if targets:
                self.hold_sources[source] = targets
            else:
                self.hold_sources.pop(source, None)
            self._apply_union()

    def clear_source(self, source):
        with self.lock:
            if self.hold_sources.pop(source, None) is None:
                return
            self._apply_union()

    def _apply_union(self):
        """Diff entre l'union demandee et ce qui est reellement enfonce."""
        union = set()
        for targets in self.hold_sources.values():
            union |= targets
        keys = {target for kind, target in union if kind == KEYBOARD}
        buttons = {target for kind, target in union if kind == MOUSE}
        if keys == self.currently_pressed and buttons == self.held_buttons:
            return
        for key in self.currently_pressed - keys:
            self._release_target(KEYBOARD, key)
        for button in self.held_buttons - buttons:
            self._release_target(MOUSE, button)
        for key in keys - self.currently_pressed:
            self.ephemeral_pressed.discard(key)
            self._press_target(KEYBOARD, key)
        for button in buttons - self.held_buttons:
            self.ephemeral_pressed.discard(button)
            self._press_target(MOUSE, button)
        self.currently_pressed = keys
        self.held_buttons = buttons
        held = sorted(self._name(t) for t in (keys | buttons))
        self._log("hold -> " + (" + ".join(held) if held else "rien"))

    def _is_held(self, target):
        return target in self.currently_pressed or target in self.held_buttons

    # ---- PRESS et combinaisons ----------------------------------------------
    def press_combo(self, inputs, duration=PRESS_MIN_DURATION):
        """SHIFT+E -> SHIFT DOWN, E DOWN, puis E UP, SHIFT UP. Sans sleep ni thread."""
        targets = self.resolve_all(inputs)
        if not targets:
            return False
        with self.lock:
            if not self.enabled:
                self._log(f"press ignore : INPUT OFF ({inputs})")
                return False
            pressed = []
            for kind, target in targets:
                if self._is_held(target):
                    continue  # deja maintenu : on ne le touche pas
                if not self._press_target(kind, target):
                    break
                pressed.append((kind, target))
            if not pressed:
                self._log("press ignore : " + "+".join(self._name(t) for _, t in targets) + " deja maintenu")
                return False
            release_order = list(reversed(pressed))
            if duration <= 0:
                for kind, target in release_order:
                    self._release_target(kind, target)
            else:
                for _, target in pressed:
                    self.ephemeral_pressed.add(target)
                self.pending_releases.append((release_order, time.monotonic() + duration))
            self._log("press -> " + "+".join(self._name(t) for _, t in pressed))
            return True

    def tick(self):
        """Une fois par frame : releases arrives a echeance puis avancement des macros."""
        self._flush_releases()
        self.update_macros()

    def _flush_releases(self):
        with self.lock:
            if not self.pending_releases:
                return
            now = time.monotonic()
            remaining = []
            for release_order, deadline in self.pending_releases:
                if now < deadline:
                    remaining.append((release_order, deadline))
                    continue
                for kind, target in release_order:
                    if self._is_held(target):
                        self.ephemeral_pressed.discard(target)  # un HOLD a pris le relais
                        continue
                    if target in self.ephemeral_pressed:
                        self._release_target(kind, target)
                        self.ephemeral_pressed.discard(target)
            self.pending_releases = remaining

    # ---- Macros (machine a etats, jamais bloquante) -------------------------
    def start_macro(self, owner, steps, label="", speed=1.0):
        if not steps:
            return False
        with self.lock:
            if not self.enabled:
                self._log(f"macro ignoree : INPUT OFF ({owner} {label})")
                return False
            if owner in self.macros:
                return False  # deja en cours pour cette main
            self.macros[owner] = {
                "steps": [dict(step) for step in steps],
                "index": 0,
                "next_time": time.monotonic(),
                "label": label,
                "speed": max(0.25, min(4.0, float(speed or 1.0))),
                "stack": [],
                "budget": MAX_MACRO_STEPS,
            }
        print(t("input.macro_start", owner=owner, label=label))
        return True

    def is_macro_running(self, owner):
        with self.lock:
            return owner in self.macros

    def running_macros(self):
        with self.lock:
            return sorted(self.macros)

    def stop_macro(self, owner):
        """Arret immediat : les HOLD poses par la macro sont relaches."""
        with self.lock:
            state = self.macros.pop(owner, None)
        if state is None:
            return False
        self.clear_source(macro_source(owner))
        print(t("input.macro_stop", owner=owner, label=state["label"]))
        return True

    def stop_all_macros(self):
        with self.lock:
            owners = list(self.macros)
        for owner in owners:
            self.stop_macro(owner)

    def _macro_hold(self, owner, targets, add):
        """Ajoute ou retire des cibles au HOLD de la macro, sans toucher aux mains."""
        source = macro_source(owner)
        with self.lock:
            current = set(self.hold_sources.get(source, set()))
            if add:
                current |= set(targets)
            else:
                current -= set(targets)
            if current:
                self.hold_sources[source] = current
            else:
                self.hold_sources.pop(source, None)
            self._apply_union()

    def update_macros(self):
        """Avance chaque macro selon time.monotonic(). Ne relache jamais les HOLD des mains."""
        with self.lock:
            if not self.macros:
                return
            owners = list(self.macros)
            running = self.enabled
        if not running:
            self.stop_all_macros()
            return

        now = time.monotonic()
        for owner in owners:
            self._advance_macro(owner, now)

    def _advance_macro(self, owner, now):
        while True:
            with self.lock:
                state = self.macros.get(owner)
                if state is None or now < state["next_time"]:
                    return
                state["budget"] -= 1
                if state["budget"] <= 0:
                    label = state["label"]
                    del self.macros[owner]
                    self.clear_source(macro_source(owner))
                    print(t("input.macro_abandon", owner=owner, label=label))
                    return
                steps = state["steps"]
                total = len(steps)
                if state["index"] >= total:
                    label = state["label"]
                    del self.macros[owner]
                    self.clear_source(macro_source(owner))
                    print(t("input.macro_end", owner=owner, label=label))
                    return
                position = state["index"]
                step = steps[position]
                state["index"] += 1
                speed = state["speed"]
                stack = state["stack"]
                action = step["action"]

                # Controle de flux : traite sous le lock, sans envoi systeme.
                if action == STEP_REPEAT_BEGIN:
                    stack.append([position, step["count"] - 1])
                    continue
                if action == STEP_REPEAT_END:
                    if stack and stack[-1][1] > 0:
                        stack[-1][1] -= 1
                        state["index"] = stack[-1][0] + 1
                    elif stack:
                        stack.pop()
                    continue

            # Envois systeme hors du lock de decision.
            if action == STEP_WAIT:
                delay = step["duration"] / 1000.0 / speed
                print(t("input.macro_wait", owner=owner, duration=step["duration"]))
                with self.lock:
                    if owner in self.macros:
                        self.macros[owner]["next_time"] = now + delay
                return

            targets = self.resolve_all(step["inputs"])
            label_inputs = "+".join(step["inputs"])
            print(f"[{owner}][MACRO] STEP {position + 1}/{total} {action} {label_inputs}")
            if targets is None:
                print(t("input.macro_skip", owner=owner))
            elif action == STEP_HOLD:
                self._macro_hold(owner, targets, add=True)
            elif action == STEP_RELEASE:
                self._macro_hold(owner, targets, add=False)
            else:  # PRESS ou COMBINATION
                self.press_combo(step["inputs"])

            with self.lock:
                if owner in self.macros:
                    self.macros[owner]["next_time"] = now

    # ---- Arret global -------------------------------------------------------
    def release_all(self):
        """Reserve aux situations globales : preview, Input OFF, fermeture, erreur."""
        self.stop_all_macros()
        with self.lock:
            names = sorted(self._name(t) for t in (self.currently_pressed | self.held_buttons | self.ephemeral_pressed))
            for release_order, _ in self.pending_releases:
                for kind, target in release_order:
                    if target in self.ephemeral_pressed:
                        self._release_target(kind, target)
            for key in self.currently_pressed:
                self._release_target(KEYBOARD, key)
            for button in self.held_buttons:
                self._release_target(MOUSE, button)
            self.hold_sources.clear()
            self.currently_pressed = set()
            self.held_buttons = set()
            self.ephemeral_pressed.clear()
            self.pending_releases.clear()
            if names:
                self._log("release_all : " + " + ".join(names))

    def get_pressed_names(self):
        with self.lock:
            targets = self.currently_pressed | self.held_buttons | self.ephemeral_pressed
        return sorted(self._name(target) for target in targets)

    # ---- Diagnostic ---------------------------------------------------------
    def keyboard_self_test(self, action="W", delay=3.0):
        """Diagnostic manuel. Jamais lance automatiquement : voir KEYBOARD_DIAGNOSTIC dans main.py."""
        print("=" * 50)
        print("[DIAG] Test clavier pynput")
        print("[DIAG] Backend clavier :", type(self.keyboard).__module__)
        print("[DIAG] Backend souris  :", type(self.mouse).__module__)
        print(f"[DIAG] Envoi de '{action}' dans la FENETRE ACTIVE dans {delay:.0f}s.")
        print("[DIAG] Cliquez maintenant dans le Bloc-notes.")
        time.sleep(delay)
        with self.lock:
            previous = self.enabled
            self.enabled = True
        try:
            ok = self.press_combo(action, duration=0.05)
            time.sleep(0.1)
            self._flush_releases()
        finally:
            with self.lock:
                self.enabled = previous
        print("[DIAG] Resultat :", "OK (pynput n'a pas leve d'erreur)" if ok else "ECHEC")
        print("[DIAG] Si rien n'apparait alors que le resultat est OK :")
        print("[DIAG]   - la fenetre active n'etait pas le Bloc-notes,")
        print("[DIAG]   - ou la cible est en mode administrateur (lancez l'app en admin).")
        print("=" * 50)
        return ok

    # ---- Global shortcuts: emergency stop and preview ----------------------
    def start_emergency_listener(self, on_cut=None, on_preview_toggle=None, on_overlay=None, overlay_hotkeys=None):
        """Physical-key capture for mapping. Overlay hotkeys were Game OSD and are gone."""
        self._on_emergency = on_cut
        self._on_preview_toggle = on_preview_toggle
        if self._listener is not None:
            return

        def on_press(key):
            name = canonical_key_name(key)
            if self._capturing:
                if not name:
                    return
                if name in MODIFIERS:
                    self._held_names.add(name)
                    if self._combo_capture:
                        return
                self._captured_name = name
                if self._combo_capture and name not in ("ESC",):
                    ordered = [item for item in MODIFIERS if item in self._held_names]
                    ordered.append(name)
                    self._captured_combo = ordered[:4]
                return

        def on_release(key):
            name = canonical_key_name(key)
            if name in MODIFIERS:
                self._held_names.discard(name)

        self._listener = KeyboardListener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def begin_key_capture(self):
        """Next physical key is stored for the mapping editor. It is not a gesture."""
        self._captured_name = None
        self._captured_combo = None
        self._held_names = set()
        self._combo_capture = False
        self._capturing = True

    def begin_combo_capture(self):
        """Modifiers held, then one final key. Stops as soon as that key arrives."""
        self.begin_key_capture()
        self._combo_capture = True

    def end_key_capture(self):
        self._capturing = False
        self._combo_capture = False
        self._captured_name = None
        self._captured_combo = None
        self._held_names = set()

    def take_captured_key(self):
        name = self._captured_name
        self._captured_name = None
        return name

    def take_captured_combo(self):
        combo = self._captured_combo
        self._captured_combo = None
        self._captured_name = None
        return combo

    def stop_emergency_listener(self):
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                self._error("arret du listener d'urgence")
            self._listener = None


def canonical_key_name(key):
    """pynput key -> the same uppercase token stored in mapping.json."""
    if isinstance(key, KeyCode):
        vk = getattr(key, "vk", None)
        if isinstance(vk, int):
            if 65 <= vk <= 90:
                return chr(vk)
            if 48 <= vk <= 57:
                return chr(vk)
        char = key.char
        if isinstance(char, str) and len(char) == 1 and char.isalnum():
            return char.upper()
        return None
    attr = str(getattr(key, "name", "") or "")
    if attr in ("ctrl", "ctrl_l", "ctrl_r"):
        return "CTRL"
    if attr in ("alt", "alt_l", "alt_r", "alt_gr"):
        return "ALT"
    if attr in ("shift", "shift_l", "shift_r"):
        return "SHIFT"
    for name, candidate in KEY_OBJECTS.items():
        if key == candidate:
            return name
    return None
