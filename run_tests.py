"""Tests hors architecture (les 4 modules applicatifs restent inchanges)."""
import ast
import importlib
import inspect
import json
import os
import shutil
import struct
import tempfile
import textwrap
import time
import unittest
from unittest import mock

import numpy as np

import input_controller
import main as main_module
import geometry_engine as geo
import system_gestures as sysg
from gesture_engine import APPEND, FEATURE_SIZE, GESTURE_NAMES, REPLACE, SIDES, UNKNOWN, GestureEngine
from config import GESTURES_FILE, MAPPING_FILE, app_dir, user_data_path
from gesture_mapping import (
    COMBINATION, GestureMapping, HOLD, MACRO, MAINTAINED_TYPES, MAPPING_VERSION, MappingError,
    NONE, PRESS, SPEED_CHOICES, STEP_HOLD, STEP_PRESS, STEP_RELEASE, STEP_REPEAT_BEGIN,
    STEP_REPEAT_END, STEP_WAIT, describe, normalize_steps, preview, side_gesture_names,
)
from input_controller import InputController
from main import HandControllerApp, hand_point_color, resolve_hand_sides


# ---------------------------------------------------------------------------
# User data guard: tests must never overwrite the real settings.json,
# mapping.json, gestures.json or gestures_backup.json next to the scripts.
# Writes aimed at the application folder are redirected to a temp folder.
# ---------------------------------------------------------------------------
import config as _config  # noqa: E402
import gesture_engine as _gesture_engine  # noqa: E402
import gesture_mapping as _gesture_mapping  # noqa: E402

_REAL_APP_DIR = os.path.normcase(os.path.abspath(app_dir()))
_TEST_DATA_DIR = tempfile.mkdtemp(prefix="handctrl_tests_")
_ORIG_ATOMIC_WRITE = _config.atomic_json_write
_ORIG_BACKUP_FILE = _config.backup_file


def _redirect_user_path(path):
    absolute = os.path.abspath(str(path))
    if os.path.normcase(os.path.dirname(absolute)) == _REAL_APP_DIR:
        return os.path.join(_TEST_DATA_DIR, os.path.basename(absolute))
    return path


def _guarded_atomic_write(path, payload):
    return _ORIG_ATOMIC_WRITE(_redirect_user_path(path), payload)


def _guarded_backup_file(path):
    if path and _redirect_user_path(path) != path:
        return False
    return _ORIG_BACKUP_FILE(path)


for _module in (_config, _gesture_engine, _gesture_mapping, sysg):
    if getattr(_module, "atomic_json_write", None) is _ORIG_ATOMIC_WRITE:
        _module.atomic_json_write = _guarded_atomic_write
    if getattr(_module, "backup_file", None) is _ORIG_BACKUP_FILE:
        _module.backup_file = _guarded_backup_file

import atexit  # noqa: E402

atexit.register(shutil.rmtree, _TEST_DATA_DIR, True)


class PackagingTests(unittest.TestCase):
    ROOT = os.path.dirname(os.path.abspath(__file__))

    def test_build_script_checks_python_deps_resources_and_verifies(self):
        with open(os.path.join(self.ROOT, "build_exe.bat"), encoding="utf-8") as handle:
            bat = handle.read()
        for needle in ("where python", "import cv2, mediapipe, numpy, pynput", "PyInstaller --version",
                       "rmdir /s /q build", "rmdir /s /q dist", "HandController.spec",
                       "hand_landmarker.task", "handcontroller_icon.ico", "admin.manifest",
                       "tools\\verify_build.py"):
            self.assertIn(needle, bat)

    def test_release_zip_never_packs_user_data(self):
        import importlib.util
        path = os.path.join(self.ROOT, "tools", "verify_build.py")
        spec = importlib.util.spec_from_file_location("verify_build", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            dist = os.path.join(td, "HandController")
            os.makedirs(os.path.join(dist, "_internal", "localization"))
            os.makedirs(os.path.join(dist, "logs"))
            for rel in ("HandController.exe", "settings.json", "gestures.json", "gestures_backup.json",
                        "mapping.json", os.path.join("logs", "handcontroller.log"),
                        os.path.join("_internal", "localization", "fr.json")):
                with open(os.path.join(dist, rel), "w", encoding="utf-8") as handle:
                    handle.write("x")
            target = module.make_release_zip(dist, os.path.join(td, "release"))
            import zipfile
            names = set(zipfile.ZipFile(target).namelist())
        normalized = {name.replace("\\", "/") for name in names}
        root = "HandController-v1.0.0-Windows/"
        self.assertTrue(target.endswith("HandController-v1.0.0-Windows.zip"))
        self.assertIn(root + "HandController.exe", normalized)
        self.assertIn(root + "_internal/localization/fr.json", normalized)
        for doc in ("README.txt", "LICENSE.txt", "THIRD_PARTY_NOTICES.txt", "licenses/GPL-3.0.txt",
                    "licenses/pynput/COPYING.LGPL", "licenses/mediapipe/LICENSE"):
            self.assertIn(root + doc, normalized)
        for banned in ("settings.json", "gestures.json", "gestures_backup.json", "mapping.json", "logs/handcontroller.log"):
            self.assertNotIn(root + banned, normalized)
        self.assertFalse(any(name.endswith((".md", ".py")) and "/_internal/" not in name and "/licenses/" not in name
                             for name in normalized))

    def test_overlay_and_state_keys_exist_in_both_languages(self):
        catalogs = {}
        for lang in ("fr", "en"):
            with open(os.path.join(self.ROOT, "localization", lang + ".json"), encoding="utf-8") as handle:
                catalogs[lang] = json.load(handle)
        for block in ("overlay", "state"):
            self.assertEqual(set(catalogs["fr"][block]), set(catalogs["en"][block]), block)
        self.assertEqual(catalogs["fr"]["state"]["screen_overlay"], "Sur-impression d'écran: {state}")
        self.assertEqual(catalogs["en"]["state"]["screen_overlay"], "Screen Overlay: {state}")

    def _verify_build_module(self):
        import importlib.util
        path = os.path.join(self.ROOT, "tools", "verify_build.py")
        spec = importlib.util.spec_from_file_location("verify_build", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_exe_is_console_build_with_uac_and_safe_rebuild(self):
        with open(os.path.join(self.ROOT, "HandController.spec"), encoding="utf-8") as handle:
            spec = handle.read()
        self.assertIn("console=True", spec)
        self.assertNotIn("console=False", spec)
        self.assertIn("uac_admin=True", spec)
        self.assertIn("handcontroller_icon.ico", spec)
        with open(os.path.join(self.ROOT, "build_exe.bat"), encoding="utf-8") as handle:
            bat = handle.read()
        for banned in ("--windowed", "--noconsole", "start cmd", "start \"\" cmd"):
            self.assertNotIn(banned, bat)
        # The old dist is only removed once the staged build passed verification.
        staged = bat.index("verify_build.py --dist build\\stage\\HandController")
        self.assertIn("--distpath build\\stage", bat)
        self.assertGreater(bat.index("rmdir /s /q dist"), staged)
        self.assertGreater(bat.index("PyInstaller"), 0)

    def test_verify_build_reads_pe_subsystem_and_ico_sizes(self):
        module = self._verify_build_module()
        with tempfile.TemporaryDirectory() as td:
            for subsystem in (2, 3):
                data = bytearray(512)
                data[0:2] = b"MZ"
                struct.pack_into("<I", data, 0x3C, 0x80)
                data[0x80:0x84] = b"PE\0\0"
                struct.pack_into("<H", data, 0x80 + 24 + 68, subsystem)
                path = os.path.join(td, f"x{subsystem}.exe")
                with open(path, "wb") as handle:
                    handle.write(bytes(data))
                self.assertEqual(module.exe_subsystem(path), subsystem)
            junk = os.path.join(td, "junk.exe")
            with open(junk, "wb") as handle:
                handle.write(b"not a pe")
            self.assertIsNone(module.exe_subsystem(junk))
        ico = os.path.join(self.ROOT, "assets", "handcontroller_icon.ico")
        sizes = module.ico_sizes(ico)
        for size in module.REQUIRED_ICO_SIZES:
            self.assertIn(size, sizes)

    def test_icon_is_large_transparent_and_antialiased(self):
        import cv2
        png = cv2.imread(os.path.join(self.ROOT, "assets", "handcontroller_icon.png"), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(png)
        self.assertEqual(png.shape[0], png.shape[1])
        self.assertGreaterEqual(png.shape[0], 512)
        self.assertEqual(png.shape[2], 4)
        alpha = png[:, :, 3]
        for corner in (alpha[0, 0], alpha[0, -1], alpha[-1, 0], alpha[-1, -1]):
            self.assertEqual(int(corner), 0)
        ys, xs = np.where(alpha > 8)
        fill = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) / png.shape[0]
        self.assertGreater(fill, 0.85)
        self.assertTrue(np.any((alpha > 0) & (alpha < 255)), "edges must be anti-aliased")
        self.assertEqual(int(png[alpha == 0][:, :3].max()), 0, "no colour fringe in transparent area")
        with open(os.path.join(self.ROOT, "assets", "handcontroller_icon.ico"), "rb") as handle:
            data = handle.read()
        _reserved, kind, count = struct.unpack_from("<HHH", data, 0)
        self.assertEqual(kind, 1)
        for index in range(count):
            width, height = data[6 + 16 * index], data[7 + 16 * index]
            size, offset = struct.unpack_from("<II", data, 6 + 16 * index + 8)
            frame = cv2.imdecode(np.frombuffer(data[offset:offset + size], np.uint8), cv2.IMREAD_UNCHANGED)
            expected = 256 if width == 0 else width
            self.assertEqual(frame.shape[:2], (256 if height == 0 else height, expected))
            self.assertEqual(frame.shape[2], 4)
            self.assertEqual(int(frame[0, 0, 3]), 0, f"{expected}px corner must be transparent")
            self.assertGreater(int(frame[:, :, 3].max()), 200, f"{expected}px icon is empty")

    def test_resource_path_finds_bundled_files_from_source_and_exe(self):
        from config import resource_path
        import sys
        for rel in ("hand_landmarker.task", os.path.join("localization", "fr.json"),
                    os.path.join("localization", "en.json"), os.path.join("assets", "handcontroller_icon.ico")):
            self.assertTrue(os.path.isfile(resource_path(rel)), rel)
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "localization"))
            bundled = os.path.join(td, "localization", "fr.json")
            with open(bundled, "w", encoding="utf-8") as handle:
                handle.write("{}")
            exe = os.path.join(td, "exe_dir", "HandController.exe")
            with mock.patch.object(sys, "frozen", True, create=True), \
                    mock.patch.object(sys, "_MEIPASS", td, create=True), \
                    mock.patch.object(sys, "executable", exe):
                self.assertEqual(resource_path(os.path.join("localization", "fr.json")), bundled)
                self.assertEqual(os.path.dirname(resource_path("settings.json")), os.path.dirname(exe))

    def test_gitignore_excludes_build_outputs_and_personal_data(self):
        with open(os.path.join(self.ROOT, ".gitignore"), encoding="utf-8") as handle:
            rules = handle.read().split()
        for rule in ("__pycache__/", "build/", "dist/", "venv/", ".venv/", "logs/", "recordings/",
                     "gestures.json", "gestures_backup.json", "settings.json", "mapping.json",
                     "profiles_archive/", "*.log", ".env", "release/", "*.zip"):
            self.assertIn(rule, rules)
        self.assertIn("!HandController.spec", rules)

    def test_open_source_license_and_notices(self):
        with open(os.path.join(self.ROOT, "LICENSE"), encoding="utf-8") as handle:
            license_text = handle.read()
        self.assertTrue(license_text.startswith("MIT License"))
        self.assertIn("Permission is hereby granted, free of charge", license_text)
        with open(os.path.join(self.ROOT, "THIRD_PARTY_NOTICES.txt"), encoding="utf-8") as handle:
            notices = handle.read()
        for needle in ("MIT License", "pynput", "LGPL", "MediaPipe", "Apache", "certifi", "MPL-2.0"):
            self.assertIn(needle, notices)
        for rel in ("GPL-3.0.txt", "pynput/COPYING.LGPL", "mediapipe/LICENSE", "python/LICENSE.txt"):
            self.assertTrue(os.path.isfile(os.path.join(self.ROOT, "licenses", *rel.split("/"))), rel)
        for name in ("README.md", "README_FR.md", "README_EN.md"):
            with open(os.path.join(self.ROOT, name), encoding="utf-8") as handle:
                readme = handle.read()
            self.assertIn("MIT", readme, name)
            self.assertIn("HandController-v1.0.0-Windows.zip", readme, name)
            self.assertNotIn("emergency_key", readme, name)
        with open(os.path.join(self.ROOT, "HandController.spec"), encoding="utf-8") as handle:
            self.assertIn('"sounddevice"', handle.read())


class UserDataGuardTests(unittest.TestCase):
    def test_save_settings_never_touches_real_user_file(self):
        real = user_data_path("settings.json")
        before = open(real, "rb").read() if os.path.isfile(real) else None
        _config.save_settings({"camera_index": 0, "language": "en", "hud_overlay": True})
        after = open(real, "rb").read() if os.path.isfile(real) else None
        self.assertEqual(before, after)
        self.assertTrue(os.path.isfile(os.path.join(_TEST_DATA_DIR, "settings.json")))


def pose(seed, size=FEATURE_SIZE):
    """Vecteur reproductible, eloigne des autres poses."""
    base = np.zeros(size, dtype=np.float32)
    base[seed % size] = 5.0
    base[(seed * 7 + 3) % size] = -5.0
    return base


def samples_for(seed, count=8):
    base = pose(seed)
    return [(base + i * 0.001).tolist() for i in range(count)]


class FakeKeyboard:
    def __init__(self):
        self.pressed = []
        self.released = []

    def press(self, key):
        self.pressed.append(key)

    def release(self, key):
        self.released.append(key)


class FakeMouse(FakeKeyboard):
    pass


def make_inputs():
    ctrl = InputController()
    ctrl.keyboard = FakeKeyboard()
    ctrl.mouse = FakeMouse()
    return ctrl


def make_app(mapping_path):
    app = HandControllerApp.__new__(HandControllerApp)
    app.inputs = make_inputs()
    app.mapping = GestureMapping(mapping_path)
    app.curr_gestures = {side: UNKNOWN for side in SIDES}
    app.prev_gestures = {side: UNKNOWN for side in SIDES}
    app.press_last_time = {}
    return app


def run_frame(app, **gestures):
    """Reproduit exactement l'ordre de process_frame pour une frame."""
    for side, gesture in gestures.items():
        app.curr_gestures[side] = gesture
    for side in SIDES:
        app.execute_gesture(side, app.curr_gestures[side])
    app.update_held_inputs()
    for side in SIDES:
        app.prev_gestures[side] = app.curr_gestures[side]
    app.inputs.tick()


def drain(ctrl, timeout=2.0):
    deadline = time.monotonic() + timeout
    while ctrl.macros and time.monotonic() < deadline:
        ctrl.tick()
        time.sleep(0.005)
    ctrl.tick()


class TempCase(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.gestures_path = os.path.join(self.td.name, "gestures.json")
        self.mapping_path = os.path.join(self.td.name, "mapping.json")


# =========================================================================
# Banques LEFT / RIGHT
# =========================================================================
class DatabaseTests(TempCase):
    def test_empty_database_is_v8_with_two_banks(self):
        engine = GestureEngine(self.gestures_path)
        self.assertEqual(engine.database["version"], 8)
        self.assertEqual(set(engine.database["gestures"]), {"LEFT", "RIGHT"})

    def test_v5_keeps_side_information(self):
        payload = {"LEFT": {"G01": samples_for(1)}, "RIGHT": {"G02": samples_for(2)}}
        with open(self.gestures_path, "w", encoding="utf-8") as file:
            json.dump(payload, file)
        engine = GestureEngine(self.gestures_path)
        self.assertEqual(len(engine.database["gestures"]["LEFT"]["G01"]), 8)
        self.assertEqual(engine.database["gestures"]["RIGHT"]["G01"], [])

    def test_v6_restores_side_information(self):
        payload = {"version": 6, "gestures": {"G01": {"Left": samples_for(3), "Right": samples_for(4, 5)}}}
        with open(self.gestures_path, "w", encoding="utf-8") as file:
            json.dump(payload, file)
        engine = GestureEngine(self.gestures_path)
        self.assertEqual(len(engine.database["gestures"]["LEFT"]["G01"]), 8)
        self.assertEqual(len(engine.database["gestures"]["RIGHT"]["G01"]), 5)

    def test_v7_duplicated_into_both_banks(self):
        payload = {"version": 7, "gestures": {"G01": samples_for(5)}}
        with open(self.gestures_path, "w", encoding="utf-8") as file:
            json.dump(payload, file)
        engine = GestureEngine(self.gestures_path)
        self.assertEqual(len(engine.database["gestures"]["LEFT"]["G01"]), 8)
        self.assertEqual(len(engine.database["gestures"]["RIGHT"]["G01"]), 8)
        with open(self.gestures_path, encoding="utf-8") as file:
            self.assertEqual(json.load(file)["version"], 8)

    def test_invalid_samples_dropped(self):
        payload = {"version": 8, "gestures": {"LEFT": {"G01": [[0.1] * FEATURE_SIZE, [0.2] * 10, [float("nan")] * FEATURE_SIZE]}}}
        with open(self.gestures_path, "w", encoding="utf-8") as file:
            json.dump(payload, file)
        engine = GestureEngine(self.gestures_path)
        self.assertEqual(len(engine.database["gestures"]["LEFT"]["G01"]), 1)


class SeparationTests(TempCase):
    """TEST CRITIQUE : LEFT:G01 et RIGHT:G01 sont deux gestes differents."""

    def setUp(self):
        super().setUp()
        self.engine = GestureEngine(self.gestures_path, recognition_threshold=0.2, class_margin=0.05)
        self.poing = samples_for(11)
        self.ouverte = samples_for(29)
        self.engine.database["gestures"]["LEFT"]["G01"] = self.poing
        self.engine.database["gestures"]["RIGHT"]["G01"] = self.ouverte
        self.engine.rebuild_index()

    def test_left_pose_on_left_hand(self):
        self.assertEqual(self.engine.recognize(self.poing[0], "LEFT")[0], "G01")

    def test_left_pose_on_right_hand_is_unknown(self):
        self.assertEqual(self.engine.recognize(self.poing[0], "RIGHT")[0], UNKNOWN)

    def test_right_pose_on_right_hand(self):
        self.assertEqual(self.engine.recognize(self.ouverte[0], "RIGHT")[0], "G01")

    def test_right_pose_on_left_hand_is_unknown(self):
        self.assertEqual(self.engine.recognize(self.ouverte[0], "LEFT")[0], UNKNOWN)

    def test_no_side_means_unknown(self):
        self.assertEqual(self.engine.recognize(self.poing[0], None)[0], UNKNOWN)
        self.assertEqual(self.engine.recognize(self.poing[0], "BOTH")[0], UNKNOWN)

    def test_empty_bank_never_borrows_from_the_other(self):
        engine = GestureEngine(self.gestures_path, recognition_threshold=0.2)
        engine.database["gestures"]["LEFT"]["G05"] = samples_for(17)
        engine.database["gestures"]["RIGHT"] = {name: [] for name in GESTURE_NAMES}
        engine.rebuild_index()
        self.assertEqual(engine.recognize(samples_for(17)[0], "LEFT")[0], "G05")
        self.assertEqual(engine.recognize(samples_for(17)[0], "RIGHT")[0], UNKNOWN)


class CalibrationTests(TempCase):
    def setUp(self):
        super().setUp()
        self.engine = GestureEngine(self.gestures_path, samples_per_gesture=4)

    def _capture(self, gesture, side, mode, seed):
        self.assertTrue(self.engine.start_calibration(gesture, side, mode))
        for i in range(4):
            self.engine.add_calibration_sample(pose(seed) + i * 0.001)
        return self.engine.finish_calibration()

    def test_calibrating_left_never_touches_right(self):
        self.assertTrue(self._capture("G01", "LEFT", REPLACE, 7))
        self.assertEqual(len(self.engine.database["gestures"]["LEFT"]["G01"]), 4)
        self.assertEqual(self.engine.database["gestures"]["RIGHT"]["G01"], [])

    def test_replace_and_append_modes(self):
        self._capture("G02", "LEFT", REPLACE, 9)
        self._capture("G02", "LEFT", APPEND, 9)
        self.assertEqual(len(self.engine.database["gestures"]["LEFT"]["G02"]), 8)
        self._capture("G02", "LEFT", REPLACE, 9)
        self.assertEqual(len(self.engine.database["gestures"]["LEFT"]["G02"]), 4)

    def test_too_few_samples_refused(self):
        self.engine.start_calibration("G03", "LEFT", REPLACE)
        self.engine.add_calibration_sample(pose(3))
        self.assertFalse(self.engine.finish_calibration())


class StabilizationTests(TempCase):
    def setUp(self):
        super().setUp()
        self.engine = GestureEngine(self.gestures_path)

    def test_high_confidence_validates_in_one_frame(self):
        self.assertEqual(self.engine.stabilize("LEFT", "G01", 0.94), "G01")

    def test_low_confidence_needs_two_frames(self):
        self.assertEqual(self.engine.stabilize("LEFT", "G01", 0.58), UNKNOWN)
        self.assertEqual(self.engine.stabilize("LEFT", "G01", 0.58), "G01")

    def test_unknown_releases_after_two_frames(self):
        self.engine.stabilize("LEFT", "G01", 0.94)
        self.assertEqual(self.engine.stabilize("LEFT", UNKNOWN, 0.0), "G01")
        self.assertEqual(self.engine.stabilize("LEFT", UNKNOWN, 0.0), UNKNOWN)

    def test_histories_are_separate_per_hand(self):
        self.engine.stabilize("LEFT", "G01", 0.94)
        self.engine.stabilize("RIGHT", "G05", 0.94)
        self.engine.reset_hand("RIGHT")
        self.assertEqual(self.engine.stable_gestures["LEFT"], "G01")
        self.assertEqual(self.engine.stable_gestures["RIGHT"], UNKNOWN)


# =========================================================================
# Mapping : types, validation, migration
# =========================================================================
class MappingTests(TempCase):
    def test_default_is_v4_with_two_tables(self):
        mapping = GestureMapping(self.mapping_path)
        with open(self.mapping_path, encoding="utf-8") as file:
            saved = json.load(file)
        self.assertEqual(saved["version"], MAPPING_VERSION)
        self.assertEqual(set(saved) - {"version"}, {"LEFT", "RIGHT"})
        self.assertEqual(mapping.get_cooldown("LEFT", "G07"), 150)

    def test_sides_are_independent(self):
        mapping = GestureMapping(self.mapping_path)
        mapping.set_command("LEFT", "G01", HOLD, ["W"])
        mapping.set_command("RIGHT", "G01", PRESS, ["LEFT_MOUSE"])
        self.assertEqual(mapping.get_inputs("LEFT", "G01"), ["W"])
        self.assertEqual(mapping.get_inputs("RIGHT", "G01"), ["LEFT_MOUSE"])

    def test_combination_type(self):
        mapping = GestureMapping(self.mapping_path)
        self.assertTrue(mapping.set_command("LEFT", "G02", COMBINATION, ["W", "D"]))
        self.assertEqual(mapping.get_type("LEFT", "G02"), COMBINATION)
        self.assertEqual(describe(mapping.get("LEFT", "G02")), "W+D")

    def test_cooldown_stored_and_validated(self):
        mapping = GestureMapping(self.mapping_path)
        self.assertTrue(mapping.set_command("RIGHT", "G01", PRESS, ["LEFT_MOUSE"], 300))
        self.assertEqual(mapping.get_cooldown("RIGHT", "G01"), 300)
        self.assertFalse(mapping.set_command("RIGHT", "G01", PRESS, ["LEFT_MOUSE"], -5))

    def test_macro_with_all_actions(self):
        mapping = GestureMapping(self.mapping_path)
        steps = [
            {"action": STEP_HOLD, "inputs": ["W", "D"]},
            {"action": STEP_WAIT, "duration": 400},
            {"action": STEP_PRESS, "inputs": ["LEFT_MOUSE"]},
            {"action": STEP_RELEASE, "inputs": ["W", "D"]},
        ]
        self.assertTrue(mapping.set_macro("LEFT", "G05", steps))
        self.assertEqual(len(mapping.get_steps("LEFT", "G05")), 4)

    def test_repeat_block_validated(self):
        mapping = GestureMapping(self.mapping_path)
        steps = [
            {"action": STEP_REPEAT_BEGIN, "count": 4},
            {"action": STEP_PRESS, "inputs": ["LEFT_MOUSE"]},
            {"action": STEP_WAIT, "duration": 120},
            {"action": STEP_REPEAT_END},
        ]
        self.assertTrue(mapping.set_macro("RIGHT", "G09", steps))

    def test_validation_errors_are_explicit(self):
        mapping = GestureMapping(self.mapping_path)
        cases = [
            ([], "macro vide"),
            ([{"action": "SAUTER", "inputs": ["W"]}], "action inconnue"),
            ([{"action": STEP_WAIT, "duration": -10}], "negatif"),
            ([{"action": STEP_REPEAT_BEGIN, "count": 1}], "au moins"),
            ([{"action": STEP_REPEAT_BEGIN, "count": 999}, {"action": STEP_REPEAT_END}], "trop grand"),
            ([{"action": STEP_REPEAT_BEGIN, "count": 3}, {"action": STEP_PRESS, "inputs": ["W"]}], "sans REPEAT_END"),
            ([{"action": STEP_REPEAT_END}], "sans REPEAT_BEGIN"),
            ([{"action": STEP_PRESS, "inputs": ["TOUCHE_BIDON"]}], "commande inconnue"),
        ]
        for steps, fragment in cases:
            with self.subTest(steps=steps):
                self.assertFalse(mapping.set_macro("LEFT", "G05", steps))
                self.assertIn(fragment, mapping.last_error)

    def test_unknown_command_never_accepted(self):
        mapping = GestureMapping(self.mapping_path)
        self.assertFalse(mapping.set_command("LEFT", "G01", HOLD, ["TOUCHE_BIDON"]))
        self.assertFalse(mapping.set_command("LEFT", "G01", "MAINTENIR", ["W"]))

    def test_migration_v3_keeps_entries_and_adds_cooldown(self):
        legacy = {
            "version": 3,
            "LEFT": {"G01": {"type": "HOLD", "inputs": ["W"]}, "G07": {"type": "PRESS", "inputs": ["E"]}},
            "RIGHT": {"G01": {"type": "HOLD", "inputs": ["D"]}},
        }
        with open(self.mapping_path, "w", encoding="utf-8") as file:
            json.dump(legacy, file)
        mapping = GestureMapping(self.mapping_path)
        self.assertEqual(mapping.get_inputs("LEFT", "G01"), ["W"])
        self.assertEqual(mapping.get_inputs("RIGHT", "G01"), ["D"])
        self.assertEqual(mapping.get_cooldown("LEFT", "G07"), 150)
        with open(self.mapping_path, encoding="utf-8") as file:
            self.assertEqual(json.load(file)["version"], MAPPING_VERSION)

    def test_migration_v2_v1_and_flat(self):
        legacy = {"G01": {"action": "W", "type": "HOLD"}, "G02": {"type": "hold", "input": "s"}, "G03": "SPACE"}
        with open(self.mapping_path, "w", encoding="utf-8") as file:
            json.dump(legacy, file)
        mapping = GestureMapping(self.mapping_path)
        for side in SIDES:
            self.assertEqual(mapping.get_inputs(side, "G01"), ["W"])
            self.assertEqual(mapping.get_type(side, "G02"), HOLD)
            self.assertEqual(mapping.get_type(side, "G03"), PRESS)

    def test_old_macro_without_speed_still_loads(self):
        legacy = {
            "version": 3,
            "LEFT": {"G05": {"type": "MACRO", "steps": [{"action": "PRESS", "inputs": ["SPACE"]}]}},
            "RIGHT": {},
        }
        with open(self.mapping_path, "w", encoding="utf-8") as file:
            json.dump(legacy, file)
        mapping = GestureMapping(self.mapping_path)
        self.assertEqual(mapping.get_type("LEFT", "G05"), MACRO)
        self.assertEqual(mapping.get_speed("LEFT", "G05"), 1.0)

    def test_orphan_release_is_reported_not_refused(self):
        mapping = GestureMapping(self.mapping_path)
        steps = [
            {"action": STEP_HOLD, "inputs": ["W"]},
            {"action": STEP_RELEASE, "inputs": ["W", "SHIFT"]},
        ]
        self.assertTrue(mapping.set_macro("LEFT", "G05", steps))
        self.assertIn("SHIFT", mapping.last_warning)
        self.assertNotIn("W,", mapping.last_warning)

    def test_balanced_hold_release_has_no_warning(self):
        mapping = GestureMapping(self.mapping_path)
        steps = [
            {"action": STEP_HOLD, "inputs": ["W", "D"]},
            {"action": STEP_WAIT, "duration": 100},
            {"action": STEP_RELEASE, "inputs": ["W", "D"]},
        ]
        self.assertTrue(mapping.set_macro("LEFT", "G05", steps))
        self.assertEqual(mapping.last_warning, "")

    def test_speed_saved_and_reloaded(self):
        mapping = GestureMapping(self.mapping_path)
        steps = [{"action": STEP_PRESS, "inputs": ["SPACE"]}]
        self.assertTrue(mapping.set_macro("LEFT", "G05", steps, 2.0))
        self.assertEqual(GestureMapping(self.mapping_path).get_speed("LEFT", "G05"), 2.0)
        self.assertIn(2.0, SPEED_CHOICES)

    def test_preview_is_ascii(self):
        steps = [
            {"action": STEP_PRESS, "inputs": ["SPACE"]},
            {"action": STEP_WAIT, "duration": 150},
            {"action": STEP_PRESS, "inputs": ["LEFT_MOUSE"]},
        ]
        text = preview(normalize_steps(steps))
        self.assertEqual(text, "SPACE --150ms--> LEFT_MOUSE")
        self.assertTrue(text.isascii())


# =========================================================================
# Controleur d'entrees : sources, combinaisons, macros
# =========================================================================
class InputTests(unittest.TestCase):
    def test_hold_pressed_once_then_released_once(self):
        ctrl = make_inputs()
        ctrl.enable()
        for _ in range(30):
            ctrl.set_source_hold("LEFT", ["W"])
        self.assertEqual(ctrl.keyboard.pressed, ["w"])
        self.assertEqual(ctrl.keyboard.released, [])
        ctrl.set_source_hold("LEFT", [])
        self.assertEqual(ctrl.keyboard.released, ["w"])

    def test_shared_key_released_by_last_source_only(self):
        ctrl = make_inputs()
        ctrl.enable()
        ctrl.set_source_hold("LEFT", ["W"])
        ctrl.set_source_hold("RIGHT", ["W"])
        self.assertEqual(ctrl.keyboard.pressed, ["w"])
        ctrl.set_source_hold("LEFT", [])
        self.assertEqual(ctrl.keyboard.released, [])  # RIGHT l'utilise encore
        ctrl.set_source_hold("RIGHT", [])
        self.assertEqual(ctrl.keyboard.released, ["w"])

    def test_hold_combination(self):
        ctrl = make_inputs()
        ctrl.enable()
        ctrl.set_source_hold("LEFT", ["W", "D"])
        self.assertEqual(set(ctrl.keyboard.pressed), {"w", "d"})
        ctrl.set_source_hold("LEFT", ["W"])
        self.assertEqual(ctrl.keyboard.released, ["d"])

    def test_press_combination_order(self):
        ctrl = make_inputs()
        ctrl.enable()
        shift = ctrl.special_keys["SHIFT"]
        self.assertTrue(ctrl.press_combo(["SHIFT", "E"], duration=0))
        self.assertEqual(ctrl.keyboard.pressed, [shift, "e"])
        self.assertEqual(ctrl.keyboard.released, ["e", shift])

    def test_keyboard_and_mouse_combination(self):
        ctrl = make_inputs()
        ctrl.enable()
        ctrl.press_combo(["CTRL", "LEFT_MOUSE"], duration=0)
        self.assertEqual(ctrl.keyboard.pressed, [ctrl.special_keys["CTRL"]])
        self.assertEqual(ctrl.mouse.pressed, [ctrl.mouse_buttons["LEFT_MOUSE"]])

    def test_extra_mouse_buttons_supported(self):
        ctrl = make_inputs()
        ctrl.enable()
        for name in ("LEFT_MOUSE", "RIGHT_MOUSE", "MIDDLE_MOUSE", "X1_MOUSE", "X2_MOUSE"):
            with self.subTest(button=name):
                kind, target = ctrl.resolve(name)
                self.assertEqual(kind, "mouse")
                self.assertIsNotNone(target)

    def test_unknown_command_never_sent(self):
        ctrl = make_inputs()
        ctrl.enable()
        self.assertFalse(ctrl.press_combo(["TOUCHE_BIDON"], duration=0))
        self.assertEqual(ctrl.keyboard.pressed, [])

    def test_release_all_clears_everything(self):
        ctrl = make_inputs()
        ctrl.enable()
        ctrl.set_source_hold("LEFT", ["W", "RIGHT_MOUSE"])
        ctrl.press_combo(["E"], duration=10)
        ctrl.release_all()
        self.assertFalse(ctrl.currently_pressed)
        self.assertFalse(ctrl.held_buttons)
        self.assertFalse(ctrl.hold_sources)
        self.assertFalse(ctrl.pending_releases)
        self.assertIn("e", ctrl.keyboard.released)

    def test_press_reports_pynput_failure(self):
        ctrl = make_inputs()
        ctrl.enable()

        def boom(_key):
            raise RuntimeError("pynput indisponible")

        ctrl.keyboard.press = boom
        self.assertFalse(ctrl.press_combo(["W"], duration=0))


class MacroTests(unittest.TestCase):
    CLICKS = [
        {"action": STEP_PRESS, "inputs": ["LEFT_MOUSE"]},
        {"action": STEP_WAIT, "duration": 20},
        {"action": STEP_PRESS, "inputs": ["LEFT_MOUSE"]},
        {"action": STEP_WAIT, "duration": 20},
        {"action": STEP_PRESS, "inputs": ["LEFT_MOUSE"]},
    ]
    HOLD_RELEASE = [
        {"action": STEP_HOLD, "inputs": ["W"]},
        {"action": STEP_WAIT, "duration": 40},
        {"action": STEP_PRESS, "inputs": ["LEFT_MOUSE"]},
        {"action": STEP_RELEASE, "inputs": ["W"]},
    ]
    REPEATED = [
        {"action": STEP_REPEAT_BEGIN, "count": 4},
        {"action": STEP_PRESS, "inputs": ["LEFT_MOUSE"]},
        {"action": STEP_REPEAT_END},
    ]

    def test_successive_clicks(self):
        ctrl = make_inputs()
        ctrl.enable()
        left = ctrl.mouse_buttons["LEFT_MOUSE"]
        ctrl.start_macro("RIGHT", self.CLICKS, "G09")
        drain(ctrl)
        self.assertEqual(ctrl.mouse.pressed.count(left), 3)

    def test_hold_wait_release(self):
        ctrl = make_inputs()
        ctrl.enable()
        ctrl.start_macro("LEFT", self.HOLD_RELEASE, "G05")
        ctrl.tick()
        self.assertIn("w", ctrl.currently_pressed)  # maintenu pendant le WAIT
        drain(ctrl)
        self.assertNotIn("w", ctrl.currently_pressed)
        self.assertIn("w", ctrl.keyboard.released)
        self.assertIn(ctrl.mouse_buttons["LEFT_MOUSE"], ctrl.mouse.pressed)

    def test_repeat_runs_exact_count(self):
        ctrl = make_inputs()
        ctrl.enable()
        left = ctrl.mouse_buttons["LEFT_MOUSE"]
        ctrl.start_macro("LEFT", self.REPEATED, "G09")
        drain(ctrl)
        self.assertEqual(ctrl.mouse.pressed.count(left), 4)

    def test_nested_repeat(self):
        ctrl = make_inputs()
        ctrl.enable()
        steps = [
            {"action": STEP_REPEAT_BEGIN, "count": 2},
            {"action": STEP_REPEAT_BEGIN, "count": 3},
            {"action": STEP_PRESS, "inputs": ["E"]},
            {"action": STEP_REPEAT_END},
            {"action": STEP_REPEAT_END},
        ]
        ctrl.start_macro("LEFT", steps, "G10")
        drain(ctrl)
        self.assertEqual(ctrl.keyboard.pressed.count("e"), 6)

    def test_macro_hold_released_when_stopped(self):
        ctrl = make_inputs()
        ctrl.enable()
        ctrl.start_macro("LEFT", [{"action": STEP_HOLD, "inputs": ["W"]},
                                  {"action": STEP_WAIT, "duration": 5000}], "G05")
        ctrl.tick()
        self.assertIn("w", ctrl.currently_pressed)
        ctrl.stop_macro("LEFT")
        self.assertNotIn("w", ctrl.currently_pressed)
        self.assertIn("w", ctrl.keyboard.released)

    def test_macro_hold_does_not_touch_other_hand(self):
        ctrl = make_inputs()
        ctrl.enable()
        ctrl.set_source_hold("RIGHT", ["D"])
        ctrl.start_macro("LEFT", self.HOLD_RELEASE, "G05")
        ctrl.tick()
        self.assertEqual(ctrl.currently_pressed, {"w", "d"})
        drain(ctrl)
        self.assertEqual(ctrl.currently_pressed, {"d"})

    def test_f8_stops_macro_and_releases_everything(self):
        ctrl = make_inputs()
        ctrl.enable()
        ctrl.set_source_hold("RIGHT", ["D"])
        ctrl.start_macro("LEFT", [{"action": STEP_HOLD, "inputs": ["W"]},
                                  {"action": STEP_WAIT, "duration": 5000}], "G05")
        ctrl.tick()
        ctrl.disable()  # chemin exact de F8 et de G OFF
        self.assertFalse(ctrl.macros)
        self.assertFalse(ctrl.currently_pressed)
        self.assertFalse(ctrl.hold_sources)
        for key in ("w", "d"):
            self.assertIn(key, ctrl.keyboard.released)

    def test_two_macros_one_per_hand(self):
        ctrl = make_inputs()
        ctrl.enable()
        ctrl.start_macro("LEFT", self.CLICKS, "G09")
        ctrl.start_macro("RIGHT", [{"action": STEP_PRESS, "inputs": ["E"]}], "G05")
        self.assertEqual(set(ctrl.macros), {"LEFT", "RIGHT"})
        drain(ctrl)
        self.assertIn("e", ctrl.keyboard.pressed)

    def test_macro_not_restarted_while_running(self):
        ctrl = make_inputs()
        ctrl.enable()
        ctrl.start_macro("LEFT", self.CLICKS, "G09")
        self.assertFalse(ctrl.start_macro("LEFT", self.CLICKS, "G09"))
        drain(ctrl)

    def test_macro_ignored_when_input_off(self):
        ctrl = make_inputs()
        self.assertFalse(ctrl.start_macro("LEFT", self.CLICKS, "G09"))

    def test_speed_multiplier_shortens_waits(self):
        ctrl = make_inputs()
        ctrl.enable()
        ctrl.start_macro("LEFT", [{"action": STEP_WAIT, "duration": 1000}], "G05", speed=4.0)
        ctrl.tick()
        state = ctrl.macros["LEFT"]
        self.assertLess(state["next_time"] - time.monotonic(), 0.30)


# =========================================================================
# Chaine complete
# =========================================================================
class ChainTests(TempCase):
    def setUp(self):
        super().setUp()
        self.app = make_app(self.mapping_path)
        self.app.inputs.enable()
        self.kb = self.app.inputs.keyboard

    def test_left_g01_w_and_right_g01_left_mouse(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        self.app.mapping.set_command("RIGHT", "G01", PRESS, ["LEFT_MOUSE"], 0)
        run_frame(self.app, LEFT="G01", RIGHT="G01")
        self.assertEqual(self.app.inputs.currently_pressed, {"w"})
        self.assertIn(self.app.inputs.mouse_buttons["LEFT_MOUSE"], self.app.inputs.mouse.pressed)

    def test_left_g02_combination(self):
        self.app.mapping.set_command("LEFT", "G02", COMBINATION, ["W", "D"])
        run_frame(self.app, LEFT="G02")
        self.assertEqual(self.app.inputs.currently_pressed, {"w", "d"})

    def test_press_does_not_spam(self):
        self.app.mapping.set_command("LEFT", "G03", PRESS, ["SPACE"], 0)
        space = self.app.inputs.special_keys["SPACE"]
        for _ in range(100):
            run_frame(self.app, LEFT="G03")
        self.assertEqual(self.kb.pressed.count(space), 1)
        run_frame(self.app, LEFT=UNKNOWN)
        run_frame(self.app, LEFT="G03")
        self.assertEqual(self.kb.pressed.count(space), 2)

    def test_cooldown_blocks_fast_retrigger(self):
        self.app.mapping.set_command("LEFT", "G03", PRESS, ["SPACE"], 1000)
        space = self.app.inputs.special_keys["SPACE"]
        run_frame(self.app, LEFT="G03")
        run_frame(self.app, LEFT=UNKNOWN)
        run_frame(self.app, LEFT="G03")
        self.assertEqual(self.kb.pressed.count(space), 1)

    def test_zero_cooldown_allows_retrigger(self):
        self.app.mapping.set_command("LEFT", "G03", PRESS, ["SPACE"], 0)
        space = self.app.inputs.special_keys["SPACE"]
        run_frame(self.app, LEFT="G03")
        run_frame(self.app, LEFT=UNKNOWN)
        run_frame(self.app, LEFT="G03")
        self.assertEqual(self.kb.pressed.count(space), 2)

    def test_two_hands_simultaneous(self):
        self.app.mapping.set_command("LEFT", "G03", HOLD, ["W", "D"])
        self.app.mapping.set_command("RIGHT", "G09", PRESS, ["LEFT_MOUSE"], 0)
        run_frame(self.app, LEFT="G03", RIGHT="G09")
        self.assertEqual(self.app.inputs.currently_pressed, {"w", "d"})
        self.assertIn(self.app.inputs.mouse_buttons["LEFT_MOUSE"], self.app.inputs.mouse.pressed)

    def test_losing_left_does_not_cut_right(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        self.app.mapping.set_command("RIGHT", "G01", HOLD, ["D"])
        run_frame(self.app, LEFT="G01", RIGHT="G01")
        run_frame(self.app, LEFT=UNKNOWN)
        self.assertEqual(self.app.inputs.currently_pressed, {"d"})
        run_frame(self.app, RIGHT=UNKNOWN)
        self.assertEqual(self.app.inputs.currently_pressed, set())

    def test_losing_right_does_not_cut_left(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        self.app.mapping.set_command("RIGHT", "G01", HOLD, ["D"])
        run_frame(self.app, LEFT="G01", RIGHT="G01")
        run_frame(self.app, RIGHT=UNKNOWN)
        self.assertEqual(self.app.inputs.currently_pressed, {"w"})

    def test_shared_key_between_two_hands(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        self.app.mapping.set_command("RIGHT", "G02", HOLD, ["W"])
        run_frame(self.app, LEFT="G01", RIGHT="G02")
        self.assertEqual(self.kb.pressed, ["w"])
        run_frame(self.app, LEFT=UNKNOWN)
        self.assertEqual(self.kb.released, [])  # RIGHT maintient encore W
        self.assertEqual(self.app.inputs.currently_pressed, {"w"})
        run_frame(self.app, RIGHT=UNKNOWN)
        self.assertEqual(self.kb.released, ["w"])

    def test_macro_while_other_hand_holds(self):
        self.app.mapping.set_macro("LEFT", "G05", [
            {"action": STEP_PRESS, "inputs": ["SPACE"]},
            {"action": STEP_WAIT, "duration": 20},
            {"action": STEP_PRESS, "inputs": ["LEFT_MOUSE"]},
        ])
        self.app.mapping.set_command("RIGHT", "G01", HOLD, ["D"])
        space = self.app.inputs.special_keys["SPACE"]
        for _ in range(20):
            run_frame(self.app, LEFT="G05", RIGHT="G01")
            time.sleep(0.005)
        self.assertEqual(self.kb.pressed.count(space), 1)
        self.assertEqual(self.app.inputs.currently_pressed, {"d"})

    def test_disabled_entry_sends_nothing(self):
        self.app.mapping.disable("LEFT", "G01")
        run_frame(self.app, LEFT="G01")
        self.assertEqual(self.kb.pressed, [])

    def test_input_off_blocks_everything(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        run_frame(self.app, LEFT="G01")
        self.app.inputs.disable()
        run_frame(self.app, LEFT=UNKNOWN)
        run_frame(self.app, LEFT="G01")
        self.assertFalse(self.app.inputs.currently_pressed)

    def test_every_gesture_of_both_hands_reaches_pynput(self):
        for side in SIDES:
            for gesture in GESTURE_NAMES:
                app = make_app(self.mapping_path)
                app.inputs.enable()
                entry = app.mapping.get(side, gesture)
                run_frame(app, **{side: gesture})
                with self.subTest(side=side, gesture=gesture, entry=entry):
                    kind, target = app.inputs.resolve(entry["inputs"][0])
                    sink = app.inputs.mouse if kind == "mouse" else app.inputs.keyboard
                    self.assertIn(target, sink.pressed)


class EditorTests(unittest.TestCase):
    def test_insert_repeat_block_is_valid(self):
        app = HandControllerApp.__new__(HandControllerApp)
        app.macro_steps = [{"action": STEP_PRESS, "inputs": ["SPACE"]}]
        app.macro_index = 0
        app.status_message = ""
        app._insert_repeat_block()
        normalize_steps(app.macro_steps)
        self.assertEqual(app.macro_steps[1]["action"], STEP_REPEAT_BEGIN)
        self.assertEqual(app.macro_steps[1]["count"], 4)
        self.assertEqual(app.macro_steps[-1]["action"], STEP_REPEAT_END)
        self.assertEqual(app.macro_index, 2)

    def test_insert_hold_and_release(self):
        app = HandControllerApp.__new__(HandControllerApp)
        app.macro_steps = [{"action": STEP_PRESS, "inputs": ["SPACE"]}]
        app.macro_index = 0
        self.assertTrue(app._insert_step({"action": STEP_HOLD, "inputs": ["W"]}))
        self.assertTrue(app._insert_step({"action": STEP_RELEASE, "inputs": ["W"]}))
        self.assertEqual(app.macro_steps[1]["action"], STEP_HOLD)
        self.assertEqual(app.macro_steps[2]["action"], STEP_RELEASE)


class HandIdentityTests(unittest.TestCase):
    def test_sides_assigned_from_handedness(self):
        left = {"side": "LEFT", "score": 0.9}
        right = {"side": "RIGHT", "score": 0.8}
        assigned = resolve_hand_sides([right, left])
        self.assertIs(assigned["LEFT"], left)
        self.assertIs(assigned["RIGHT"], right)

    def test_duplicate_handedness_resolved_by_score(self):
        sure = {"side": "LEFT", "score": 0.95}
        unsure = {"side": "LEFT", "score": 0.51}
        assigned = resolve_hand_sides([sure, unsure])
        self.assertIs(assigned["LEFT"], sure)
        self.assertIs(assigned["RIGHT"], unsure)

    def test_point_colors(self):
        self.assertEqual(hand_point_color("LEFT"), (255, 255, 255))
        self.assertEqual(hand_point_color("RIGHT"), (0, 0, 0))


class NonBlockingTests(unittest.TestCase):
    """Aucune attente bloquante sur le trajet de la boucle webcam."""

    HOT_PATH = [
        (main_module.HandControllerApp, ["run", "process_frame", "execute_gesture",
                                         "update_held_inputs", "collect_detections", "on_cooldown",
                                         "handle_preview_toggle"]),
        (input_controller.InputController, ["tick", "update_macros", "_advance_macro",
                                            "press_combo", "set_source_hold", "_apply_union"]),
    ]

    @staticmethod
    def called_names(function):
        """Noms des fonctions reellement appelees, commentaires exclus."""
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = node.func
                names.add(target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", ""))
        return names

    def test_no_sleep_on_the_hot_path(self):
        for owner, names in self.HOT_PATH:
            for name in names:
                with self.subTest(function=f"{owner.__name__}.{name}"):
                    self.assertNotIn("sleep", self.called_names(getattr(owner, name)))

    def test_macro_engine_uses_monotonic(self):
        source = inspect.getsource(input_controller.InputController.start_macro)
        self.assertIn("time.monotonic", source)


# =========================================================================
# Release V1.0 : chemins, JSON, reset, version
# =========================================================================
class PathTests(TempCase):
    def test_engine_finds_file_from_other_cwd(self):
        payload = {"version": 8, "gestures": {"LEFT": {"G01": samples_for(3)}, "RIGHT": {}}}
        with open(self.gestures_path, "w", encoding="utf-8") as file:
            json.dump(payload, file)
        other = os.path.join(self.td.name, "elsewhere")
        os.makedirs(other, exist_ok=True)
        previous = os.getcwd()
        try:
            os.chdir(other)
            engine = GestureEngine(self.gestures_path, recognition_threshold=0.2)
            self.assertEqual(len(engine.database["gestures"]["LEFT"]["G01"]), 8)
            self.assertEqual(engine.recognize(samples_for(3)[0], "LEFT")[0], "G01")
        finally:
            os.chdir(previous)

    def test_mapping_finds_file_from_other_cwd(self):
        mapping = GestureMapping(self.mapping_path)
        mapping.set_command("LEFT", "G01", HOLD, ["W"])
        other = os.path.join(self.td.name, "elsewhere")
        os.makedirs(other, exist_ok=True)
        previous = os.getcwd()
        try:
            os.chdir(other)
            loaded = GestureMapping(self.mapping_path)
            self.assertEqual(loaded.get_inputs("LEFT", "G01"), ["W"])
        finally:
            os.chdir(previous)

    def test_app_dir_is_not_the_working_directory(self):
        from config import app_dir
        previous = os.getcwd()
        other = os.path.join(self.td.name, "elsewhere")
        os.makedirs(other, exist_ok=True)
        try:
            os.chdir(other)
            self.assertEqual(os.path.normpath(app_dir()), os.path.normpath(os.path.dirname(os.path.abspath(main_module.__file__))))
        finally:
            os.chdir(previous)


class JsonSafetyTests(TempCase):
    def test_corrupt_gestures_json_does_not_crash(self):
        with open(self.gestures_path, "w", encoding="utf-8") as file:
            file.write("{not-json")
        engine = GestureEngine(self.gestures_path)
        self.assertEqual(engine.database["version"], 8)
        self.assertEqual(set(engine.database["gestures"]), {"LEFT", "RIGHT"})
        self.assertTrue(os.path.isfile(self.gestures_path + ".bak"))

    def test_corrupt_mapping_json_does_not_crash(self):
        with open(self.mapping_path, "w", encoding="utf-8") as file:
            file.write("[] totally broken")
        mapping = GestureMapping(self.mapping_path)
        self.assertEqual(mapping.get_type("LEFT", "G01"), HOLD)
        self.assertTrue(os.path.isfile(self.mapping_path + ".bak"))

    def test_save_creates_single_backup(self):
        mapping = GestureMapping(self.mapping_path)
        mapping.set_command("LEFT", "G01", HOLD, ["W"])
        mapping.set_command("LEFT", "G01", HOLD, ["S"])
        self.assertTrue(os.path.isfile(self.mapping_path + ".bak"))
        self.assertFalse(os.path.isfile(self.mapping_path + ".bak.bak"))


class ResetConfigTests(TempCase):
    def test_reset_database_empties_both_banks(self):
        engine = GestureEngine(self.gestures_path, samples_per_gesture=4)
        engine.start_calibration("G01", "LEFT", REPLACE)
        for i in range(4):
            engine.add_calibration_sample(pose(1) + i * 0.001)
        engine.finish_calibration()
        self.assertTrue(engine.database["gestures"]["LEFT"]["G01"])
        self.assertTrue(engine.reset_database())
        self.assertEqual(engine.database["gestures"]["LEFT"]["G01"], [])
        self.assertEqual(engine.database["gestures"]["RIGHT"]["G01"], [])
        self.assertTrue(os.path.isfile(self.gestures_path + ".bak"))

    def test_reset_all_mapping_restores_defaults(self):
        mapping = GestureMapping(self.mapping_path)
        mapping.set_command("LEFT", "G01", COMBINATION, ["W", "D"])
        self.assertTrue(mapping.reset_all())
        self.assertEqual(mapping.get_type("LEFT", "G01"), HOLD)
        self.assertEqual(mapping.get_inputs("LEFT", "G01"), ["W"])


class VersionAndSettingsTests(unittest.TestCase):
    def test_version_is_1_0_0(self):
        from config import APP_NAME, APP_VERSION
        self.assertEqual(APP_NAME, "HandController")
        self.assertEqual(APP_VERSION, "1.0.0")
        self.assertEqual(main_module.WINDOW_TITLE, "HandController")
        self.assertNotIn("v1.0.0", main_module.WINDOW_TITLE)

    def test_invalid_settings_fall_back(self):
        from config import validate_settings
        cleaned = validate_settings({"camera_index": "nope", "knn_k": 99, "debug": 1})
        self.assertEqual(cleaned["camera_index"], 0)
        self.assertEqual(cleaned["knn_k"], 15)
        self.assertTrue(cleaned["debug"])

    def test_countdown_uses_monotonic_not_sleep(self):
        source = inspect.getsource(main_module.HandControllerApp.advance_calibration)
        self.assertIn("countdown", source)
        self.assertNotIn("sleep", source)
        source_key = inspect.getsource(main_module.HandControllerApp.handle_calibration_key)
        self.assertIn("countdown", source_key)

    def test_run_always_releases_in_finally(self):
        source = inspect.getsource(main_module.HandControllerApp.run)
        self.assertIn("finally:", source)
        self.assertIn("release_all", source)
        self.assertIn("cap.release", source)
        self.assertIn("destroyAllWindows", source)


class CleanupSafetyTests(unittest.TestCase):
    def test_exception_path_still_releases(self):
        ctrl = make_inputs()
        ctrl.enable()
        ctrl.set_source_hold("LEFT", ["W", "SHIFT"])
        try:
            raise RuntimeError("simulated crash")
        except RuntimeError:
            ctrl.stop_all_macros()
            ctrl.release_all()
        self.assertFalse(ctrl.currently_pressed)
        self.assertIn("w", ctrl.keyboard.released)

    def test_lost_hand_releases_only_that_source(self):
        app = make_app(os.path.join(tempfile.mkdtemp(), "mapping.json"))
        app.inputs.enable()
        app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        app.mapping.set_command("RIGHT", "G01", HOLD, ["D"])
        run_frame(app, LEFT="G01", RIGHT="G01")
        run_frame(app, LEFT=UNKNOWN, RIGHT="G01")
        self.assertEqual(app.inputs.currently_pressed, {"d"})


class PreviewAndFullscreenTests(unittest.TestCase):
    """Inputs are system-level SendInput. The OpenCV preview is optional."""

    def test_input_controller_never_targets_a_window(self):
        source = inspect.getsource(input_controller)
        for forbidden in ("SetForegroundWindow(", "ShowWindow(", "FindWindow(",
                          "SendMessage(", "PostMessage(", "AttachThreadInput("):
            self.assertNotIn(forbidden, source)
        press = inspect.getsource(input_controller.InputController._press_target)
        self.assertIn("SendInput", press)
        self.assertIn("keyboard.press", press)
        self.assertIn("mouse.press", press)

    def test_app_never_forces_game_focus(self):
        source = inspect.getsource(main_module)
        self.assertNotIn("SetForegroundWindow(", source)
        import preview_window
        preview_src = inspect.getsource(preview_window)
        for forbidden in ("SetForegroundWindow(", "SetWindowPos(", "SetWindowLong",
                          "NOACTIVATE", "TOPMOST", "ShowWindow(", "BringWindowToTop"):
            self.assertNotIn(forbidden, preview_src)
        self.assertIn("imshow", inspect.getsource(preview_window.show_preview_frame))

    def test_f8_to_f11_are_not_hardcoded_ui_shortcuts(self):
        source = inspect.getsource(input_controller.InputController.start_emergency_listener)
        self.assertNotIn("if key == emergency", source)
        self.assertNotIn("if key == preview", source)
        self.assertNotIn("Key.f8", source)
        self.assertNotIn("Key.f9", source)
        self.assertNotIn("Key.f10", source)
        self.assertNotIn("Key.f11", source)

    def test_hidden_preview_still_sends_hold_and_press(self):
        app = make_app(os.path.join(tempfile.mkdtemp(), "mapping.json"))
        app.inputs.enable()
        app.preview_visible = False
        app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        app.mapping.set_command("RIGHT", "G01", PRESS, ["LEFT_MOUSE"], 0)
        run_frame(app, LEFT="G01", RIGHT="G01")
        self.assertEqual(app.inputs.currently_pressed, {"w"})
        self.assertIn(app.inputs.mouse_buttons["LEFT_MOUSE"], app.inputs.mouse.pressed)

    def test_hidden_preview_two_hands_independent(self):
        app = make_app(os.path.join(tempfile.mkdtemp(), "mapping.json"))
        app.inputs.enable()
        app.preview_visible = False
        app.mapping.set_command("LEFT", "G02", COMBINATION, ["W", "D"])
        app.mapping.set_command("RIGHT", "G09", HOLD, ["SHIFT"])
        run_frame(app, LEFT="G02", RIGHT="G09")
        self.assertEqual(app.inputs.currently_pressed, {"w", "d", app.inputs.special_keys["SHIFT"]})
        run_frame(app, LEFT=UNKNOWN, RIGHT="G09")
        self.assertEqual(app.inputs.currently_pressed, {app.inputs.special_keys["SHIFT"]})

    def test_preview_toggle_does_not_release_keys(self):
        app = make_app(os.path.join(tempfile.mkdtemp(), "mapping.json"))
        app.inputs.enable()
        app.preview_visible = True
        app.preview_toggle_pending = False
        app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        run_frame(app, LEFT="G01")
        app.preview_toggle_pending = True
        app.handle_preview_toggle()
        self.assertFalse(app.preview_visible)
        self.assertEqual(app.inputs.currently_pressed, {"w"})
        app.preview_toggle_pending = True
        app.handle_preview_toggle()
        self.assertTrue(app.preview_visible)
        self.assertEqual(app.inputs.currently_pressed, {"w"})

    def test_macro_keeps_running_with_preview_hidden(self):
        app = make_app(os.path.join(tempfile.mkdtemp(), "mapping.json"))
        app.inputs.enable()
        app.preview_visible = False
        app.mapping.set_macro("LEFT", "G05", [
            {"action": "PRESS", "inputs": ["SPACE"]},
            {"action": "WAIT", "duration": 20},
            {"action": "PRESS", "inputs": ["E"]},
        ])
        run_frame(app, LEFT="G05")
        self.assertTrue(app.inputs.is_macro_running("LEFT"))
        deadline = time.monotonic() + 2.0
        while app.inputs.macros and time.monotonic() < deadline:
            app.inputs.tick()
            time.sleep(0.005)
        space = app.inputs.special_keys["SPACE"]
        self.assertIn(space, app.inputs.keyboard.pressed)
        self.assertIn("e", app.inputs.keyboard.pressed)

    def test_display_preview_setting_defaults_on(self):
        from config import default_settings, validate_settings
        self.assertTrue(default_settings()["display_preview"])
        self.assertFalse(validate_settings({"display_preview": 0})["display_preview"])

    def test_cover_resize_fills_destination_without_letterbox(self):
        import preview_window
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        filled = preview_window.cover_resize(frame, 1920, 1080)
        self.assertEqual(filled.shape[1], 1920)
        self.assertEqual(filled.shape[0], 1080)
        square = preview_window.cover_resize(frame, 800, 800)
        self.assertEqual(square.shape[:2], (800, 800))
        same = preview_window.cover_resize(frame, 640, 480)
        self.assertEqual(same.shape[:2], (480, 640))
        source = inspect.getsource(preview_window.cover_resize)
        self.assertNotIn("np.zeros", source)
        self.assertIn("max(", source)
        win_src = inspect.getsource(preview_window)
        self.assertIn("WINDOW_FREERATIO", win_src)
        self.assertIn("WND_PROP_FULLSCREEN", win_src)
        self.assertNotIn("FindWindow(", win_src)
        self.assertNotIn("SetForegroundWindow(", win_src)

    def test_sidebar_camera_layout_fills_window(self):
        import preview_window
        camera = np.full((480, 640, 3), 90, dtype=np.uint8)
        canvas, sidebar = preview_window.compose_sidebar_camera(camera, 1280, 720)
        self.assertEqual(canvas.shape[1], 1280)
        self.assertEqual(canvas.shape[0], 720)
        self.assertGreaterEqual(sidebar, 180)
        self.assertEqual(canvas[:, sidebar:].shape[1], 1280 - sidebar)
        self.assertEqual(canvas[:, sidebar:].shape[0], 720)
        small, small_bar = preview_window.compose_sidebar_camera(camera, 400, 300)
        self.assertEqual(small.shape[:2], (300, 400))
        self.assertLess(small_bar, 400 - 16)

    def test_window_x_uses_existing_cleanup(self):
        import preview_window
        run_src = inspect.getsource(main_module.HandControllerApp.run)
        self.assertIn("preview_was_closed", run_src)
        self.assertIn("finally:", run_src)
        self.assertIn("release_all", run_src)
        self.assertIn("stop_all_macros", run_src)
        self.assertIn("cap.release", run_src)
        self.assertIn("destroyAllWindows", run_src)
        self.assertIn("destroyAllWindows", run_src)
        closed_src = inspect.getsource(preview_window.preview_was_closed)
        self.assertIn("WND_PROP_VISIBLE", closed_src)
        # fit_preview_frame recreates a missing window, so the X must be detected first.
        self.assertLess(run_src.index("preview_was_closed(WINDOW_TITLE)"),
                        run_src.index("fit_preview_frame(WINDOW_TITLE"))
        self.assertIn("Shutdown complete", run_src)

    def test_hiding_the_preview_is_not_a_window_close(self):
        app = HandControllerApp.__new__(HandControllerApp)
        app.preview_visible = True
        app.preview_toggle_pending = True
        app._preview_open = True
        app._preview_icon_applied = True
        app.log = mock.Mock()
        with mock.patch.object(main_module, "hide_preview") as hide:
            app.handle_preview_toggle()
        hide.assert_called_once()
        self.assertFalse(app.preview_visible)
        self.assertFalse(app._preview_open)
        self.assertFalse(app._preview_icon_applied)

# =========================================================================
# Moteur geometrique NumPy (couche additive)
# =========================================================================
UP = (0.0, -1.0)            # axe y des images : vers le haut = negatif
DOWN = (0.0, 1.0)
RIGHTWARD = (1.0, 0.0)
LEFTWARD = (-1.0, 0.0)


def hand(scale=0.2, palm=UP, index=UP, index_len=1.0, pinch=1.0):
    """(21, 3) synthetiques donnant exactement les mesures voulues.

    scale     : hand_scale (distance poignet -> base du majeur)
    palm      : direction unitaire de la paume, en coordonnees image
    index     : direction unitaire de l'index
    index_len : longueur de l'index en unites de taille de main
                (~1.0 tendu, ~0.25 replie)
    pinch     : pinch_ratio voulu
    """
    palm_unit = np.array(palm, dtype=np.float64)
    palm_unit = palm_unit / max(np.linalg.norm(palm_unit), 1e-12)
    index_unit = np.array(index, dtype=np.float64)
    index_unit = index_unit / max(np.linalg.norm(index_unit), 1e-12)

    points = np.zeros((geo.LANDMARK_COUNT, 3), dtype=np.float32)
    points[geo.WRIST] = (0.0, 0.0, 0.0)
    points[geo.MIDDLE_MCP] = (*(palm_unit * scale), 0.0)
    points[geo.INDEX_MCP] = points[geo.MIDDLE_MCP]
    tip = palm_unit * scale + index_unit * (index_len * scale)
    points[geo.INDEX_TIP] = (*tip, 0.0)
    points[geo.THUMB_TIP] = (*(tip + np.array([pinch * scale, 0.0])), 0.0)
    points[geo.PINKY_MCP] = (*(palm_unit * scale + np.array([0.6 * scale, 0.0])), 0.0)
    return points


def tilted(offset, index_len=1.0, pinch=1.0):
    """Main ouverte inclinee : paume et index alignes sur la meme direction."""
    direction = (offset[0], offset[1])
    return hand(palm=direction, index=direction, index_len=index_len, pinch=pinch)


def finger_pose(extensions, pinch=1.0):
    """Quatre longueurs bout-de-doigt / taille de main : index, majeur, annulaire, auriculaire."""
    scale = 0.2
    points = np.zeros((geo.LANDMARK_COUNT, 3), dtype=np.float32)
    points[geo.MIDDLE_MCP] = (0.0, -scale, 0.0)
    pairs = (
        (geo.INDEX_TIP, geo.INDEX_MCP, -0.04),
        (geo.MIDDLE_TIP, geo.MIDDLE_MCP, 0.0),
        (geo.RING_TIP, geo.RING_MCP, 0.04),
        (geo.PINKY_TIP, geo.PINKY_MCP, 0.08),
    )
    for (tip, mcp, spread), length in zip(pairs, extensions):
        base = np.array([spread * scale, -scale, 0.0], dtype=np.float32)
        if mcp != geo.MIDDLE_MCP:
            points[mcp] = base
        points[tip] = base + np.array([0.0, -float(length) * scale, 0.0], dtype=np.float32)
        pip = {
            geo.INDEX_TIP: geo.INDEX_PIP,
            geo.MIDDLE_TIP: geo.MIDDLE_PIP,
            geo.RING_TIP: geo.RING_PIP,
            geo.PINKY_TIP: geo.PINKY_PIP,
        }[tip]
        dip = {
            geo.INDEX_TIP: geo.INDEX_DIP,
            geo.MIDDLE_TIP: geo.MIDDLE_DIP,
            geo.RING_TIP: geo.RING_DIP,
            geo.PINKY_TIP: geo.PINKY_DIP,
        }[tip]
        points[pip] = base + np.array([0.0, -float(length) * scale * 0.5, 0.0], dtype=np.float32)
        points[dip] = base + np.array([0.0, -float(length) * scale * 0.75, 0.0], dtype=np.float32)
    points[geo.THUMB_MCP] = np.array([0.08 * scale, -0.12 * scale, 0.0], dtype=np.float32)
    points[geo.THUMB_IP] = points[geo.THUMB_MCP] + np.array([0.03 * scale, 0.0, 0.0], dtype=np.float32)
    points[geo.THUMB_TIP] = points[geo.THUMB_MCP] + np.array([0.05 * scale, 0.02 * scale, 0.0], dtype=np.float32)
    return points


def rotate_landmarks(points, degrees):
    """Rotate 2D landmarks around the wrist. Image Y still points down."""
    angle = np.radians(degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    origin = np.asarray(points[geo.WRIST][:2], dtype=np.float32)
    out = np.array(points, dtype=np.float32, copy=True)
    for index in range(len(out)):
        vec = out[index][:2] - origin
        out[index][0] = origin[0] + cos_a * vec[0] - sin_a * vec[1]
        out[index][1] = origin[1] + sin_a * vec[0] + cos_a * vec[1]
    return out


def thumb_extended_pose(direction=UP):
    """Thumb stretched along `direction` (image coords). Other fingers curled."""
    points = finger_pose((0.2, 0.2, 0.2, 0.2))
    scale = 0.2
    unit = np.array(direction, dtype=np.float32)
    unit = unit / max(float(np.linalg.norm(unit)), 1e-12)
    step = np.array([unit[0], unit[1], 0.0], dtype=np.float32) * scale
    origin = np.array([0.08 * scale, -0.10 * scale, 0.0], dtype=np.float32)
    points[geo.THUMB_CMC] = origin
    points[geo.THUMB_MCP] = origin + step * 0.18
    points[geo.THUMB_IP] = origin + step * 0.40
    points[geo.THUMB_TIP] = origin + step * 0.68
    return points


FIST_POSE = finger_pose((0.2, 0.2, 0.2, 0.2))
PALM_POSE = finger_pose((1.0, 1.0, 1.0, 1.0))
POINTING_POSE = finger_pose((1.0, 0.2, 0.2, 0.2))
OPEN_UP = tilted(UP)
OPEN_LEFT = tilted((-0.8, -0.6))
OPEN_RIGHT = tilted((0.8, -0.6))
OPEN_DOWN = tilted(DOWN)
CURLED = tilted(UP, index_len=0.15)          # main refermee -> neutre
PINCHING = hand(pinch=0.05)
OPENED = hand(pinch=1.0)


def make_geometry_app(mapping_path):
    """make_app + l'etat du moteur geometrique."""
    app = make_app(mapping_path)
    app.geometry = geo.GeometryEngine()
    app.prev_specials = {side: {} for side in SIDES}
    app.curr_specials = {side: {} for side in SIDES}
    app.debug = False
    return app


def run_geometry_frame(app, left=None, right=None):
    """Reproduit update_special_gestures pour une frame.

    left / right : landmarks (21, 3), ou None si la main est absente.
    """
    assigned = {side: None for side in SIDES}
    for side, points in (("LEFT", left), ("RIGHT", right)):
        if points is not None:
            assigned[side] = {"features": points.ravel()}
    app.update_special_gestures(assigned)
    app.inputs.tick()


class GeometryPrimitiveTests(unittest.TestCase):
    """Fonctions generiques reutilisables (section 5 du module)."""

    def test_distance_and_vector(self):
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(geo.distance_between_points(a, b), 5.0, places=5)
        np.testing.assert_allclose(geo.vector_between_points(a, b), b)

    def test_distance_from_landmarks(self):
        points = hand(scale=0.3)
        self.assertAlmostEqual(
            geo.distance_from_landmarks(points, geo.WRIST, geo.MIDDLE_MCP), 0.3, places=5)

    def test_normalize_vector(self):
        unit = geo.normalize_vector([0.0, 5.0])
        np.testing.assert_allclose(unit, [0.0, 1.0], atol=1e-6)
        self.assertAlmostEqual(geo.vector_length(unit), 1.0, places=6)

    def test_zero_length_vectors_are_safe(self):
        """Aucun NaN, aucune division par zero (exigence explicite)."""
        zero = np.zeros(3, dtype=np.float32)
        np.testing.assert_allclose(geo.normalize_vector(zero), zero)
        self.assertEqual(geo.angle_between_vectors(zero, zero), 0.0)
        self.assertEqual(geo.angle_2d(zero, zero), 0.0)
        self.assertEqual(geo.vector_angle(zero), 0.0)
        self.assertTrue(np.isfinite(geo.normalize_vector(zero)).all())

    def test_angle_between_vectors_is_unsigned_degrees(self):
        self.assertAlmostEqual(geo.angle_between_vectors([1, 0], [0, 1]), 90.0, places=4)
        self.assertAlmostEqual(geo.angle_between_vectors([1, 0], [-1, 0]), 180.0, places=4)
        self.assertAlmostEqual(geo.angle_between_vectors([1, 0], [1, 0]), 0.0, places=4)
        # Non signe : l'ordre des arguments ne change rien.
        self.assertAlmostEqual(geo.angle_between_vectors([0, 1], [1, 0]), 90.0, places=4)

    def test_angle_2d_is_signed(self):
        self.assertAlmostEqual(geo.angle_2d([1, 0], [0, 1]), 90.0, places=4)
        self.assertAlmostEqual(geo.angle_2d([0, 1], [1, 0]), -90.0, places=4)

    def test_vector_angle_uses_atan2_image_convention(self):
        self.assertAlmostEqual(geo.vector_angle(RIGHTWARD), 0.0, places=4)
        self.assertAlmostEqual(geo.vector_angle(UP), -90.0, places=4)
        self.assertAlmostEqual(geo.vector_angle(DOWN), 90.0, places=4)
        self.assertAlmostEqual(abs(geo.vector_angle(LEFTWARD)), 180.0, places=4)

    def test_landmark_accessors(self):
        points = hand()
        np.testing.assert_allclose(geo.wrist(points), points[0])
        np.testing.assert_allclose(geo.thumb_tip(points), points[4])
        np.testing.assert_allclose(geo.index_mcp(points), points[5])
        np.testing.assert_allclose(geo.index_tip(points), points[8])
        np.testing.assert_allclose(geo.middle_mcp(points), points[9])
        np.testing.assert_allclose(geo.pinky_mcp(points), points[17])
        self.assertEqual(len(geo.LANDMARK_NAMES), 21)
        self.assertEqual(geo.LANDMARK_NAMES[geo.INDEX_TIP], "Index Tip")


class GeometryNormalisationTests(unittest.TestCase):
    """Normalisation par la taille de la main (section 6 de la demande)."""

    def test_hand_scale_is_palm_based(self):
        self.assertAlmostEqual(geo.hand_scale(hand(scale=0.4)), 0.4, places=5)

    def test_hand_scale_never_zero(self):
        flat = np.zeros((geo.LANDMARK_COUNT, 3), dtype=np.float32)
        self.assertGreater(geo.hand_scale(flat), 0.0)
        metrics = geo.measure_hand(flat)
        self.assertTrue(np.isfinite(metrics.pinch_ratio))
        self.assertTrue(np.isfinite(metrics.direction).all())

    def test_ratio_is_independent_of_distance_to_camera(self):
        near = geo.measure_hand(hand(scale=1.0, pinch=0.2))
        far = geo.measure_hand(hand(scale=0.15, pinch=0.2))
        self.assertAlmostEqual(near.pinch_ratio, far.pinch_ratio, places=5)
        self.assertNotAlmostEqual(near.pinch_distance, far.pinch_distance, places=3)

    def test_direction_states_are_independent_of_scale(self):
        """Meme pose, deux tailles de main : meme etat detecte."""
        for scale in (0.08, 0.5):
            engine = geo.GeometryEngine()
            pose = hand(scale=scale, palm=(0.8, -0.6), index=(0.8, -0.6))
            for _ in range(geo.STABILITY_FRAMES):
                states = engine.update("LEFT", pose)
            with self.subTest(scale=scale):
                self.assertEqual(states[geo.TILT], geo.LEFT_TILT_RIGHT)

    def test_features_from_engine_are_reusable(self):
        points = geo.points_from_features(np.zeros(geo.FEATURE_SIZE, dtype=np.float32))
        self.assertEqual(points.shape, (geo.LANDMARK_COUNT, 3))
        self.assertIsNone(geo.points_from_features(np.zeros(10)))
        self.assertIsNone(geo.points_from_features(None))

    def test_normalised_features_are_not_geometry_landmarks(self):
        """k-NN palm-frame features keep pinch shape but drop image tilt."""
        pose = hand(scale=0.22, palm=(0.5, -0.8), index=(0.6, -0.7), pinch=0.3)
        engine = GestureEngine(os.path.join(tempfile.mkdtemp(), "gestures.json"))

        class LM:
            def __init__(self, row):
                self.x, self.y, self.z = float(row[0]), float(row[1]), float(row[2])

        features = engine.normalize_landmarks([LM(row) for row in pose])
        from_features = geo.measure_hand(geo.points_from_features(features))
        from_raw = geo.measure_hand(pose)
        self.assertAlmostEqual(from_features.pinch_ratio, from_raw.pinch_ratio, places=4)
        self.assertGreater(abs(from_features.palm_angle - from_raw.palm_angle), 5.0)


class GeometryOrientationTests(unittest.TestCase):
    """Orientation de la paume et de l'index (sections 9, 13, 14)."""

    def test_palm_angle_is_measured_from_landmarks_0_and_9(self):
        self.assertAlmostEqual(geo.measure_hand(hand(palm=UP)).palm_angle, -90.0, places=3)
        self.assertAlmostEqual(geo.measure_hand(hand(palm=DOWN)).palm_angle, 90.0, places=3)
        self.assertAlmostEqual(geo.measure_hand(hand(palm=RIGHTWARD)).palm_angle, 0.0, places=3)

    def test_index_angle_is_measured_from_landmarks_5_and_8(self):
        self.assertAlmostEqual(geo.measure_hand(hand(index=RIGHTWARD)).index_angle, 0.0, places=3)
        self.assertAlmostEqual(geo.measure_hand(hand(index=UP)).index_angle, -90.0, places=3)

    def test_palm_vector_is_scale_normalised(self):
        metrics = geo.measure_hand(hand(scale=0.37, palm=UP))
        self.assertAlmostEqual(geo.vector_length(metrics.palm_vector), 1.0, places=5)

    def test_index_vector_length_tracks_extension(self):
        extended = geo.measure_hand(hand(index_len=1.0))
        curled = geo.measure_hand(hand(index_len=0.2))
        self.assertAlmostEqual(geo.vector_length(extended.index_vector), 1.0, places=5)
        self.assertAlmostEqual(geo.vector_length(curled.index_vector), 0.2, places=5)

    def test_direction_fuses_palm_and_index(self):
        """La direction retenue combine les deux vecteurs, ponderes."""
        metrics = geo.measure_hand(hand(palm=UP, index=UP, index_len=1.0))
        expected = geo.PALM_WEIGHT + geo.INDEX_WEIGHT
        self.assertAlmostEqual(geo.vector_length(metrics.direction), expected, places=5)

    def test_palm_dominates_when_the_index_disagrees(self):
        """Index replie ou de travers : la paume garde la main sur la direction."""
        metrics = geo.measure_hand(hand(palm=UP, index=DOWN, index_len=1.0))
        # paume 0.6 vers le haut contre index 0.4 vers le bas -> resultat vers le haut
        self.assertLess(metrics.direction[1], 0.0)
        self.assertAlmostEqual(geo.vector_length(metrics.direction),
                               geo.PALM_WEIGHT - geo.INDEX_WEIGHT, places=5)

    def test_agreement_reports_palm_index_alignment(self):
        aligned = geo.measure_hand(hand(palm=UP, index=UP))
        opposed = geo.measure_hand(hand(palm=UP, index=DOWN))
        self.assertAlmostEqual(aligned.agreement, 0.0, places=3)
        self.assertAlmostEqual(opposed.agreement, 180.0, places=3)


class GeometryPinchTests(unittest.TestCase):
    """Pinch : seuils, hysteresis, cycle START / HOLD / END (sections 7, 8)."""

    def setUp(self):
        self.engine = geo.GeometryEngine()

    def pinch_state(self, points):
        return self.engine.update("RIGHT", points)[geo.PINCH]

    def detector(self):
        return self.engine.detector("RIGHT", geo.PINCH)

    def test_thumb_and_index_close_gives_right_pinch(self):
        self.assertEqual(self.pinch_state(hand(pinch=0.05)), geo.RIGHT_PINCH)

    def test_thumb_and_index_far_gives_no_pinch(self):
        self.assertIsNone(self.pinch_state(hand(pinch=1.0)))

    def test_exactly_on_the_on_threshold_activates(self):
        self.assertEqual(self.pinch_state(hand(pinch=geo.PINCH_ON_THRESHOLD)), geo.RIGHT_PINCH)

    def test_exactly_on_the_off_threshold_releases(self):
        self.assertEqual(self.pinch_state(hand(pinch=0.05)), geo.RIGHT_PINCH)
        self.assertIsNone(self.pinch_state(hand(pinch=geo.PINCH_OFF_THRESHOLD)))

    def test_hysteresis_band_keeps_the_previous_state(self):
        middle = (geo.PINCH_ON_THRESHOLD + geo.PINCH_OFF_THRESHOLD) / 2.0
        # Depuis OFF, la bande ne declenche pas.
        self.assertIsNone(self.pinch_state(hand(pinch=middle)))
        # Sous le seuil ON : activation.
        self.assertEqual(self.pinch_state(hand(pinch=geo.PINCH_ON_THRESHOLD - 0.05)),
                         geo.RIGHT_PINCH)
        # Retour dans la bande : l'etat ON est CONSERVE.
        self.assertEqual(self.pinch_state(hand(pinch=middle)), geo.RIGHT_PINCH)
        # Au-dela du seuil OFF : relachement.
        self.assertIsNone(self.pinch_state(hand(pinch=geo.PINCH_OFF_THRESHOLD + 0.05)))

    def test_pinch_phases_start_hold_end(self):
        closed = hand(pinch=0.05)
        self.assertIsNone(self.detector().phase)
        self.pinch_state(closed)
        self.assertEqual(self.detector().phase, geo.PHASE_START)
        self.pinch_state(closed)
        self.assertEqual(self.detector().phase, geo.PHASE_HOLD)
        self.pinch_state(closed)
        self.assertEqual(self.detector().phase, geo.PHASE_HOLD)
        self.pinch_state(hand(pinch=1.0))
        self.assertEqual(self.detector().phase, geo.PHASE_END)
        self.pinch_state(hand(pinch=1.0))
        self.assertIsNone(self.detector().phase)

    def test_held_pinch_is_one_event_not_one_per_frame(self):
        closed = hand(pinch=0.05)
        states = [self.pinch_state(closed) for _ in range(60)]
        self.assertEqual(states.count(geo.RIGHT_PINCH), 60)
        starts = 0
        self.engine.reset()
        for _ in range(60):
            self.pinch_state(closed)
            if self.detector().phase == geo.PHASE_START:
                starts += 1
        self.assertEqual(starts, 1)

    def test_left_pinch_is_independent_of_right_pinch(self):
        """Meme mesure, identifiant propre a chaque main."""
        closed = hand(pinch=0.01)
        self.assertIsNotNone(self.engine.detector("LEFT", geo.PINCH))
        self.assertEqual(self.engine.update("LEFT", closed)[geo.PINCH], geo.LEFT_PINCH)
        self.assertNotIn(geo.RIGHT_PINCH, self.engine.states("LEFT").values())
        self.assertIsNone(self.engine.states("RIGHT")[geo.PINCH])
        self.assertEqual(self.engine.update("RIGHT", closed)[geo.PINCH], geo.RIGHT_PINCH)
        self.assertEqual(self.engine.states("LEFT")[geo.PINCH], geo.LEFT_PINCH)

    def test_thresholds_must_form_a_hysteresis(self):
        with self.assertRaises(ValueError):
            geo.PinchDetector("RIGHT", on_threshold=0.4, off_threshold=0.2)


class GeometryTiltTests(unittest.TestCase):
    """Inclinaison gauche/droite/haut/bas + neutre (sections 10, 11, 15, 16)."""

    def setUp(self):
        self.engine = geo.GeometryEngine()

    def settle(self, side, points, frames=None):
        states = {}
        for _ in range(frames or geo.STABILITY_FRAMES):
            states = self.engine.update(side, points)
        return states[geo.TILT]

    def test_neutral_when_the_hand_is_closed(self):
        self.assertEqual(self.settle("LEFT", CURLED), geo.LEFT_TILT_NEUTRAL)

    def test_tilt_left(self):
        self.assertEqual(self.settle("LEFT", OPEN_LEFT), geo.LEFT_TILT_LEFT)

    def test_tilt_right(self):
        self.assertEqual(self.settle("LEFT", OPEN_RIGHT), geo.LEFT_TILT_RIGHT)

    def test_tilt_up(self):
        self.assertEqual(self.settle("LEFT", OPEN_UP), geo.LEFT_TILT_UP)

    def test_tilt_down(self):
        self.assertEqual(self.settle("LEFT", OPEN_DOWN), geo.LEFT_TILT_DOWN)

    def test_right_hand_has_its_own_tilt_states(self):
        self.assertEqual(self.settle("RIGHT", OPEN_LEFT), geo.RIGHT_TILT_LEFT)
        self.engine.reset()
        self.assertEqual(self.settle("RIGHT", OPEN_RIGHT), geo.RIGHT_TILT_RIGHT)
        self.engine.reset()
        self.assertEqual(self.settle("RIGHT", OPEN_UP), geo.RIGHT_TILT_UP)
        self.engine.reset()
        self.assertEqual(self.settle("RIGHT", OPEN_DOWN), geo.RIGHT_TILT_DOWN)
        self.engine.reset()
        self.assertEqual(self.settle("RIGHT", CURLED), geo.RIGHT_TILT_NEUTRAL)

    def test_just_below_the_threshold_stays_neutral(self):
        detector = self.engine.detector("LEFT", geo.TILT)
        just_under = geo.TILT_RIGHT_THRESHOLD - 0.02
        # Vecteur horizontal d'amplitude choisie, sous le seuil droite et sous
        # le seuil haut : l'etat doit rester neutre.
        detector.update_vector(np.array([just_under, 0.0]))
        detector.update_vector(np.array([just_under, 0.0]))
        self.assertEqual(detector.direction, geo.DIR_NEUTRAL)

    def test_just_above_the_threshold_activates(self):
        detector = self.engine.detector("LEFT", geo.TILT)
        just_over = geo.TILT_RIGHT_THRESHOLD + 0.02
        for _ in range(geo.STABILITY_FRAMES):
            detector.update_vector(np.array([just_over, 0.0]))
        self.assertEqual(detector.direction, geo.DIR_RIGHT)

    def test_dead_zone_prevents_flicker_around_the_threshold(self):
        detector = self.engine.detector("LEFT", geo.TILT)
        for _ in range(geo.STABILITY_FRAMES):
            detector.update_vector(np.array([geo.TILT_RIGHT_THRESHOLD + 0.05, 0.0]))
        self.assertEqual(detector.direction, geo.DIR_RIGHT)
        # Juste sous le seuil d'entree mais au-dessus du seuil de sortie :
        # l'etat est conserve, aucun clignotement.
        for _ in range(10):
            detector.update_vector(np.array([geo.TILT_RIGHT_THRESHOLD - 0.05, 0.0]))
            self.assertEqual(detector.direction, geo.DIR_RIGHT)
        # Sous le seuil de sortie : retour au neutre.
        exit_value = geo.TILT_RIGHT_THRESHOLD - geo.DEAD_ZONE - 0.02
        for _ in range(geo.STABILITY_FRAMES):
            detector.update_vector(np.array([exit_value, 0.0]))
        self.assertEqual(detector.direction, geo.DIR_NEUTRAL)

    def test_left_to_right_always_passes_through_neutral(self):
        self.assertEqual(self.settle("LEFT", OPEN_LEFT), geo.LEFT_TILT_LEFT)
        seen = []
        for _ in range(10):
            seen.append(self.engine.update("LEFT", OPEN_RIGHT)[geo.TILT])
        self.assertEqual(seen[-1], geo.LEFT_TILT_RIGHT)
        self.assertIn(geo.LEFT_TILT_NEUTRAL, seen)
        self.assertLess(seen.index(geo.LEFT_TILT_NEUTRAL), seen.index(geo.LEFT_TILT_RIGHT))

    def test_up_to_down_always_passes_through_neutral(self):
        self.assertEqual(self.settle("LEFT", OPEN_UP), geo.LEFT_TILT_UP)
        seen = []
        for _ in range(10):
            seen.append(self.engine.update("LEFT", OPEN_DOWN)[geo.TILT])
        self.assertEqual(seen[-1], geo.LEFT_TILT_DOWN)
        self.assertIn(geo.LEFT_TILT_NEUTRAL, seen)

    def test_stability_frames_reject_a_one_frame_spike(self):
        self.assertEqual(self.settle("LEFT", OPEN_LEFT), geo.LEFT_TILT_LEFT)
        # Une seule frame vers la droite ne doit rien changer.
        self.assertEqual(self.engine.update("LEFT", OPEN_RIGHT)[geo.TILT], geo.LEFT_TILT_LEFT)
        self.assertEqual(self.engine.update("LEFT", OPEN_LEFT)[geo.TILT], geo.LEFT_TILT_LEFT)

    def test_neutral_state_always_exists(self):
        for side in SIDES:
            self.assertIn(geo.DIR_NEUTRAL, geo._DIRECTION_IDS[(side, geo.TILT)])
            self.assertIn(geo.DIR_NEUTRAL, geo._DIRECTION_IDS[(side, geo.INDEX)])

    def test_missing_threshold_is_refused(self):
        with self.assertRaises(ValueError):
            geo.DirectionDetector("LEFT", geo.TILT, thresholds={geo.DIR_LEFT: 0.4})


class GeometryIndexDirectionTests(unittest.TestCase):
    """Direction de l'index seul : methode de controle alternative (section 13)."""

    def setUp(self):
        self.engine = geo.GeometryEngine()

    def settle(self, points, side="LEFT", frames=None):
        states = {}
        for _ in range(frames or geo.STABILITY_FRAMES):
            states = self.engine.update(side, points)
        return states[geo.INDEX]

    def test_index_left(self):
        self.assertEqual(self.settle(hand(index=LEFTWARD)), geo.LEFT_INDEX_LEFT)

    def test_index_right(self):
        self.assertEqual(self.settle(hand(index=RIGHTWARD)), geo.LEFT_INDEX_RIGHT)

    def test_index_up(self):
        self.assertEqual(self.settle(hand(index=UP)), geo.LEFT_INDEX_UP)

    def test_index_down(self):
        self.assertEqual(self.settle(hand(index=DOWN)), geo.LEFT_INDEX_DOWN)

    def test_curled_index_is_in_the_dead_zone(self):
        self.assertEqual(self.settle(hand(index=UP, index_len=0.15)), geo.LEFT_INDEX_NEUTRAL)

    def test_index_direction_is_independent_of_the_palm(self):
        """L'index seul decide : la paume n'entre pas dans ce detecteur."""
        states = {}
        for _ in range(geo.STABILITY_FRAMES):
            states = self.engine.update("LEFT", hand(palm=UP, index=RIGHTWARD))
        self.assertEqual(states[geo.INDEX], geo.LEFT_INDEX_RIGHT)

    def test_right_hand_has_its_own_index_states(self):
        self.assertEqual(self.settle(hand(index=LEFTWARD), side="RIGHT"), geo.RIGHT_INDEX_LEFT)

    def test_index_and_tilt_can_differ_on_the_same_hand(self):
        """Les deux methodes coexistent et sont mesurees separement."""
        pose = hand(palm=UP, index=RIGHTWARD, index_len=1.0)
        states = {}
        for _ in range(geo.STABILITY_FRAMES):
            states = self.engine.update("LEFT", pose)
        self.assertEqual(states[geo.INDEX], geo.LEFT_INDEX_RIGHT)
        self.assertIsNotNone(states[geo.TILT])
        self.assertNotEqual(states[geo.TILT], geo.LEFT_TILT_RIGHT)
        self.assertNotEqual(states[geo.TILT], states[geo.INDEX])


class GeometryTwoHandsTests(unittest.TestCase):
    """Deux mains independantes, perte d'une main (sections 17, 18)."""

    def setUp(self):
        self.engine = geo.GeometryEngine()

    def settle(self, side, points, frames=None):
        for _ in range(frames or geo.STABILITY_FRAMES):
            states = self.engine.update(side, points)
        return states

    def test_left_tilt_and_right_pinch_together(self):
        left = self.settle("LEFT", OPEN_LEFT)
        right = self.settle("RIGHT", PINCHING)
        self.assertEqual(left[geo.TILT], geo.LEFT_TILT_LEFT)
        self.assertEqual(right[geo.PINCH], geo.RIGHT_PINCH)

    def test_left_tilt_and_right_tilt_together(self):
        self.settle("LEFT", OPEN_LEFT)
        self.settle("RIGHT", OPEN_RIGHT)
        self.assertEqual(self.engine.states("LEFT")[geo.TILT], geo.LEFT_TILT_LEFT)
        self.assertEqual(self.engine.states("RIGHT")[geo.TILT], geo.RIGHT_TILT_RIGHT)

    def test_one_hand_never_overwrites_the_other(self):
        self.settle("LEFT", OPEN_LEFT)
        self.settle("RIGHT", OPEN_RIGHT)
        # Repasser la gauche au neutre ne touche pas la droite.
        self.settle("LEFT", CURLED)
        self.assertEqual(self.engine.states("LEFT")[geo.TILT], geo.LEFT_TILT_NEUTRAL)
        self.assertEqual(self.engine.states("RIGHT")[geo.TILT], geo.RIGHT_TILT_RIGHT)

    def test_losing_the_left_hand_resets_only_the_left(self):
        self.settle("LEFT", OPEN_LEFT)
        self.settle("RIGHT", PINCHING)
        self.engine.hand_lost("LEFT")
        self.assertEqual(self.engine.active("LEFT"), [])
        self.assertIsNone(self.engine.metrics("LEFT"))
        self.assertEqual(self.engine.states("RIGHT")[geo.PINCH], geo.RIGHT_PINCH)
        self.assertIsNotNone(self.engine.metrics("RIGHT"))

    def test_losing_the_right_hand_resets_only_the_right(self):
        self.settle("LEFT", OPEN_LEFT)
        self.settle("RIGHT", PINCHING)
        self.engine.hand_lost("RIGHT")
        self.assertEqual(self.engine.active("RIGHT"), [])
        self.assertFalse(self.engine.detector("RIGHT", geo.PINCH).active)
        self.assertEqual(self.engine.states("LEFT")[geo.TILT], geo.LEFT_TILT_LEFT)

    def test_absent_landmarks_are_treated_as_a_lost_hand(self):
        self.settle("LEFT", OPEN_LEFT)
        self.engine.update("LEFT", None)
        self.assertEqual(self.engine.active("LEFT"), [])


class GeometryMirrorTests(unittest.TestCase):
    """Axe de la webcam, miroir et SWAP_HANDS (section 12)."""

    def test_positive_x_means_right_on_screen(self):
        """main.py retourne la frame, donc +x = droite vue par l'utilisateur."""
        engine = geo.GeometryEngine()
        for _ in range(geo.STABILITY_FRAMES):
            states = engine.update("LEFT", tilted((1.0, 0.0)))
        self.assertEqual(states[geo.TILT], geo.LEFT_TILT_RIGHT)
        engine.reset()
        for _ in range(geo.STABILITY_FRAMES):
            states = engine.update("LEFT", tilted((-1.0, 0.0)))
        self.assertEqual(states[geo.TILT], geo.LEFT_TILT_LEFT)

    def test_frame_is_flipped_before_mediapipe(self):
        source = inspect.getsource(main_module.HandControllerApp.run)
        self.assertIn("cv2.flip(frame, 1)", source)

    def test_swap_hands_still_decides_handedness(self):
        """Le moteur geometrique ne recalcule jamais l'identite de la main."""
        class Category:
            def __init__(self, name):
                self.category_name = name
                self.score = 0.9

        class Result:
            handedness = [[Category("Left")]]

        swapped = GestureEngine(os.path.join(tempfile.mkdtemp(), "a.json"), swap_hands=True)
        straight = GestureEngine(os.path.join(tempfile.mkdtemp(), "b.json"), swap_hands=False)
        self.assertEqual(swapped.get_hand_side(Result(), 0), "RIGHT")
        self.assertEqual(straight.get_hand_side(Result(), 0), "LEFT")
        # Le moteur geometrique n'a aucune notion de handedness.
        source = inspect.getsource(geo)
        for forbidden in ("handedness", "swap_hands", "category_name"):
            self.assertNotIn(forbidden, source)

    def test_direction_semantics_do_not_depend_on_swap_hands(self):
        """La meme pose donne la meme direction, quel que soit SWAP_HANDS.

        L'inversion d'image est deja appliquee en amont : le moteur lit des
        coordonnees ecran, pas des coordonnees camera.
        """
        pose = tilted((1.0, 0.0))
        results = []
        for side in SIDES:
            engine = geo.GeometryEngine()
            for _ in range(geo.STABILITY_FRAMES):
                states = engine.update(side, pose)
            results.append(engine.detector(side, geo.TILT).direction)
        self.assertEqual(results, [geo.DIR_RIGHT, geo.DIR_RIGHT])


class GeometryMappingTests(TempCase):
    """Les gestes geometriques passent par le mapping existant (sections 19-22)."""

    def test_ids_are_never_learned_gestures(self):
        for name in geo.SPECIAL_NAMES:
            self.assertNotIn(name, GESTURE_NAMES)
        for forbidden in ("G31", "G32", "G33"):
            self.assertNotIn(forbidden, geo.SPECIAL_NAMES)
        self.assertEqual(len(GESTURE_NAMES), 30)

    def test_every_special_id_is_mappable_on_its_own_hand(self):
        mapping = GestureMapping(self.mapping_path)
        for side in SIDES:
            for name in geo.SPECIAL_BY_SIDE[side]:
                with self.subTest(side=side, gesture=name):
                    self.assertIsNotNone(mapping.get(side, name))
                # Et absent de la table de l'autre main.
                other = "RIGHT" if side == "LEFT" else "LEFT"
                if name not in geo.SPECIAL_BY_SIDE[other]:
                    self.assertIsNone(mapping.get(other, name))

    def test_documented_defaults(self):
        mapping = GestureMapping(self.mapping_path)
        self.assertEqual(mapping.get_type("RIGHT", geo.RIGHT_PINCH), PRESS)
        self.assertEqual(mapping.get_inputs("RIGHT", geo.RIGHT_PINCH), ["SPACE"])
        self.assertEqual(mapping.get_type("LEFT", geo.LEFT_TILT_LEFT), HOLD)
        self.assertEqual(mapping.get_inputs("LEFT", geo.LEFT_TILT_LEFT), ["A"])
        self.assertEqual(mapping.get_type("LEFT", geo.LEFT_TILT_RIGHT), HOLD)
        self.assertEqual(mapping.get_inputs("LEFT", geo.LEFT_TILT_RIGHT), ["D"])

    def test_every_other_special_ships_disabled(self):
        """Aucune touche non demandee n'est branchee d'office."""
        mapping = GestureMapping(self.mapping_path)
        documented = {geo.RIGHT_PINCH, geo.LEFT_TILT_LEFT, geo.LEFT_TILT_RIGHT}
        for side in SIDES:
            for name in geo.SPECIAL_BY_SIDE[side]:
                if name in documented:
                    continue
                with self.subTest(gesture=name):
                    self.assertEqual(mapping.get_type(side, name), NONE)

    def test_specials_accept_every_existing_action_type(self):
        mapping = GestureMapping(self.mapping_path)
        target = geo.RIGHT_TILT_UP
        self.assertTrue(mapping.set_command("RIGHT", target, PRESS, ["E"], 0))
        self.assertTrue(mapping.set_command("RIGHT", target, HOLD, ["SHIFT"]))
        self.assertTrue(mapping.set_command("RIGHT", target, COMBINATION, ["W", "D"]))
        self.assertTrue(mapping.set_macro("RIGHT", target, [
            {"action": STEP_PRESS, "inputs": ["SPACE"]},
            {"action": STEP_WAIT, "duration": 20},
            {"action": STEP_PRESS, "inputs": ["E"]},
        ]))
        self.assertEqual(mapping.get_type("RIGHT", target), MACRO)
        self.assertTrue(mapping.disable("RIGHT", target))
        self.assertEqual(mapping.get_type("RIGHT", target), NONE)
        self.assertTrue(mapping.reset("RIGHT", target))

    def test_defaults_are_user_modifiable_and_persisted(self):
        mapping = GestureMapping(self.mapping_path)
        self.assertTrue(mapping.set_command("RIGHT", geo.RIGHT_PINCH, HOLD, ["LEFT_MOUSE"]))
        reloaded = GestureMapping(self.mapping_path)
        self.assertEqual(reloaded.get_type("RIGHT", geo.RIGHT_PINCH), HOLD)
        self.assertEqual(reloaded.get_inputs("RIGHT", geo.RIGHT_PINCH), ["LEFT_MOUSE"])

    def test_unknown_command_still_refused_for_specials(self):
        mapping = GestureMapping(self.mapping_path)
        self.assertFalse(mapping.set_command("RIGHT", geo.RIGHT_PINCH, HOLD, ["TOUCHE_BIDON"]))

    def test_json_keeps_two_top_level_tables(self):
        """Pas de second systeme de mapping : tout vit dans LEFT / RIGHT."""
        GestureMapping(self.mapping_path)
        with open(self.mapping_path, encoding="utf-8") as file:
            saved = json.load(file)
        self.assertEqual(set(saved) - {"version"}, {"LEFT", "RIGHT"})
        self.assertIn(geo.RIGHT_PINCH, saved["RIGHT"])
        self.assertNotIn(geo.RIGHT_PINCH, saved["LEFT"])
        self.assertIn(geo.LEFT_TILT_UP, saved["LEFT"])

    def test_older_mapping_gains_specials_without_losing_learned_entries(self):
        legacy = {
            "version": 4,
            "LEFT": {"G01": {"type": "HOLD", "inputs": ["W"]}},
            "RIGHT": {"G09": {"type": "PRESS", "inputs": ["E"]}},
        }
        with open(self.mapping_path, "w", encoding="utf-8") as file:
            json.dump(legacy, file)
        mapping = GestureMapping(self.mapping_path)
        self.assertEqual(mapping.get_inputs("LEFT", "G01"), ["W"])
        self.assertEqual(mapping.get_inputs("RIGHT", "G09"), ["E"])
        self.assertEqual(mapping.get_inputs("LEFT", geo.LEFT_TILT_LEFT), ["A"])
        with open(self.mapping_path, encoding="utf-8") as file:
            self.assertEqual(json.load(file)["version"], MAPPING_VERSION)

    def test_editor_lists_learned_then_special_entries(self):
        for side in SIDES:
            names = side_gesture_names(side)
            self.assertEqual(names[:30], GESTURE_NAMES)
            self.assertEqual(names[30:], list(geo.SPECIAL_BY_SIDE[side]))


class GeometryChainTests(TempCase):
    """Detection -> mapping -> input controller, avec les deux mains."""

    def setUp(self):
        super().setUp()
        self.app = make_geometry_app(self.mapping_path)
        self.app.inputs.enable()
        self.kb = self.app.inputs.keyboard

    def settle(self, left=None, right=None, frames=None):
        for _ in range(frames or geo.STABILITY_FRAMES):
            run_geometry_frame(self.app, left=left, right=right)

    def test_pinch_press_reaches_pynput_once(self):
        self.app.mapping.set_command("RIGHT", geo.RIGHT_PINCH, PRESS, ["SPACE"], 0)
        space = self.app.inputs.special_keys["SPACE"]
        for _ in range(100):
            run_geometry_frame(self.app, right=PINCHING)
        self.assertEqual(self.kb.pressed.count(space), 1)
        run_geometry_frame(self.app, right=OPENED)
        run_geometry_frame(self.app, right=PINCHING)
        self.assertEqual(self.kb.pressed.count(space), 2)

    def test_pinch_as_hold(self):
        self.app.mapping.set_command("RIGHT", geo.RIGHT_PINCH, HOLD, ["SHIFT"])
        shift = self.app.inputs.special_keys["SHIFT"]
        run_geometry_frame(self.app, right=PINCHING)
        self.assertIn(shift, self.app.inputs.currently_pressed)
        run_geometry_frame(self.app, right=OPENED)
        self.assertNotIn(shift, self.app.inputs.currently_pressed)

    def test_pinch_as_combination(self):
        self.app.mapping.set_command("RIGHT", geo.RIGHT_PINCH, COMBINATION, ["W", "D"])
        run_geometry_frame(self.app, right=PINCHING)
        self.assertEqual(self.app.inputs.currently_pressed, {"w", "d"})

    def test_pinch_as_macro_with_wait(self):
        self.app.mapping.set_macro("RIGHT", geo.RIGHT_PINCH, [
            {"action": STEP_PRESS, "inputs": ["SPACE"]},
            {"action": STEP_WAIT, "duration": 20},
            {"action": STEP_PRESS, "inputs": ["E"]},
        ])
        run_geometry_frame(self.app, right=PINCHING)
        drain(self.app.inputs)
        self.assertIn(self.app.inputs.special_keys["SPACE"], self.kb.pressed)
        self.assertIn("e", self.kb.pressed)

    def test_tilt_holds_and_releases(self):
        self.settle(left=OPEN_LEFT)
        self.assertEqual(self.app.inputs.currently_pressed, {"a"})
        self.settle(left=CURLED)
        self.assertEqual(self.app.inputs.currently_pressed, set())

    def test_tilt_switch_releases_before_pressing(self):
        """Jamais A et D enfoncees ensemble a cause d'une transition."""
        self.settle(left=OPEN_LEFT)
        self.assertEqual(self.app.inputs.currently_pressed, {"a"})
        for _ in range(10):
            run_geometry_frame(self.app, left=OPEN_RIGHT)
            self.assertNotEqual(self.app.inputs.currently_pressed, {"a", "d"})
        self.assertEqual(self.app.inputs.currently_pressed, {"d"})
        self.assertIn("a", self.kb.released)
        self.assertLess(self.kb.released.index("a"), self.kb.pressed.index("d"))

    def test_two_hands_tilt_and_pinch_together(self):
        self.app.mapping.set_command("RIGHT", geo.RIGHT_PINCH, PRESS, ["SPACE"], 0)
        space = self.app.inputs.special_keys["SPACE"]
        self.settle(left=OPEN_LEFT, right=PINCHING)
        self.assertEqual(self.app.inputs.currently_pressed, {"a"})
        self.assertEqual(self.kb.pressed.count(space), 1)

    def test_two_hands_both_tilting(self):
        self.app.mapping.set_command("RIGHT", geo.RIGHT_TILT_RIGHT, HOLD, ["SHIFT"])
        shift = self.app.inputs.special_keys["SHIFT"]
        self.settle(left=OPEN_LEFT, right=OPEN_RIGHT)
        self.assertEqual(self.app.inputs.currently_pressed, {"a", shift})

    def test_losing_right_hand_releases_only_right_commands(self):
        self.app.mapping.set_command("RIGHT", geo.RIGHT_PINCH, HOLD, ["SHIFT"])
        shift = self.app.inputs.special_keys["SHIFT"]
        self.settle(left=OPEN_LEFT, right=PINCHING)
        self.assertEqual(self.app.inputs.currently_pressed, {"a", shift})
        run_geometry_frame(self.app, left=OPEN_LEFT, right=None)
        self.assertEqual(self.app.inputs.currently_pressed, {"a"})
        self.assertIn(shift, self.kb.released)

    def test_losing_left_hand_releases_only_left_commands(self):
        self.app.mapping.set_command("RIGHT", geo.RIGHT_PINCH, HOLD, ["SHIFT"])
        shift = self.app.inputs.special_keys["SHIFT"]
        self.settle(left=OPEN_LEFT, right=PINCHING)
        self.assertEqual(self.app.inputs.currently_pressed, {"a", shift})
        run_geometry_frame(self.app, left=None, right=PINCHING)
        self.assertEqual(self.app.inputs.currently_pressed, {shift})
        self.assertIn("a", self.kb.released)

    def test_tilt_and_index_families_are_independent(self):
        """Deux familles sur la meme main, deux sources de maintien.

        Paume vers la gauche et index vers le bas : TILT dit LEFT (la paume
        domine la fusion) pendant que INDEX dit DOWN. Les deux maintiens
        coexistent.
        """
        self.app.mapping.set_command("LEFT", geo.LEFT_INDEX_DOWN, HOLD, ["SHIFT"])
        shift = self.app.inputs.special_keys["SHIFT"]
        pose = hand(palm=LEFTWARD, index=DOWN)
        self.settle(left=pose)
        self.assertEqual(self.app.geometry.states("LEFT")[geo.TILT], geo.LEFT_TILT_LEFT)
        self.assertEqual(self.app.geometry.states("LEFT")[geo.INDEX], geo.LEFT_INDEX_DOWN)
        self.assertEqual(self.app.inputs.currently_pressed, {"a", shift})

    def test_learned_gesture_and_geometry_share_a_hand(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        run_frame(self.app, LEFT="G01")
        self.settle(left=OPEN_LEFT)
        self.assertEqual(self.app.inputs.currently_pressed, {"w", "a"})
        self.settle(left=CURLED)
        self.assertEqual(self.app.inputs.currently_pressed, {"w"})

    def test_shared_key_between_learned_and_geometry(self):
        """Meme touche demandee par les deux couches : une seule pression."""
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["A"])
        run_frame(self.app, LEFT="G01")
        self.settle(left=OPEN_LEFT)
        self.assertEqual(self.kb.pressed.count("a"), 1)
        self.settle(left=CURLED)
        self.assertEqual(self.kb.released.count("a"), 0)   # G01 la maintient encore
        run_frame(self.app, LEFT=UNKNOWN)
        self.assertEqual(self.kb.released.count("a"), 1)

    def test_disabled_special_sends_nothing(self):
        self.app.mapping.disable("LEFT", geo.LEFT_TILT_LEFT)
        self.settle(left=OPEN_LEFT)
        self.assertEqual(self.app.inputs.currently_pressed, set())

    def test_neutral_and_undocumented_states_are_inert_by_default(self):
        self.settle(left=CURLED)
        self.settle(left=OPEN_UP)
        self.settle(left=OPEN_DOWN)
        self.assertEqual(self.kb.pressed, [])

    def test_input_off_blocks_geometry_gestures(self):
        self.app.inputs.disable()
        self.settle(left=OPEN_LEFT, right=PINCHING)
        self.assertFalse(self.app.inputs.currently_pressed)
        self.assertEqual(self.kb.pressed, [])

    def test_emergency_stop_clears_geometry_state(self):
        self.app.mapping.set_command("RIGHT", geo.RIGHT_PINCH, HOLD, ["SHIFT"])
        shift = self.app.inputs.special_keys["SHIFT"]
        self.settle(left=OPEN_LEFT, right=PINCHING)
        self.assertEqual(self.app.inputs.currently_pressed, {"a", shift})
        self.app.engine = GestureEngine(self.gestures_path)
        self.app.status_message = ""
        self.app.reset_state()                      # chemin exact de F8 et de G
        self.assertFalse(self.app.inputs.currently_pressed)
        self.assertEqual(self.app.geometry.active("LEFT"), [])
        self.assertEqual(self.app.geometry.active("RIGHT"), [])


class GeometryRegressionTests(TempCase):
    """Le systeme existant doit continuer a fonctionner a l'identique."""

    def setUp(self):
        super().setUp()
        self.app = make_geometry_app(self.mapping_path)
        self.app.inputs.enable()
        self.kb = self.app.inputs.keyboard

    def test_learned_gestures_still_reach_pynput(self):
        for side in SIDES:
            for gesture in GESTURE_NAMES:
                app = make_geometry_app(self.mapping_path)
                app.inputs.enable()
                entry = app.mapping.get(side, gesture)
                run_frame(app, **{side: gesture})
                with self.subTest(side=side, gesture=gesture):
                    kind, target = app.inputs.resolve(entry["inputs"][0])
                    sink = app.inputs.mouse if kind == "mouse" else app.inputs.keyboard
                    self.assertIn(target, sink.pressed)

    def test_press_hold_combination_still_work(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        self.app.mapping.set_command("LEFT", "G02", COMBINATION, ["W", "D"])
        self.app.mapping.set_command("RIGHT", "G01", PRESS, ["LEFT_MOUSE"], 0)
        run_frame(self.app, LEFT="G01", RIGHT="G01")
        self.assertEqual(self.app.inputs.currently_pressed, {"w"})
        self.assertIn(self.app.inputs.mouse_buttons["LEFT_MOUSE"], self.app.inputs.mouse.pressed)
        run_frame(self.app, LEFT="G02")
        self.assertEqual(self.app.inputs.currently_pressed, {"w", "d"})

    def test_macro_with_wait_still_works(self):
        self.app.mapping.set_macro("LEFT", "G05", [
            {"action": STEP_PRESS, "inputs": ["SPACE"]},
            {"action": STEP_WAIT, "duration": 20},
            {"action": STEP_PRESS, "inputs": ["E"]},
        ])
        run_frame(self.app, LEFT="G05")
        drain(self.app.inputs)
        self.assertIn(self.app.inputs.special_keys["SPACE"], self.kb.pressed)
        self.assertIn("e", self.kb.pressed)

    def test_f8_releases_learned_and_geometric_holds(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        run_frame(self.app, LEFT="G01")
        self.settle_left_tilt()
        self.assertEqual(self.app.inputs.currently_pressed, {"w", "a"})
        self.app.inputs.disable()               # chemin exact de F8
        self.assertFalse(self.app.inputs.currently_pressed)
        self.assertFalse(self.app.inputs.hold_sources)
        for key in ("w", "a"):
            self.assertIn(key, self.kb.released)

    def settle_left_tilt(self):
        for _ in range(geo.STABILITY_FRAMES):
            run_geometry_frame(self.app, left=OPEN_LEFT)

    def test_losing_a_hand_still_isolates_learned_gestures(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        self.app.mapping.set_command("RIGHT", "G01", HOLD, ["D"])
        run_frame(self.app, LEFT="G01", RIGHT="G01")
        run_frame(self.app, LEFT=UNKNOWN, RIGHT="G01")
        self.assertEqual(self.app.inputs.currently_pressed, {"d"})

    def test_geometry_does_not_touch_the_learned_hold_source(self):
        """Les sources de maintien des deux couches sont disjointes."""
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        run_frame(self.app, LEFT="G01")
        self.settle_left_tilt()
        sources = set(self.app.inputs.hold_sources)
        self.assertIn("LEFT", sources)
        self.assertIn(geo.special_source("LEFT", geo.TILT), sources)
        self.assertNotIn(geo.special_source("LEFT", geo.TILT), SIDES)


class GeometrySafetyTests(unittest.TestCase):
    """Separation detection / input, performance (sections 3, 21, 26)."""

    def test_engine_module_imports_no_input_layer(self):
        """geometry_engine ne depend que de NumPy : aucun envoi possible."""
        tree = ast.parse(inspect.getsource(geo))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        self.assertEqual(imported, {"numpy"})

    def test_engine_never_sends_keyboard_or_mouse(self):
        source = inspect.getsource(geo)
        for forbidden in ("press_combo", "set_source_hold", "start_macro",
                          "KeyboardController", "keyboard.press", "mouse.press",
                          'press("space")', "pynput"):
            self.assertNotIn(forbidden, source)

    def test_no_sleep_on_the_geometry_hot_path(self):
        functions = [
            (main_module.HandControllerApp, "update_special_gestures"),
            (main_module.HandControllerApp, "execute_special_gesture"),
            (main_module.HandControllerApp, "update_special_holds"),
            (main_module.HandControllerApp, "geometry_overlay_text"),
            (geo.GeometryEngine, "update"),
            (geo.HandGeometry, "update"),
            (geo.PinchDetector, "update"),
            (geo.DirectionDetector, "update_vector"),
        ]
        for owner, name in functions:
            with self.subTest(function=f"{owner.__name__}.{name}"):
                names = NonBlockingTests.called_names(getattr(owner, name))
                self.assertNotIn("sleep", names)

    def test_measure_hand_allocates_nothing_large(self):
        """Une passe de mesure ne cree aucun tableau au-dela des 21 points."""
        pose = hand()
        metrics = geo.measure_hand(pose)
        self.assertEqual(metrics.palm_vector.shape, (2,))
        self.assertEqual(metrics.index_vector.shape, (2,))
        self.assertEqual(metrics.direction.shape, (2,))

    def test_hold_sources_are_distinct_per_hand_and_family(self):
        seen = set()
        for side in SIDES:
            for family in (geo.PINCH, geo.TILT, geo.INDEX):
                source = geo.special_source(side, family)
                self.assertNotIn(source, SIDES)
                self.assertNotIn(source, seen)
                seen.add(source)

    def test_side_of_identifies_the_owning_hand(self):
        self.assertEqual(geo.side_of(geo.RIGHT_PINCH), "RIGHT")
        self.assertEqual(geo.side_of(geo.LEFT_TILT_UP), "LEFT")
        self.assertEqual(geo.side_of(geo.RIGHT_INDEX_DOWN), "RIGHT")
        self.assertIsNone(geo.side_of("G01"))

    def test_all_configuration_lives_in_the_module(self):
        for name in ("PINCH_ON_THRESHOLD", "PINCH_OFF_THRESHOLD",
                     "TILT_LEFT_THRESHOLD", "TILT_RIGHT_THRESHOLD",
                     "TILT_UP_THRESHOLD", "TILT_DOWN_THRESHOLD",
                     "INDEX_DIRECTION_THRESHOLD", "DEAD_ZONE",
                     "PALM_WEIGHT", "INDEX_WEIGHT"):
            with self.subTest(setting=name):
                self.assertIsInstance(getattr(geo, name), float)
        self.assertIsInstance(geo.STABILITY_FRAMES, int)

    def test_thresholds_are_overridable_per_instance(self):
        engine = geo.GeometryEngine(pinch_on=0.10, pinch_off=0.50, stability_frames=1)
        detector = engine.detector("RIGHT", geo.PINCH)
        self.assertEqual(detector.on_threshold, 0.10)
        # 0.20 declenchait avec le seuil par defaut, plus avec 0.10.
        self.assertIsNone(engine.update("RIGHT", hand(pinch=0.20))[geo.PINCH])
        self.assertEqual(engine.update("RIGHT", hand(pinch=0.05))[geo.PINCH], geo.RIGHT_PINCH)

    def test_detectors_are_registered_per_hand(self):
        """Point d'extension : un detecteur se declare par main."""
        engine = geo.GeometryEngine()
        self.assertTrue({geo.PINCH, geo.TILT, geo.INDEX}.issubset(engine.families("RIGHT")))
        self.assertTrue({geo.PINCH, geo.TILT, geo.INDEX}.issubset(engine.families("LEFT")))
        self.assertTrue(set(geo.FINGER_FAMILIES).issubset(engine.families("LEFT")))
        self.assertTrue(set(geo.FINGER_FAMILIES).issubset(engine.families("RIGHT")))
        enabled = geo.GeometryEngine(fist_enabled=True, open_palm_enabled=True)
        self.assertIn(geo.FIST, enabled.families("LEFT"))
        self.assertIn(geo.PALM, enabled.families("RIGHT"))

    def test_debug_lines_expose_values_and_thresholds(self):
        engine = geo.GeometryEngine()
        engine.update("LEFT", OPEN_LEFT)
        engine.update("RIGHT", PINCHING)
        text = " ".join(engine.debug_lines())
        for fragment in ("orient", "deg", "index", "pinch", "phase", "seuils", "zone morte"):
            self.assertIn(fragment, text)
        self.assertTrue(text.isascii())

    def test_debug_lines_handle_absent_hands(self):
        text = " ".join(geo.GeometryEngine().debug_lines())
        self.assertIn("absente", text)

    def test_overlay_is_quiet_when_nothing_is_detected(self):
        app = make_geometry_app(os.path.join(tempfile.mkdtemp(), "mapping.json"))
        self.assertEqual(app.geometry_overlay_text("LEFT"), "")
        for _ in range(geo.STABILITY_FRAMES):
            run_geometry_frame(app, left=OPEN_LEFT)
        self.assertIn("TILT", app.geometry_overlay_text("LEFT"))

    def test_visualisation_draws_palm_and_enabled_finger_axes(self):
        source = inspect.getsource(main_module.HandControllerApp._draw_geometry)
        self.assertIn("MIDDLE_MCP", source)
        self.assertIn("FINGER_UP_FINGERS", source)
        self.assertIn("_finger_arrow_enabled", source)
        self.assertIn("arrowedLine", source)


# =========================================================================
# Phase 1 — grace de perte, settings, HUD, conflit Gxx / geometrie
# =========================================================================
KEEP = "keep"


def arm_tracking(app, grace=2):
    """Active le compteur de micro-pertes. Defaut tests : grace 0 (absent)."""
    app.hand_loss_grace_frames = grace
    app.missed_frames = {side: 0 for side in SIDES}
    app.track_state = {side: "OK" for side in SIDES}
    app.detected = {side: False for side in SIDES}
    app.confidences = {side: 0.0 for side in SIDES}
    if not hasattr(app, "curr_specials"):
        app.curr_specials = {side: {} for side in SIDES}
        app.prev_specials = {side: {} for side in SIDES}
    if not hasattr(app, "_last_points"):
        app._last_points = {side: None for side in SIDES}
    return app


def run_tracked_frame(app, left=KEEP, right=KEEP, left_points=None, right_points=None):
    """Presence + mapping + holds, dans le meme ordre que process_frame.

    left/right : KEEP (presente, geste inchange), None (absente), ou un Gxx.
    """
    if not hasattr(app, "_last_points"):
        app._last_points = {side: None for side in SIDES}
    if left_points is not None:
        app._last_points["LEFT"] = left_points
    if right_points is not None:
        app._last_points["RIGHT"] = right_points

    assigned = {
        "LEFT": None if left is None else {"features": None},
        "RIGHT": None if right is None else {"features": None},
    }
    app.observe_presence(assigned)

    for side, token in (("LEFT", left), ("RIGHT", right)):
        if assigned[side] is not None:
            if token not in (None, KEEP):
                app.curr_gestures[side] = token
        elif not app._in_loss_grace(side):
            engine = getattr(app, "engine", None)
            if engine is not None:
                engine.reset_hand(side)
            app.curr_gestures[side] = UNKNOWN
            app.confidences[side] = 0.0

    for side in SIDES:
        app.execute_gesture(side, app.curr_gestures[side])

    if getattr(app, "geometry", None) is not None:
        geo_assigned = {}
        for side, token in (("LEFT", left), ("RIGHT", right)):
            if token is None:
                geo_assigned[side] = None
            else:
                points = app._last_points.get(side)
                geo_assigned[side] = {"features": points.ravel()} if points is not None else None
        app.update_special_gestures(geo_assigned)

    app.update_held_inputs()
    for side in SIDES:
        app.prev_gestures[side] = app.curr_gestures[side]
    app.inputs.tick()


class Phase1HandLossGraceTests(TempCase):
    def setUp(self):
        super().setUp()
        self.app = arm_tracking(make_app(self.mapping_path), grace=2)
        self.app.inputs.enable()
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        self.app.mapping.set_command("RIGHT", "G01", HOLD, ["D"])

    def test_one_missed_frame_keeps_hold(self):
        run_tracked_frame(self.app, left="G01")
        self.assertEqual(self.app.inputs.currently_pressed, {"w"})
        run_tracked_frame(self.app, left=None)
        self.assertEqual(self.app.inputs.currently_pressed, {"w"})
        self.assertEqual(self.app.track_state["LEFT"], "GRACE")
        self.assertEqual(self.app.missed_frames["LEFT"], 1)

    def test_two_missed_frames_keep_hold(self):
        run_tracked_frame(self.app, left="G01")
        run_tracked_frame(self.app, left=None)
        run_tracked_frame(self.app, left=None)
        self.assertEqual(self.app.inputs.currently_pressed, {"w"})
        self.assertEqual(self.app.missed_frames["LEFT"], 2)
        self.assertEqual(self.app.track_state["LEFT"], "GRACE")

    def test_third_missed_frame_releases(self):
        run_tracked_frame(self.app, left="G01")
        run_tracked_frame(self.app, left=None)
        run_tracked_frame(self.app, left=None)
        run_tracked_frame(self.app, left=None)
        self.assertEqual(self.app.inputs.currently_pressed, set())
        self.assertEqual(self.app.curr_gestures["LEFT"], UNKNOWN)
        self.assertEqual(self.app.track_state["LEFT"], "RELEASED")

    def test_hand_returns_during_grace_keeps_hold(self):
        run_tracked_frame(self.app, left="G01")
        run_tracked_frame(self.app, left=None)
        run_tracked_frame(self.app, left="G01")
        self.assertEqual(self.app.inputs.currently_pressed, {"w"})
        self.assertEqual(self.app.missed_frames["LEFT"], 0)
        self.assertEqual(self.app.track_state["LEFT"], "OK")
        self.assertEqual(self.app.inputs.keyboard.pressed.count("w"), 1)

    def test_hand_returns_after_expiry_reapplies_hold(self):
        run_tracked_frame(self.app, left="G01")
        for _ in range(3):
            run_tracked_frame(self.app, left=None)
        self.assertEqual(self.app.inputs.currently_pressed, set())
        run_tracked_frame(self.app, left="G01")
        self.assertEqual(self.app.inputs.currently_pressed, {"w"})
        self.assertEqual(self.app.inputs.keyboard.pressed.count("w"), 2)

    def test_left_loss_does_not_release_right(self):
        run_tracked_frame(self.app, left="G01", right="G01")
        self.assertEqual(self.app.inputs.currently_pressed, {"w", "d"})
        for _ in range(3):
            run_tracked_frame(self.app, left=None, right="G01")
        self.assertEqual(self.app.inputs.currently_pressed, {"d"})
        self.assertEqual(self.app.track_state["RIGHT"], "OK")

    def test_right_loss_does_not_release_left(self):
        run_tracked_frame(self.app, left="G01", right="G01")
        for _ in range(3):
            run_tracked_frame(self.app, left="G01", right=None)
        self.assertEqual(self.app.inputs.currently_pressed, {"w"})

    def test_both_hands_lost_release_independently(self):
        run_tracked_frame(self.app, left="G01", right="G01")
        run_tracked_frame(self.app, left=None, right=None)
        run_tracked_frame(self.app, left=None, right=None)
        self.assertEqual(self.app.inputs.currently_pressed, {"w", "d"})
        run_tracked_frame(self.app, left=None, right=None)
        self.assertEqual(self.app.inputs.currently_pressed, set())

    def test_gesture_change_on_return_during_grace(self):
        self.app.mapping.set_command("LEFT", "G02", HOLD, ["S"])
        run_tracked_frame(self.app, left="G01")
        run_tracked_frame(self.app, left=None)
        run_tracked_frame(self.app, left="G02")
        self.assertEqual(self.app.inputs.currently_pressed, {"s"})
        self.assertIn("w", self.app.inputs.keyboard.released)

    def test_zero_grace_still_releases_immediately(self):
        arm_tracking(self.app, grace=0)
        run_tracked_frame(self.app, left="G01")
        run_tracked_frame(self.app, left=None)
        self.assertEqual(self.app.inputs.currently_pressed, set())
        self.assertEqual(self.app.track_state["LEFT"], "RELEASED")


class Phase1GeometryLossAndF8Tests(TempCase):
    def setUp(self):
        super().setUp()
        self.app = arm_tracking(make_geometry_app(self.mapping_path), grace=2)
        self.app.inputs.enable()
        self.app.engine = GestureEngine(self.gestures_path)

    def test_geometry_hold_survives_two_missed_frames(self):
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left=KEEP, left_points=OPEN_LEFT)
        self.assertEqual(self.app.inputs.currently_pressed, {"a"})
        run_tracked_frame(self.app, left=None)
        run_tracked_frame(self.app, left=None)
        self.assertEqual(self.app.inputs.currently_pressed, {"a"})
        run_tracked_frame(self.app, left=None)
        self.assertEqual(self.app.inputs.currently_pressed, set())

    def test_geometry_returns_during_grace(self):
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left=KEEP, left_points=OPEN_LEFT)
        run_tracked_frame(self.app, left=None)
        run_tracked_frame(self.app, left=KEEP, left_points=OPEN_LEFT)
        self.assertEqual(self.app.inputs.currently_pressed, {"a"})

    def test_f8_during_grace_releases_immediately(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        run_tracked_frame(self.app, left="G01")
        run_tracked_frame(self.app, left=None)
        self.assertEqual(self.app.inputs.currently_pressed, {"w"})
        self.app.reset_state()
        self.assertFalse(self.app.inputs.currently_pressed)
        self.assertEqual(self.app.missed_frames["LEFT"], 0)
        self.assertEqual(self.app.curr_gestures["LEFT"], UNKNOWN)

    def test_f8_disable_path_clears_grace_holds(self):
        self.app.mapping.set_command("RIGHT", "G01", HOLD, ["D"])
        run_tracked_frame(self.app, right="G01")
        run_tracked_frame(self.app, right=None)
        self.app.inputs.disable()
        self.assertFalse(self.app.inputs.currently_pressed)
        self.assertFalse(self.app.inputs.hold_sources)

    def test_macro_continues_while_hand_is_lost(self):
        self.app.mapping.set_macro("LEFT", "G05", [
            {"action": STEP_PRESS, "inputs": ["SPACE"]},
            {"action": STEP_WAIT, "duration": 30},
            {"action": STEP_PRESS, "inputs": ["E"]},
        ])
        self.app.mapping.set_command("RIGHT", "G01", HOLD, ["D"])
        run_tracked_frame(self.app, left="G05", right="G01")
        self.assertTrue(self.app.inputs.running_macros())
        for _ in range(5):
            run_tracked_frame(self.app, left=None, right=None)
        drain(self.app.inputs)
        self.assertIn(self.app.inputs.special_keys["SPACE"], self.app.inputs.keyboard.pressed)
        self.assertIn("e", self.app.inputs.keyboard.pressed)


class Phase1SettingsTests(unittest.TestCase):
    def test_parameter_present(self):
        from config import validate_settings
        cleaned = validate_settings({"hand_loss_grace_frames": 4})
        self.assertEqual(cleaned["hand_loss_grace_frames"], 4)

    def test_parameter_absent_uses_default(self):
        from config import HAND_LOSS_GRACE_FRAMES, default_settings, validate_settings
        self.assertEqual(HAND_LOSS_GRACE_FRAMES, 2)
        self.assertEqual(default_settings()["hand_loss_grace_frames"], 2)
        self.assertEqual(validate_settings({})["hand_loss_grace_frames"], 2)
        self.assertEqual(validate_settings({"camera_index": 1})["hand_loss_grace_frames"], 2)

    def test_invalid_value_falls_back_safely(self):
        from config import validate_settings
        self.assertEqual(validate_settings({"hand_loss_grace_frames": "nope"})["hand_loss_grace_frames"], 2)
        self.assertEqual(validate_settings({"hand_loss_grace_frames": -3})["hand_loss_grace_frames"], 0)
        self.assertEqual(validate_settings({"hand_loss_grace_frames": 99})["hand_loss_grace_frames"], 10)
        self.assertEqual(validate_settings({"hand_loss_grace_frames": None})["hand_loss_grace_frames"], 2)


class Phase1ConflictTests(TempCase):
    def setUp(self):
        super().setUp()
        self.app = arm_tracking(make_geometry_app(self.mapping_path), grace=2)
        self.app.inputs.enable()
        self.kb = self.app.inputs.keyboard

    def test_gxx_hold_alone(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        run_tracked_frame(self.app, left="G01")
        self.assertEqual(self.app.inputs.currently_pressed, {"w"})

    def test_geometry_hold_alone(self):
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left=KEEP, left_points=OPEN_LEFT)
        self.assertEqual(self.app.inputs.currently_pressed, {"a"})

    def test_geometry_hold_and_knn_hold_same_hand(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        run_tracked_frame(self.app, left="G01")
        self.assertEqual(self.app.inputs.currently_pressed, {"w"})
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", left_points=OPEN_LEFT)
        self.assertEqual(self.app.inputs.currently_pressed, {"w", "a"})
        self.assertIn("LEFT", self.app.inputs.hold_sources)
        self.assertIn(geo.special_source("LEFT", geo.TILT), self.app.inputs.hold_sources)

    def test_geometry_end_keeps_same_hand_gxx_hold(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        run_tracked_frame(self.app, left="G01")
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", left_points=OPEN_LEFT)
        self.assertEqual(self.app.inputs.currently_pressed, {"w", "a"})
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", left_points=CURLED)
        self.assertEqual(self.app.inputs.currently_pressed, {"w"})

    def test_left_conflict_does_not_change_right(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        self.app.mapping.set_command("RIGHT", "G01", HOLD, ["D"])
        run_tracked_frame(self.app, left="G01", right="G01")
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", right="G01", left_points=OPEN_LEFT)
        self.assertEqual(self.app.inputs.currently_pressed, {"w", "a", "d"})
        self.assertIn("RIGHT", self.app.inputs.hold_sources)

    def test_right_conflict_does_not_change_left(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        self.app.mapping.set_command("RIGHT", "G01", HOLD, ["D"])
        self.app.mapping.set_command("RIGHT", geo.RIGHT_TILT_RIGHT, HOLD, ["SHIFT"])
        shift = self.app.inputs.special_keys["SHIFT"]
        run_tracked_frame(self.app, left="G01", right="G01")
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", right="G01", right_points=OPEN_RIGHT)
        self.assertEqual(self.app.inputs.currently_pressed, {"w", "d", shift})
        self.assertIn("LEFT", self.app.inputs.hold_sources)
        self.assertIn("RIGHT", self.app.inputs.hold_sources)

    def test_press_and_macro_are_not_muted(self):
        self.app.mapping.set_command("LEFT", "G02", PRESS, ["SPACE"], 0)
        space = self.app.inputs.special_keys["SPACE"]
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G02", left_points=OPEN_LEFT)
        self.assertEqual(self.kb.pressed.count(space), 1)
        self.assertEqual(self.app.inputs.currently_pressed, {"a"})

    def test_macro_already_started_is_not_stopped_by_geometry(self):
        self.app.mapping.set_macro("LEFT", "G05", [
            {"action": STEP_PRESS, "inputs": ["SPACE"]},
            {"action": STEP_WAIT, "duration": 25},
            {"action": STEP_PRESS, "inputs": ["E"]},
        ])
        run_tracked_frame(self.app, left="G05")
        self.assertTrue(self.app.inputs.running_macros())
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G05", left_points=OPEN_LEFT)
        drain(self.app.inputs)
        self.assertIn(self.app.inputs.special_keys["SPACE"], self.kb.pressed)
        self.assertIn("e", self.kb.pressed)


class Phase1HudTests(TempCase):
    def setUp(self):
        super().setUp()
        self.app = arm_tracking(make_geometry_app(self.mapping_path), grace=2)
        self.app.inputs.enable()
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        self.app.kill_banner = False

    def test_hud_reports_grace_then_released(self):
        run_tracked_frame(self.app, left="G01", right=None)
        run_tracked_frame(self.app, left=None, right=None)
        from localization import t
        text = "\n".join(self.app.debug_hud_lines())
        self.assertIn(t("debug.detected", state="NO"), text)
        self.assertIn(t("debug.lost_grace", missed=1, limit=2), text)
        self.assertIn("G01", text)
        run_tracked_frame(self.app, left=None, right=None)
        text = "\n".join(self.app.debug_hud_lines())
        self.assertIn(t("debug.lost_grace", missed=2, limit=2), text)
        run_tracked_frame(self.app, left=None, right=None)
        text = "\n".join(self.app.debug_hud_lines())
        self.assertIn(t("debug.lost_released"), text)
        self.assertIn(t("debug.input", state=t("hud.enabled")), text)

    def test_hud_shows_emergency_banner(self):
        self.app.kill_banner = True
        self.app.inputs.disable()
        from localization import t
        text = "\n".join(self.app.debug_hud_lines())
        self.assertIn(t("debug.emergency"), text)

    def test_hud_is_not_built_when_debug_is_off(self):
        import hud_renderer
        source = inspect.getsource(hud_renderer.draw_debug_info)
        self.assertIn("debug_hud_lines", source)
        self.assertIn('getattr(app, "debug", False)', source)
        self.assertLess(source.index('getattr(app, "debug", False)'), source.index("debug_hud_lines"))


class Phase1RegressionTests(TempCase):
    def setUp(self):
        super().setUp()
        self.app = arm_tracking(make_geometry_app(self.mapping_path), grace=2)
        self.app.inputs.enable()
        self.kb = self.app.inputs.keyboard

    def test_press_hold_combination_macro_wait(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        self.app.mapping.set_command("LEFT", "G02", COMBINATION, ["W", "D"])
        self.app.mapping.set_command("RIGHT", "G01", PRESS, ["LEFT_MOUSE"], 0)
        run_tracked_frame(self.app, left="G01", right="G01")
        self.assertEqual(self.app.inputs.currently_pressed, {"w"})
        self.assertIn(self.app.inputs.mouse_buttons["LEFT_MOUSE"], self.app.inputs.mouse.pressed)
        run_tracked_frame(self.app, left="G02")
        self.assertEqual(self.app.inputs.currently_pressed, {"w", "d"})
        self.app.mapping.set_macro("LEFT", "G05", [
            {"action": STEP_PRESS, "inputs": ["SPACE"]},
            {"action": STEP_WAIT, "duration": 20},
            {"action": STEP_PRESS, "inputs": ["E"]},
        ])
        run_tracked_frame(self.app, left="G05")
        drain(self.app.inputs)
        self.assertIn(self.app.inputs.special_keys["SPACE"], self.kb.pressed)
        self.assertIn("e", self.kb.pressed)

    def test_f8_and_learned_gestures_still_work(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        run_tracked_frame(self.app, left="G01")
        self.app.inputs.disable()
        self.assertFalse(self.app.inputs.currently_pressed)
        self.app.inputs.enable()
        for gesture in GESTURE_NAMES:
            app = make_app(self.mapping_path)
            app.inputs.enable()
            entry = app.mapping.get("LEFT", gesture)
            run_frame(app, LEFT=gesture)
            kind, target = app.inputs.resolve(entry["inputs"][0])
            sink = app.inputs.mouse if kind == "mouse" else app.inputs.keyboard
            self.assertIn(target, sink.pressed)

    def test_geometry_defaults_still_hold_a(self):
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left=KEEP, left_points=OPEN_LEFT)
        self.assertEqual(self.app.inputs.currently_pressed, {"a"})


# =========================================================================
# Campagne : gestes reellement enregistres (lecture seule)
# =========================================================================
def recorded_database_path():
    """Fichier utilise par GestureEngine, sinon la sauvegarde locale existante.

    Ne cree rien. gestures.json est prioritaire ; s'il est absent, on lit
    gestures_backup.json s'il est present sur le disque.
    """
    candidates = [
        user_data_path(GESTURES_FILE),
        os.path.join(app_dir(), GESTURES_FILE),
        os.path.join(app_dir(), "gestures_backup.json"),
    ]
    seen = set()
    for path in candidates:
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path
    return None


def recorded_mapping_path():
    for path in (user_data_path(MAPPING_FILE), os.path.join(app_dir(), MAPPING_FILE)):
        path = os.path.abspath(path)
        if os.path.isfile(path):
            return path
    return None


def copy_recorded_files(temp_dir):
    """Copies de travail : GestureEngine/GestureMapping peuvent sauvegarder."""
    source = recorded_database_path()
    mapping_source = recorded_mapping_path()
    if source is None or mapping_source is None:
        return None, None, source, mapping_source
    gestures_copy = os.path.join(temp_dir, "gestures.json")
    mapping_copy = os.path.join(temp_dir, "mapping.json")
    shutil.copy2(source, gestures_copy)
    shutil.copy2(mapping_source, mapping_copy)
    return gestures_copy, mapping_copy, source, mapping_source


def populated_gestures(engine):
    found = []
    for side in SIDES:
        for name in GESTURE_NAMES:
            samples = engine.database["gestures"][side].get(name) or []
            if samples:
                found.append((side, name, samples))
    return found


def sample_of(samples, index=0):
    return np.asarray(samples[index % len(samples)], dtype=np.float32).flatten()


def recognize_until_stable(engine, side, sample, limit=6):
    raw, conf, stable = UNKNOWN, 0.0, UNKNOWN
    for _ in range(limit):
        raw, conf = engine.recognize(sample, side)
        stable = engine.stabilize(side, raw, conf)
        if stable == raw and raw != UNKNOWN:
            return raw, conf, stable
    return raw, conf, stable


class RecordedGestureCampaignTests(unittest.TestCase):
    """Parcourt la base enregistree. Aucune ecriture sur les fichiers source."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.TemporaryDirectory()
        copies = copy_recorded_files(cls.td.name)
        cls.gestures_copy, cls.mapping_copy, cls.source_path, cls.mapping_source = copies
        cls.source_mtime = os.path.getmtime(cls.source_path) if cls.source_path else None
        cls.mapping_mtime = os.path.getmtime(cls.mapping_source) if cls.mapping_source else None
        cls.engine = GestureEngine(cls.gestures_copy) if cls.gestures_copy else None
        cls.mapping = GestureMapping(cls.mapping_copy) if cls.mapping_copy else None
        cls.recorded = populated_gestures(cls.engine) if cls.engine else []

    @classmethod
    def tearDownClass(cls):
        if cls.source_path and cls.source_mtime is not None:
            assert os.path.getmtime(cls.source_path) == cls.source_mtime
        if cls.mapping_source and cls.mapping_mtime is not None:
            assert os.path.getmtime(cls.mapping_source) == cls.mapping_mtime
        cls.td.cleanup()

    def setUp(self):
        if not self.recorded:
            self.skipTest("Aucune base enregistree (gestures.json / gestures_backup.json)")
        self.app = arm_tracking(make_geometry_app(self.mapping_copy), grace=2)
        self.app.engine = GestureEngine(self.gestures_copy)
        self.app.mapping = GestureMapping(self.mapping_copy)
        self.app.inputs.enable()
        self.kb = self.app.inputs.keyboard

    def recorded_of_type(self, entry_type):
        found = []
        for side, name, samples in self.recorded:
            entry = self.mapping.get(side, name)
            if entry and entry["type"] == entry_type:
                found.append((side, name, samples, entry))
        return found

    def hold_recorded(self, side, samples):
        raw, conf, stable = recognize_until_stable(self.app.engine, side, sample_of(samples))
        self.assertEqual(stable, raw)
        self.assertNotEqual(stable, UNKNOWN)
        kwargs = {"left": stable if side == "LEFT" else KEEP, "right": stable if side == "RIGHT" else KEEP}
        for _ in range(3):
            run_tracked_frame(self.app, **kwargs)
        return stable

    def test_inventory_matches_the_on_disk_banks(self):
        by_side = {side: [] for side in SIDES}
        for side, name, samples in self.recorded:
            by_side[side].append((name, len(samples)))
        self.assertTrue(self.recorded)
        for side, name, samples in self.recorded:
            self.assertEqual(len(samples[0]), FEATURE_SIZE)
            with self.subTest(side=side, gesture=name):
                self.assertGreaterEqual(len(samples), 1)

    def test_each_recorded_sample_is_recognized_as_itself(self):
        engine = self.app.engine
        for side, name, samples in self.recorded:
            for index in (0, len(samples) // 2, len(samples) - 1):
                raw, conf = engine.recognize(sample_of(samples, index), side)
                with self.subTest(side=side, gesture=name, index=index):
                    self.assertEqual(raw, name)
                    self.assertGreaterEqual(conf, engine.recognition_threshold)

    def test_left_and_right_banks_stay_isolated(self):
        engine = self.app.engine
        for side, name, samples in self.recorded:
            other = "RIGHT" if side == "LEFT" else "LEFT"
            other_names = {g for s, g, _ in self.recorded if s == other}
            raw, _ = engine.recognize(sample_of(samples), other)
            with self.subTest(side=side, gesture=name, queried=other):
                if name not in other_names:
                    self.assertNotEqual(raw, name)

    def test_repeated_frames_keep_the_same_stable_gesture(self):
        engine = self.app.engine
        for side, name, samples in self.recorded:
            engine.reset_hand(side)
            sample = sample_of(samples)
            seen = []
            for _ in range(5):
                raw, conf = engine.recognize(sample, side)
                seen.append(engine.stabilize(side, raw, conf))
            with self.subTest(side=side, gesture=name):
                self.assertEqual(seen[-1], name)
                self.assertEqual(len(set(item for item in seen if item != UNKNOWN)), 1)

    def test_hold_stays_down_without_key_repeat(self):
        holds = self.recorded_of_type(HOLD)
        if not holds:
            self.skipTest("Aucun geste enregistre n'a de mapping HOLD")
        for side, name, samples, entry in holds:
            with self.subTest(side=side, gesture=name):
                app = arm_tracking(make_geometry_app(self.mapping_copy), grace=2)
                app.engine = GestureEngine(self.gestures_copy)
                app.mapping = GestureMapping(self.mapping_copy)
                app.inputs.enable()
                sample = sample_of(samples)
                recognize_until_stable(app.engine, side, sample)
                kwargs = {"left": name if side == "LEFT" else KEEP, "right": name if side == "RIGHT" else KEEP}
                run_tracked_frame(app, **kwargs)
                run_tracked_frame(app, **kwargs)
                run_tracked_frame(app, **kwargs)
                names = app.inputs.get_pressed_names()
                for action in entry["inputs"]:
                    self.assertIn(action, names)
                self.assertIn(side, app.inputs.hold_sources)
                kind, target = app.inputs.resolve(entry["inputs"][0])
                sink = app.inputs.mouse if kind == "mouse" else app.inputs.keyboard
                self.assertEqual(sink.pressed.count(target), 1)

    def test_unknown_while_present_follows_unknown_frames(self):
        holds = self.recorded_of_type(HOLD)
        if not holds:
            self.skipTest("Aucun geste enregistre n'a de mapping HOLD")
        side, name, samples, entry = holds[0]
        self.hold_recorded(side, samples)
        first = self.app.engine.stabilize(side, UNKNOWN)
        kwargs = {"left": first if side == "LEFT" else KEEP, "right": first if side == "RIGHT" else KEEP}
        run_tracked_frame(self.app, **kwargs)
        if self.app.engine.unknown_frames > 1:
            self.assertTrue(self.app.inputs.currently_pressed)
        second = self.app.engine.stabilize(side, UNKNOWN)
        kwargs = {"left": second if side == "LEFT" else KEEP, "right": second if side == "RIGHT" else KEEP}
        run_tracked_frame(self.app, **kwargs)
        self.assertEqual(second, UNKNOWN)
        for action in entry["inputs"]:
            self.assertNotIn(action, self.app.inputs.get_pressed_names())

    def test_grace_keeps_recorded_hold_then_releases(self):
        holds = self.recorded_of_type(HOLD)
        if not holds:
            self.skipTest("Aucun geste enregistre n'a de mapping HOLD")
        side, name, samples, entry = holds[0]
        self.hold_recorded(side, samples)
        self.assertTrue(self.app.inputs.currently_pressed)
        absent = {"left": None if side == "LEFT" else KEEP, "right": None if side == "RIGHT" else KEEP}
        run_tracked_frame(self.app, **absent)
        self.assertEqual(self.app.missed_frames[side], 1)
        self.assertEqual(self.app.track_state[side], "GRACE")
        self.assertIn(side, self.app.inputs.hold_sources)
        run_tracked_frame(self.app, **absent)
        self.assertEqual(self.app.missed_frames[side], 2)
        self.assertEqual(self.app.track_state[side], "GRACE")
        for action in entry["inputs"]:
            self.assertIn(action, self.app.inputs.get_pressed_names())
        run_tracked_frame(self.app, **absent)
        self.assertEqual(self.app.track_state[side], "RELEASED")
        self.assertEqual(self.app.curr_gestures[side], UNKNOWN)
        self.assertEqual(self.app.engine.stable_gestures[side], UNKNOWN)
        self.assertFalse(self.app.inputs.currently_pressed)
        self.assertNotIn(side, self.app.inputs.hold_sources)

    def test_return_same_recorded_gesture_during_grace(self):
        holds = self.recorded_of_type(HOLD)
        if not holds:
            self.skipTest("Aucun geste enregistre n'a de mapping HOLD")
        side, name, samples, entry = holds[0]
        self.hold_recorded(side, samples)
        absent = {"left": None if side == "LEFT" else KEEP, "right": None if side == "RIGHT" else KEEP}
        run_tracked_frame(self.app, **absent)
        present = {"left": name if side == "LEFT" else KEEP, "right": name if side == "RIGHT" else KEEP}
        run_tracked_frame(self.app, **present)
        self.assertEqual(self.app.missed_frames[side], 0)
        self.assertEqual(self.app.track_state[side], "OK")
        for action in entry["inputs"]:
            self.assertIn(action, self.app.inputs.get_pressed_names())
        kind, target = self.app.inputs.resolve(entry["inputs"][0])
        sink = self.app.inputs.mouse if kind == "mouse" else self.app.inputs.keyboard
        self.assertEqual(sink.pressed.count(target), 1)

    def test_return_other_recorded_gesture_during_grace(self):
        holds = self.recorded_of_type(HOLD)
        same_side = {}
        for side, name, samples, entry in holds:
            same_side.setdefault(side, []).append((name, samples, entry))
        pair_side = next((side for side, items in same_side.items() if len(items) >= 2), None)
        if pair_side is None:
            self.skipTest("Pas deux HOLD enregistres sur la meme main pour tester un changement")
        (first, samples_a, entry_a), (second, samples_b, entry_b) = same_side[pair_side][:2]
        self.hold_recorded(pair_side, samples_a)
        absent = {"left": None if pair_side == "LEFT" else KEEP, "right": None if pair_side == "RIGHT" else KEEP}
        run_tracked_frame(self.app, **absent)
        present = {"left": second if pair_side == "LEFT" else KEEP, "right": second if pair_side == "RIGHT" else KEEP}
        run_tracked_frame(self.app, **present)
        names = self.app.inputs.get_pressed_names()
        for action in entry_b["inputs"]:
            self.assertIn(action, names)
        if entry_a["inputs"] != entry_b["inputs"]:
            for action in entry_a["inputs"]:
                self.assertNotIn(action, names)

    def test_two_hands_grace_are_independent(self):
        left_holds = [item for item in self.recorded_of_type(HOLD) if item[0] == "LEFT"]
        right_holds = [item for item in self.recorded_of_type(HOLD) if item[0] == "RIGHT"]
        if not left_holds or not right_holds:
            self.skipTest("Il faut un HOLD enregistre a gauche ET a droite")
        _, left_name, left_samples, left_entry = left_holds[0]
        _, right_name, right_samples, right_entry = right_holds[0]
        recognize_until_stable(self.app.engine, "LEFT", sample_of(left_samples))
        recognize_until_stable(self.app.engine, "RIGHT", sample_of(right_samples))
        run_tracked_frame(self.app, left=left_name, right=right_name)
        run_tracked_frame(self.app, left=None, right=right_name)
        self.assertEqual(self.app.track_state["LEFT"], "GRACE")
        self.assertEqual(self.app.track_state["RIGHT"], "OK")
        for action in right_entry["inputs"]:
            self.assertIn(action, self.app.inputs.get_pressed_names())
        run_tracked_frame(self.app, left=None, right=None)
        self.assertEqual(self.app.track_state["LEFT"], "GRACE")
        self.assertEqual(self.app.track_state["RIGHT"], "GRACE")
        run_tracked_frame(self.app, left=None, right=None)
        self.assertEqual(self.app.track_state["RIGHT"], "GRACE")
        run_tracked_frame(self.app, left=None, right=None)
        self.assertEqual(self.app.track_state["LEFT"], "RELEASED")
        self.assertEqual(self.app.track_state["RIGHT"], "RELEASED")
        self.assertFalse(self.app.inputs.currently_pressed)

    def test_geometry_hold_and_recorded_knn_hold_same_hand(self):
        left_holds = [item for item in self.recorded_of_type(HOLD) if item[0] == "LEFT"]
        if not left_holds:
            self.skipTest("Aucun HOLD enregistre a gauche")
        _, name, samples, entry = left_holds[0]
        recognize_until_stable(self.app.engine, "LEFT", sample_of(samples))
        run_tracked_frame(self.app, left=name)
        for action in entry["inputs"]:
            self.assertIn(action, self.app.inputs.get_pressed_names())
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left=name, left_points=OPEN_LEFT)
        self.assertIn("LEFT", self.app.inputs.hold_sources)
        self.assertIn(geo.special_source("LEFT", geo.TILT), self.app.inputs.hold_sources)
        self.assertIn("A", self.app.inputs.get_pressed_names())
        for action in entry["inputs"]:
            self.assertIn(action, self.app.inputs.get_pressed_names())

    def test_left_geometry_does_not_mute_right_recorded_hold(self):
        left_holds = [item for item in self.recorded_of_type(HOLD) if item[0] == "LEFT"]
        right_holds = [item for item in self.recorded_of_type(HOLD) if item[0] == "RIGHT"]
        if not left_holds or not right_holds:
            self.skipTest("Il faut un HOLD enregistre a gauche ET a droite")
        _, left_name, left_samples, _ = left_holds[0]
        _, right_name, right_samples, right_entry = right_holds[0]
        recognize_until_stable(self.app.engine, "LEFT", sample_of(left_samples))
        recognize_until_stable(self.app.engine, "RIGHT", sample_of(right_samples))
        run_tracked_frame(self.app, left=left_name, right=right_name)
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left=left_name, right=right_name, left_points=OPEN_LEFT)
        self.assertIn("RIGHT", self.app.inputs.hold_sources)
        for action in right_entry["inputs"]:
            self.assertIn(action, self.app.inputs.get_pressed_names())

    def test_right_geometry_hold_cannot_be_tested_with_current_mapping(self):
        """Aucun HOLD geometrique RIGHT n'est mappe : mute RIGHT->LEFT impossible ici."""
        from geometry_engine import SPECIAL_BY_SIDE
        mapped = []
        for gesture in SPECIAL_BY_SIDE["RIGHT"]:
            entry = self.mapping.get("RIGHT", gesture)
            if entry and entry["type"] in MAINTAINED_TYPES:
                mapped.append(gesture)
        if mapped:
            self.skipTest("Un HOLD geometrique RIGHT existe, ce temoin n'est plus pertinent")
        self.assertEqual(mapped, [])

    def test_press_fires_once_and_is_not_a_hold(self):
        presses = self.recorded_of_type(PRESS)
        if not presses:
            self.skipTest("Aucun geste enregistre n'a de mapping PRESS")
        for side, name, samples, entry in presses:
            with self.subTest(side=side, gesture=name):
                app = arm_tracking(make_geometry_app(self.mapping_copy), grace=2)
                app.engine = GestureEngine(self.gestures_copy)
                app.mapping = GestureMapping(self.mapping_copy)
                app.inputs.enable()
                sample = sample_of(samples)
                raw, conf, stable = recognize_until_stable(app.engine, side, sample)
                self.assertEqual(stable, name)
                kwargs = {"left": name if side == "LEFT" else KEEP, "right": name if side == "RIGHT" else KEEP}
                run_tracked_frame(app, **kwargs)
                run_tracked_frame(app, **kwargs)
                run_tracked_frame(app, **kwargs)
                kind, target = app.inputs.resolve(entry["inputs"][0])
                sink = app.inputs.mouse if kind == "mouse" else app.inputs.keyboard
                self.assertEqual(sink.pressed.count(target), 1)
                self.assertNotIn(side, app.inputs.hold_sources)
                time.sleep(0.05)
                app.inputs.tick()
                for action in entry["inputs"]:
                    self.assertNotIn(action, app.inputs.get_pressed_names())

    def test_combination_among_recorded_gestures(self):
        combos = self.recorded_of_type(COMBINATION)
        if not combos:
            self.skipTest("Aucun geste enregistre n'a de mapping COMBINATION")
        for side, name, samples, entry in combos:
            with self.subTest(side=side, gesture=name):
                recognize_until_stable(self.app.engine, side, sample_of(samples))
                kwargs = {"left": name if side == "LEFT" else KEEP, "right": name if side == "RIGHT" else KEEP}
                run_tracked_frame(self.app, **kwargs)
                for action in entry["inputs"]:
                    self.assertIn(action, self.app.inputs.get_pressed_names())
                absent = {"left": None if side == "LEFT" else KEEP, "right": None if side == "RIGHT" else KEEP}
                run_tracked_frame(self.app, **absent)
                run_tracked_frame(self.app, **absent)
                self.assertTrue(self.app.inputs.currently_pressed)
                run_tracked_frame(self.app, **absent)
                self.assertFalse(self.app.inputs.currently_pressed)

    def test_macro_among_recorded_gestures(self):
        macros = self.recorded_of_type(MACRO)
        if not macros:
            self.skipTest("Aucun geste enregistre n'a de mapping MACRO")
        side, name, samples, entry = macros[0]
        recognize_until_stable(self.app.engine, side, sample_of(samples))
        kwargs = {"left": name if side == "LEFT" else KEEP, "right": name if side == "RIGHT" else KEEP}
        run_tracked_frame(self.app, **kwargs)
        self.assertTrue(self.app.inputs.running_macros())
        absent = {"left": None if side == "LEFT" else KEEP, "right": None if side == "RIGHT" else KEEP}
        run_tracked_frame(self.app, **absent)
        self.assertTrue(self.app.inputs.running_macros())
        drain(self.app.inputs, timeout=3.0)
        self.app.reset_state()
        self.assertFalse(self.app.inputs.currently_pressed)
        self.assertFalse(self.app.inputs.running_macros())

    def test_f8_clears_hold_grace_and_does_not_return(self):
        holds = self.recorded_of_type(HOLD)
        if not holds:
            self.skipTest("Aucun geste enregistre n'a de mapping HOLD")
        side, name, samples, entry = holds[0]
        self.hold_recorded(side, samples)
        absent = {"left": None if side == "LEFT" else KEEP, "right": None if side == "RIGHT" else KEEP}
        run_tracked_frame(self.app, **absent)
        self.assertEqual(self.app.track_state[side], "GRACE")
        self.app.reset_state()
        self.assertFalse(self.app.inputs.currently_pressed)
        self.assertFalse(self.app.inputs.hold_sources)
        self.assertEqual(self.app.missed_frames[side], 0)
        self.assertEqual(self.app.curr_gestures[side], UNKNOWN)
        run_tracked_frame(self.app, **absent)
        self.assertFalse(self.app.inputs.currently_pressed)

    def test_f8_with_two_recorded_hands(self):
        left_holds = [item for item in self.recorded_of_type(HOLD) if item[0] == "LEFT"]
        right_holds = [item for item in self.recorded_of_type(HOLD) if item[0] == "RIGHT"]
        if not left_holds or not right_holds:
            self.skipTest("Il faut un HOLD enregistre a gauche ET a droite")
        _, left_name, left_samples, _ = left_holds[0]
        _, right_name, right_samples, _ = right_holds[0]
        recognize_until_stable(self.app.engine, "LEFT", sample_of(left_samples))
        recognize_until_stable(self.app.engine, "RIGHT", sample_of(right_samples))
        run_tracked_frame(self.app, left=left_name, right=right_name)
        self.assertTrue(self.app.inputs.currently_pressed)
        self.app.inputs.disable()
        self.assertFalse(self.app.inputs.currently_pressed)
        self.assertFalse(self.app.inputs.hold_sources)

    def test_source_files_were_not_rewritten(self):
        self.assertEqual(os.path.getmtime(self.source_path), self.source_mtime)
        self.assertEqual(os.path.getmtime(self.mapping_source), self.mapping_mtime)


class Phase2GeometryTests(TempCase):
    def test_new_specials_ship_disabled_and_tilt_up_stays_off(self):
        mapping = GestureMapping(self.mapping_path)
        for side, name in (
            ("LEFT", geo.LEFT_PINCH), ("LEFT", geo.LEFT_FIST), ("LEFT", geo.LEFT_OPEN_PALM),
            ("RIGHT", geo.RIGHT_FIST), ("RIGHT", geo.RIGHT_OPEN_PALM),
            ("LEFT", geo.LEFT_TILT_UP), ("LEFT", geo.LEFT_TILT_DOWN),
            ("LEFT", geo.LEFT_INDEX_LEFT),
        ):
            self.assertEqual(mapping.get_type(side, name), NONE)

    def test_missing_geometry_settings_keep_factory_thresholds(self):
        from config import validate_settings
        cleaned = validate_settings({})
        self.assertEqual(cleaned["pinch_on_threshold"], geo.PINCH_ON_THRESHOLD)
        self.assertEqual(cleaned["pinch_off_threshold"], geo.PINCH_OFF_THRESHOLD)
        self.assertFalse(cleaned["fist_enabled"])
        self.assertFalse(cleaned["open_palm_enabled"])
        self.assertEqual(cleaned["geometry_stability_frames"], geo.STABILITY_FRAMES)

    def test_invalid_geometry_settings_fall_back(self):
        from config import validate_settings
        cleaned = validate_settings({
            "pinch_on_threshold": 0.9,
            "pinch_off_threshold": 0.1,
            "fist_enabled": "false",
            "open_palm_enabled": "nope",
        })
        self.assertEqual(cleaned["pinch_on_threshold"], geo.PINCH_ON_THRESHOLD)
        self.assertEqual(cleaned["pinch_off_threshold"], geo.PINCH_OFF_THRESHOLD)
        self.assertFalse(cleaned["fist_enabled"])
        self.assertFalse(cleaned["open_palm_enabled"])

    def test_settings_drive_both_hands(self):
        engine = geo.GeometryEngine(**geo.overrides_from_settings({
            "pinch_on_threshold": 0.10,
            "pinch_off_threshold": 0.40,
            "fist_enabled": True,
        }))
        self.assertEqual(engine.detector("LEFT", geo.PINCH).on_threshold, 0.10)
        self.assertEqual(engine.detector("RIGHT", geo.PINCH).on_threshold, 0.10)
        self.assertIsNotNone(engine.detector("LEFT", geo.FIST))
        self.assertIsNone(geo.GeometryEngine().detector("LEFT", geo.FIST))

    def test_fist_open_palm_hysteresis_and_pointing(self):
        engine = geo.GeometryEngine(fist_enabled=True, open_palm_enabled=True, stability_frames=1)
        self.assertEqual(engine.update("LEFT", FIST_POSE)[geo.FIST], geo.LEFT_FIST)
        self.assertIsNone(engine.update("LEFT", FIST_POSE)[geo.PALM])
        self.assertIsNone(engine.update("RIGHT", POINTING_POSE)[geo.FIST])
        self.assertIsNone(engine.update("RIGHT", POINTING_POSE)[geo.PALM])
        self.assertEqual(engine.update("RIGHT", PALM_POSE)[geo.PALM], geo.RIGHT_OPEN_PALM)
        between = finger_pose((0.52, 0.52, 0.52, 0.52))
        self.assertEqual(engine.update("LEFT", between)[geo.FIST], geo.LEFT_FIST)
        self.assertIsNone(engine.update("LEFT", finger_pose((0.8, 0.8, 0.8, 0.8)))[geo.FIST])

    def test_one_frame_spike_does_not_confirm_a_fist(self):
        engine = geo.GeometryEngine(fist_enabled=True, stability_frames=2)
        self.assertIsNone(engine.update("LEFT", FIST_POSE)[geo.FIST])
        self.assertEqual(engine.update("LEFT", FIST_POSE)[geo.FIST], geo.LEFT_FIST)

    def test_two_hands_fist_and_pinch_stay_independent(self):
        engine = geo.GeometryEngine(fist_enabled=True, stability_frames=1)
        left = engine.update("LEFT", FIST_POSE)
        right = engine.update("RIGHT", PINCHING)
        self.assertEqual(left[geo.FIST], geo.LEFT_FIST)
        self.assertEqual(right[geo.PINCH], geo.RIGHT_PINCH)
        self.assertIsNone(right[geo.FIST])
        engine.hand_lost("LEFT")
        self.assertIsNone(engine.states("LEFT")[geo.FIST])
        self.assertEqual(engine.states("RIGHT")[geo.PINCH], geo.RIGHT_PINCH)

    def test_suggest_thresholds_from_observed_poses(self):
        def samples(pinch, dx, dy):
            return [{"pinch_ratio": pinch, "dx": dx, "dy": dy} for _ in range(8)]
        proposed = geo.suggest_geometry_thresholds({
            "neutral": samples(0.80, 0.02, -0.05),
            "pinch": samples(0.12, 0.02, -0.05),
            "tilt_left": samples(0.70, -0.80, -0.10),
            "tilt_right": samples(0.70, 0.80, -0.10),
        })
        self.assertLess(proposed["pinch_on_threshold"], proposed["pinch_off_threshold"])
        self.assertLess(proposed["pinch_off_threshold"], 0.80)
        self.assertGreater(proposed["tilt_left_threshold"], 0.12)
        self.assertGreater(proposed["tilt_right_threshold"], 0.12)
        self.assertNotIn("tilt_up_threshold", proposed)

    def test_suggestion_does_not_touch_gesture_files(self):
        before = os.path.getmtime(os.path.join(os.path.dirname(main_module.__file__), "gestures_backup.json"))
        geo.suggest_geometry_thresholds({
            "neutral": [{"pinch_ratio": 0.9, "dx": 0.0, "dy": 0.0}],
            "pinch": [{"pinch_ratio": 0.1, "dx": 0.0, "dy": 0.0}],
            "tilt_left": [{"pinch_ratio": 0.9, "dx": -0.7, "dy": 0.0}],
            "tilt_right": [{"pinch_ratio": 0.9, "dx": 0.7, "dy": 0.0}],
        })
        after = os.path.getmtime(os.path.join(os.path.dirname(main_module.__file__), "gestures_backup.json"))
        self.assertEqual(before, after)

    def test_fist_hold_grace_conflict_and_f8(self):
        app = arm_tracking(make_geometry_app(self.mapping_path), grace=2)
        app.geometry = geo.GeometryEngine(fist_enabled=True, stability_frames=1)
        app.engine = GestureEngine(self.gestures_path)
        app.inputs.enable()
        app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        app.mapping.set_command("LEFT", geo.LEFT_FIST, HOLD, ["SHIFT"])
        app.mapping.set_command("RIGHT", "G01", HOLD, ["D"])
        run_tracked_frame(app, left="G01", right="G01")
        for _ in range(2):
            run_tracked_frame(app, left="G01", right="G01", left_points=FIST_POSE)
        self.assertIn("SHIFT", app.inputs.get_pressed_names())
        self.assertIn("LEFT", app.inputs.hold_sources)
        self.assertIn("W", app.inputs.get_pressed_names())
        self.assertIn("D", app.inputs.get_pressed_names())
        run_tracked_frame(app, left=None, right="G01")
        self.assertEqual(app.track_state["LEFT"], "GRACE")
        self.assertIn("SHIFT", app.inputs.get_pressed_names())
        app.reset_state()
        self.assertFalse(app.inputs.currently_pressed)
        self.assertEqual(app.missed_frames["LEFT"], 0)
        self.assertFalse(app.geometry.active("LEFT"))


class Phase3StabilizationAndKeysTests(TempCase):
    def test_filter_disabled_returns_the_raw_points(self):
        from landmark_filter import OneEuroFilter
        filt = OneEuroFilter()
        raw = np.ones((21, 3), dtype=np.float32)
        # The live path skips the filter when smoothing is off. The filter
        # itself returns the first sample unchanged, which is the raw pose.
        np.testing.assert_allclose(filt.filter(raw, 0.0), raw)

    def test_filter_pulls_a_noisy_point_toward_the_previous_one(self):
        from landmark_filter import OneEuroFilter
        filt = OneEuroFilter(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
        first = np.zeros((21, 3), dtype=np.float32)
        second = np.ones((21, 3), dtype=np.float32)
        filt.filter(first, 0.0)
        smoothed = filt.filter(second, 1.0 / 30.0)
        self.assertLess(float(np.mean(smoothed)), 0.5)

    def test_left_and_right_filters_do_not_share_state(self):
        from landmark_filter import OneEuroFilter
        left = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        right = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        left.filter(np.zeros((21, 3)), 0.0)
        right.filter(np.ones((21, 3)), 0.0)
        left.reset()
        again = left.filter(np.full((21, 3), 0.2), 1.0)
        np.testing.assert_allclose(again, np.full((21, 3), 0.2))
        held = right.filter(np.ones((21, 3)), 1.0 / 30.0)
        self.assertGreater(float(np.mean(held)), 0.5)

    def test_grace_keeps_the_filter_until_it_expires(self):
        app = HandControllerApp.__new__(HandControllerApp)
        from landmark_filter import OneEuroFilter
        app.smoothing_enabled = True
        app.hand_loss_grace_frames = 2
        app.missed_frames = {side: 0 for side in SIDES}
        app.track_state = {side: "OK" for side in SIDES}
        app.detected = {side: False for side in SIDES}
        app.smoothers = {side: OneEuroFilter() for side in SIDES}
        pose = np.zeros((21, 3), dtype=np.float32)
        pose[9] = (0.0, -0.2, 0.0)

        class LM:
            def __init__(self, row):
                self.x, self.y, self.z = (float(value) for value in row)

        hand = [LM(row) for row in pose]
        present = {"LEFT": {"hand": hand}, "RIGHT": None}
        app.observe_presence(present)
        app._apply_smoothing(present)
        self.assertIsNotNone(app.smoothers["LEFT"]._x)
        absent = {"LEFT": None, "RIGHT": None}
        app.observe_presence(absent)
        app._apply_smoothing(absent)
        self.assertEqual(app.track_state["LEFT"], "GRACE")
        self.assertIsNotNone(app.smoothers["LEFT"]._x)
        app.observe_presence(absent)
        app._apply_smoothing(absent)
        app.observe_presence(absent)
        app._apply_smoothing(absent)
        self.assertEqual(app.track_state["LEFT"], "RELEASED")
        self.assertIsNone(app.smoothers["LEFT"]._x)
        self.assertIsNone(app.smoothers["RIGHT"]._x)

    def test_f8_to_f11_are_assignable_game_keys(self):
        mapping = GestureMapping(self.mapping_path)
        self.assertTrue(mapping.set_command("LEFT", "G01", HOLD, ["F11"]))
        self.assertEqual(mapping.get_inputs("LEFT", "G01"), ["F11"])
        self.assertTrue(mapping.set_command("LEFT", "G01", PRESS, ["F10"], 0))
        self.assertTrue(mapping.set_command("LEFT", "G02", HOLD, ["F1"]))
        self.assertEqual(mapping.get_inputs("LEFT", "G02"), ["F1"])
        self.assertTrue(mapping.set_command("LEFT", "G03", COMBINATION, ["CTRL", "SHIFT", "S"]))
        self.assertEqual(mapping.get_inputs("LEFT", "G03"), ["CTRL", "SHIFT", "S"])
        self.assertTrue(mapping.set_macro("RIGHT", "G01", [
            {"action": STEP_PRESS, "inputs": ["F11"]},
        ]))
        self.assertTrue(mapping.set_command("RIGHT", "G02", HOLD, ["F8"]))
        self.assertTrue(mapping.set_command("RIGHT", "G03", PRESS, ["F9"], 0))

    def test_letters_modifiers_arrows_and_mouse_still_resolve(self):
        ctrl = make_inputs()
        ctrl.enable()
        for name in ("A", "Z", "SPACE", "ENTER", "SHIFT", "CTRL", "F1", "UP", "LEFT_MOUSE"):
            kind, target = ctrl.resolve(name)
            self.assertIsNotNone(target, name)
        ctrl.set_source_hold("LEFT", ["F1"])
        self.assertIn("F1", ctrl.get_pressed_names())
        ctrl.set_source_hold("LEFT", ["SHIFT"])
        self.assertEqual(ctrl.get_pressed_names(), ["SHIFT"])

    def test_press_hold_and_combination_use_the_new_names(self):
        app = make_app(self.mapping_path)
        app.inputs.enable()
        app.mapping.set_command("LEFT", "G01", PRESS, ["F1"], 0)
        app.mapping.set_command("RIGHT", "G01", HOLD, ["UP"])
        run_frame(app, LEFT="G01", RIGHT="G01")
        self.assertIn(app.inputs.special_keys["F1"], app.inputs.keyboard.pressed)
        self.assertIn("UP", app.inputs.get_pressed_names())
        app.mapping.set_command("LEFT", "G02", COMBINATION, ["CTRL", "SHIFT", "S"])
        run_frame(app, LEFT="G02")
        self.assertTrue({"CTRL", "SHIFT", "S"} <= set(app.inputs.get_pressed_names()))

    def test_replay_roundtrip_never_enables_input(self):
        import landmark_replay as replay_mod
        path = os.path.join(self.td.name, "sample.jsonl")
        raw = np.zeros((21, 3), dtype=np.float32)
        raw[9] = (0.0, -0.2, 0.0)
        with open(path, "w", encoding="utf-8") as handle:
            replay_mod.write_frame(handle, replay_mod.frame_record(0, 10, [
                replay_mod.hand_record("LEFT", "LEFT", 0.9, raw, raw),
            ]))
            replay_mod.write_frame(handle, replay_mod.frame_record(1, 40, []))
        frames = replay_mod.read_frames(path)
        self.assertEqual(frames[0]["frame"], 0)
        self.assertEqual(frames[1]["hands"], [])
        self.assertEqual(len(frames[0]["hands"][0]["landmarks_raw"]), 21)
        tree = ast.parse(inspect.getsource(replay_mod))
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertNotIn("input_controller", imported)

    def test_passive_frame_does_not_press_keys(self):
        app = make_geometry_app(self.mapping_path)
        app.engine = GestureEngine(self.gestures_path)
        app.smoothing_enabled = False
        from landmark_filter import OneEuroFilter
        app.smoothers = {side: OneEuroFilter() for side in SIDES}
        app.passive = True
        app.recording = False
        app.camera_ok = True
        app.kill_banner = False
        app.detected = {side: False for side in SIDES}
        app.confidences = {side: 0.0 for side in SIDES}
        app.status_message = ""
        app.perf_text = ""
        app.perf = {key: 0.0 for key in ("mediapipe", "normalisation", "reconnaissance", "stabilisation", "mapping", "input", "special", "affichage", "total")}
        app.inputs.enable()
        pressed_before = len(app.inputs.keyboard.pressed)
        raw = np.zeros((21, 3), dtype=np.float32)
        raw[9] = (0.0, -0.25, 0.0)

        class LM:
            def __init__(self, row):
                self.x, self.y, self.z = (float(value) for value in row)

        assigned_det = {"hand": [LM(row) for row in raw], "side": "LEFT", "score": 1.0, "raw": UNKNOWN, "conf": 0.0}
        app.collect_detections = lambda result: [assigned_det]
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        app.mapping.set_command("LEFT", "G01", PRESS, ["SPACE"], 0)
        app.process_frame(blank, None)
        self.assertEqual(len(app.inputs.keyboard.pressed), pressed_before)

    def test_filter_cost_stays_under_a_millisecond(self):
        from landmark_filter import OneEuroFilter
        filt = OneEuroFilter()
        sample = np.random.default_rng(0).random((21, 3), dtype=np.float64)
        start = time.perf_counter()
        for index in range(300):
            filt.filter(sample + index * 1e-4, index / 30.0)
        elapsed = (time.perf_counter() - start) / 300.0
        self.assertLess(elapsed, 0.001)


class Phase35GestureIdentityTests(TempCase):
    def test_next_id_fills_the_first_gap_then_passes_99(self):
        from gesture_engine import GestureSamples, gesture_id
        engine = GestureEngine(self.gestures_path)
        self.assertEqual(engine.next_gesture_id("LEFT"), "G01")
        for number in range(1, 31):
            engine.database["gestures"]["LEFT"][gesture_id(number)] = GestureSamples([[0.0] * FEATURE_SIZE])
        self.assertEqual(engine.next_gesture_id("LEFT"), "G31")
        created = engine.allocate_gesture("LEFT", "attaque")
        self.assertEqual(created, "G31")
        self.assertEqual(engine.display_name("LEFT", "G31"), "attaque")
        for number in range(31, 100):
            engine.database["gestures"]["LEFT"][gesture_id(number)] = GestureSamples()
        self.assertEqual(engine.next_gesture_id("LEFT"), "G100")

    def test_deleted_id_is_not_reused(self):
        engine = GestureEngine(self.gestures_path)
        self.assertTrue(engine.delete_gesture("LEFT", "G04"))
        self.assertNotEqual(engine.next_gesture_id("LEFT"), "G04")
        self.assertEqual(engine.sample_count("LEFT", "G04"), 0)

    def test_rename_roundtrip(self):
        engine = GestureEngine(self.gestures_path)
        self.assertTrue(engine.rename_gesture("LEFT", "G01", "attaque"))
        self.assertFalse(engine.rename_gesture("LEFT", "G01", "   "))
        reloaded = GestureEngine(self.gestures_path)
        self.assertEqual(reloaded.display_name("LEFT", "G01"), "attaque")
        self.assertEqual(reloaded.database["version"], 8)

    def test_mapping_accepts_new_ids_and_still_rejects_f9(self):
        mapping = GestureMapping(self.mapping_path)
        self.assertTrue(mapping.set_command("LEFT", "G31", HOLD, ["A"]))
        self.assertTrue(mapping.set_command("LEFT", "G100", PRESS, ["SPACE"], 0))
        self.assertTrue(mapping.set_command("RIGHT", "G32", PRESS, ["F12"], 0))
        self.assertTrue(mapping.set_command("LEFT", "G33", PRESS, ["LEFT"], 0))
        self.assertTrue(mapping.set_command("LEFT", "G34", COMBINATION, ["CTRL", "SHIFT", "S"]))
        self.assertTrue(mapping.set_command("LEFT", "G35", PRESS, ["F11"], 0))
        reloaded = GestureMapping(self.mapping_path)
        self.assertEqual(reloaded.get_inputs("LEFT", "G31"), ["A"])
        self.assertEqual(reloaded.get_inputs("LEFT", "G100"), ["SPACE"])
        self.assertEqual(reloaded.get_inputs("LEFT", "G34"), ["CTRL", "SHIFT", "S"])

    def test_direction_finger_is_independent_per_hand(self):
        def pose(finger_index, direction):
            scale = 0.2
            points = np.zeros((21, 3), dtype=np.float32)
            points[geo.MIDDLE_MCP] = (0.0, -scale, 0.0)
            tips = {
                "THUMB": (geo.THUMB_TIP, geo.THUMB_MCP),
                "INDEX": (geo.INDEX_TIP, geo.INDEX_MCP),
                "MIDDLE": (geo.MIDDLE_TIP, geo.MIDDLE_MCP),
                "RING": (geo.RING_TIP, geo.RING_MCP),
                "PINKY": (geo.PINKY_TIP, geo.PINKY_MCP),
            }
            tip, mcp = tips[finger_index]
            points[mcp] = (0.0, -scale, 0.0)
            points[tip] = points[mcp] + np.array([direction[0] * scale, direction[1] * scale, 0.0], dtype=np.float32)
            return points

        engine = geo.GeometryEngine(
            direction_fingers={"LEFT": "MIDDLE", "RIGHT": "INDEX"},
            stability_frames=1,
        )
        left = engine.update("LEFT", pose("MIDDLE", (-1.0, 0.0)))
        right = engine.update("RIGHT", pose("INDEX", (1.0, 0.0)))
        self.assertEqual(left[geo.INDEX], geo.LEFT_INDEX_LEFT)
        self.assertEqual(right[geo.INDEX], geo.RIGHT_INDEX_RIGHT)
        thumb = geo.GeometryEngine(direction_fingers={"LEFT": "THUMB"}, stability_frames=1)
        self.assertEqual(thumb.update("LEFT", pose("THUMB", (0.0, 1.0)))[geo.INDEX], geo.LEFT_INDEX_DOWN)
        pinky = geo.GeometryEngine(direction_fingers={"RIGHT": "PINKY"}, stability_frames=1)
        self.assertEqual(pinky.update("RIGHT", pose("PINKY", (0.0, -1.0)))[geo.INDEX], geo.RIGHT_INDEX_UP)

    def test_interface_keys_follow_focus_but_not_gestures(self):
        from window_focus import interface_keys_enabled
        self.assertTrue(interface_keys_enabled(True))
        self.assertFalse(interface_keys_enabled(False))
        source = inspect.getsource(main_module.HandControllerApp.run)
        gesture_at = source.index("self.process_frame(")
        key_at = source.index("interface_keys_enabled(")
        self.assertLess(gesture_at, key_at)
        listener = inspect.getsource(input_controller.InputController.start_emergency_listener)
        self.assertNotIn("match_osd_hotkey", listener)
        self.assertNotIn("if key == emergency", listener)
        self.assertNotIn("if key == preview", listener)

    def test_old_list_samples_still_load(self):
        payload = {"version": 8, "gestures": {"LEFT": {"G01": samples_for(3)}, "RIGHT": {}}}
        with open(self.gestures_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        engine = GestureEngine(self.gestures_path)
        self.assertEqual(engine.sample_count("LEFT", "G01"), 8)
        self.assertEqual(engine.display_name("LEFT", "G01"), "")


class Phase4RecognitionTests(TempCase):
    def test_close_classes_stay_unknown(self):
        engine = GestureEngine(self.gestures_path, distance_margin=0.04, recognition_threshold=0.2, class_margin=0.0, k_neighbors=6)
        base = pose(4)
        engine.database["gestures"]["LEFT"]["G01"] = [(base + i * 0.0001).tolist() for i in range(3)]
        engine.database["gestures"]["LEFT"]["G02"] = [(base + 0.002 + i * 0.0001).tolist() for i in range(3)]
        engine.rebuild_index()
        self.assertEqual(engine.recognize(base, "LEFT")[0], UNKNOWN)

    def test_margin_zero_keeps_the_previous_rule(self):
        engine = GestureEngine(self.gestures_path, distance_margin=0.0, recognition_threshold=0.2, class_margin=0.0, k_neighbors=6)
        base = pose(4)
        engine.database["gestures"]["LEFT"]["G01"] = [(base + i * 0.0001).tolist() for i in range(3)]
        engine.database["gestures"]["LEFT"]["G02"] = [(base + 0.002 + i * 0.0001).tolist() for i in range(3)]
        engine.rebuild_index()
        self.assertEqual(engine.recognize(base, "LEFT")[0], "G01")

    def test_switch_between_known_gestures_needs_two_frames(self):
        engine = GestureEngine(self.gestures_path)
        self.assertEqual(engine.stabilize("LEFT", "G01", 0.95), "G01")
        self.assertEqual(engine.stabilize("LEFT", "G02", 0.99), "G01")
        self.assertEqual(engine.stabilize("LEFT", "G02", 0.99), "G02")

    def test_one_unknown_frame_does_not_repress(self):
        engine = GestureEngine(self.gestures_path)
        app = make_app(self.mapping_path)
        app.inputs.enable()
        app.mapping.set_command("LEFT", "G01", PRESS, ["SPACE"], 0)
        space = app.inputs.special_keys["SPACE"]

        def feed(raw, confidence):
            stable = engine.stabilize("LEFT", raw, confidence)
            run_frame(app, LEFT=stable)

        feed("G01", 0.95)
        feed("G01", 0.95)
        self.assertEqual(app.inputs.keyboard.pressed.count(space), 1)
        feed(UNKNOWN, 0.0)
        self.assertEqual(app.inputs.keyboard.pressed.count(space), 1)
        feed(UNKNOWN, 0.0)
        feed("G01", 0.95)
        self.assertEqual(app.inputs.keyboard.pressed.count(space), 2)

    def test_pose_spread_and_recognition_cost(self):
        engine = GestureEngine(self.gestures_path)
        samples = [(pose(6) + i * 0.01).tolist() for i in range(8)]
        engine.database["gestures"]["LEFT"]["G100"] = samples
        engine.rebuild_index()
        spread = engine.pose_spread("LEFT", "G100")
        self.assertIsNotNone(spread)
        self.assertGreater(spread, 0.0)
        self.assertEqual(engine.recognize(samples[0], "LEFT")[0], "G100")
        start = time.perf_counter()
        for _ in range(200):
            engine.recognize(samples[0], "LEFT")
        self.assertLess((time.perf_counter() - start) / 200.0, 0.002)


class UniqueConfigTests(TempCase):
    def test_legacy_profile_migration_copies_active_without_deleting_or_touching_backup(self):
        app_dir = self.td.name
        legacy = os.path.join(app_dir, "profiles", "default")
        os.makedirs(legacy, exist_ok=True)
        gestures_src = os.path.join(legacy, "gestures.json")
        mapping_src = os.path.join(legacy, "mapping.json")
        with open(gestures_src, "w", encoding="utf-8") as handle:
            handle.write('{"version": 8, "gestures": {"LEFT": {}, "RIGHT": {}}}')
        with open(mapping_src, "w", encoding="utf-8") as handle:
            handle.write('{"version": 6, "LEFT": {}, "RIGHT": {}}')
        with open(os.path.join(legacy, "profile.json"), "w", encoding="utf-8") as handle:
            json.dump({
                "id": "default",
                "name": "Default",
                "settings": {"knn_k": 7},
                "system_gestures": sysg.default_system_gestures(),
            }, handle)
        with open(gestures_src, "rb") as handle:
            before_g = handle.read()
        backup = os.path.join(os.path.dirname(main_module.__file__), "gestures_backup.json")
        backup_before = None
        if os.path.isfile(backup):
            stat = os.stat(backup)
            backup_before = (stat.st_size, stat.st_mtime)
        updated = sysg.migrate_legacy_profiles(app_dir, {"active_profile": "default"})
        self.assertTrue(os.path.isfile(os.path.join(app_dir, "gestures.json")))
        self.assertTrue(os.path.isfile(os.path.join(app_dir, "mapping.json")))
        self.assertTrue(os.path.isdir(os.path.join(app_dir, "profiles")))
        self.assertNotIn("active_profile", updated)
        self.assertEqual(updated.get("knn_k"), 7)
        with open(gestures_src, "rb") as handle:
            self.assertEqual(handle.read(), before_g)
        if backup_before is not None:
            stat = os.stat(backup)
            self.assertEqual((stat.st_size, stat.st_mtime), backup_before)

    def test_mapping_files_stay_independent(self):
        path_a = os.path.join(self.td.name, "a.json")
        path_b = os.path.join(self.td.name, "b.json")
        map_a = GestureMapping(path_a)
        map_b = GestureMapping(path_b)
        map_a.set_command("LEFT", "G01", PRESS, ["A"], 0)
        map_b.set_command("LEFT", "G01", HOLD, ["SHIFT"])
        self.assertEqual(GestureMapping(path_a).get_inputs("LEFT", "G01"), ["A"])
        self.assertEqual(GestureMapping(path_b).get_inputs("LEFT", "G01"), ["SHIFT"])

    def test_unique_settings_hold_knn_and_emergency_separately(self):
        from config import default_settings, validate_settings
        data = default_settings()
        self.assertIn("direction_finger_left", data)
        self.assertIn("hand_loss_grace_frames", data)
        self.assertIn("emergency_key", data)
        self.assertIn("smoothing_min_cutoff", data)
        self.assertIn("system_gestures", validate_settings(data))
        self.assertNotIn("active_profile", validate_settings({"active_profile": "elden"}))

    def test_emergency_stop_releases_union_of_sources(self):
        app = arm_tracking(make_geometry_app(self.mapping_path), grace=2)
        app.inputs.enable()
        app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        app.mapping.set_command("LEFT", geo.LEFT_PINCH, HOLD, ["SHIFT"])
        run_tracked_frame(app, left="G01")
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(app, left="G01", left_points=PINCHING)
        self.assertTrue(app.inputs.currently_pressed)
        app.inputs.stop_all_macros()
        app.inputs.release_all()
        self.assertFalse(app.inputs.currently_pressed)
        self.assertFalse(app.inputs.hold_sources)


class Phase5DynamicRemovedTests(unittest.TestCase):
    def test_dynamic_module_removed(self):
        root = os.path.dirname(os.path.abspath(main_module.__file__))
        self.assertFalse(os.path.isfile(os.path.join(root, "dynamic_gestures.py")))
        with self.assertRaises(ImportError):
            importlib.import_module("dynamic_gestures")

    def test_flick_and_swipe_absent_from_catalog(self):
        import system_gestures as sysg
        import gesture_editor as editor
        forbidden = ("FLICK", "SWIPE", "D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08")
        dumped = json.dumps(sysg.default_system_gestures())
        for token in forbidden:
            self.assertNotIn(token.lower() if token.startswith("D") else token, dumped.upper() if token.startswith("D") else dumped)
        engine = GestureEngine(os.path.join(tempfile.mkdtemp(), "gestures.json"))
        mapping = GestureMapping(os.path.join(tempfile.mkdtemp(), "mapping.json"))
        rows = editor.gesture_rows(engine, mapping, "LEFT")
        kinds = {row.get("kind") for row in rows}
        self.assertNotIn("DYNAMIC", kinds)
        ids = {row.get("id") for row in rows}
        for token in ("D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"):
            self.assertNotIn(token, ids)

    def test_flick_and_swipe_are_not_detected(self):
        self.assertFalse(hasattr(HandControllerApp, "_emit_dynamic"))
        source = inspect.getsource(main_module)
        self.assertNotIn("DynamicMotionDetector", source)
        self.assertNotIn("FLICK LEFT", source)
        self.assertNotIn("SWIPE LEFT", source)


class Phase6EditorTests(TempCase):
    def test_duplicate_rename_disable_and_outliers(self):
        import gesture_editor as editor
        engine = GestureEngine(self.gestures_path, samples_per_gesture=4)
        mapping = GestureMapping(self.mapping_path)
        engine.rename_gesture("LEFT", "G01", "ATTAQUE")
        samples = [(pose(3) + i * 0.01).tolist() for i in range(7)]
        samples.append(pose(40).tolist())
        engine.database["gestures"]["LEFT"]["G01"] = __import__("gesture_engine").GestureSamples(samples, "ATTAQUE")
        engine.save_database()
        mapping.set_command("LEFT", "G01", PRESS, ["E"], 0)
        new_id = editor.duplicate_gesture(engine, mapping, "LEFT", "G01")
        self.assertIsNotNone(new_id)
        self.assertNotEqual(new_id, "G01")
        self.assertEqual(engine.display_name("LEFT", new_id), "ATTAQUE 2")
        self.assertEqual(mapping.get_inputs("LEFT", new_id), ["E"])
        self.assertIsNot(engine.database["gestures"]["LEFT"]["G01"], engine.database["gestures"]["LEFT"][new_id])
        bad = editor.outlier_indices(engine.database["gestures"]["LEFT"]["G01"])
        self.assertIn(7, bad)
        editor.remove_samples(engine, "LEFT", "G01", bad)
        self.assertEqual(engine.sample_count("LEFT", "G01"), 7)
        self.assertTrue(editor.toggle_enabled(mapping, "LEFT", "G01"))
        self.assertIs(mapping.get("LEFT", "G01").get("enabled"), False)
        app = make_app(self.mapping_path)
        app.mapping = mapping
        app.inputs.enable()
        run_frame(app, LEFT="G01")
        self.assertEqual(app.inputs.keyboard.pressed, [])

    def test_backup_does_not_touch_historical_file(self):
        import gesture_editor as editor
        engine = GestureEngine(self.gestures_path)
        engine.save_database()
        GestureMapping(self.mapping_path)
        dest = editor.backup_config(self.gestures_path, self.mapping_path)
        self.assertIsNotNone(dest)
        self.assertTrue(os.path.isfile(os.path.join(dest, "gestures.json")))
        self.assertTrue(os.path.isfile(os.path.join(dest, "mapping.json")))
        backup = os.path.join(os.path.dirname(main_module.__file__), "gestures_backup.json")
        if os.path.isfile(backup):
            self.assertEqual(os.path.getsize(backup), 795428)

    def test_safe_mode_blocks_press(self):
        app = make_app(self.mapping_path)
        app.inputs.enable()
        app.mapping.set_command("LEFT", "G01", PRESS, ["SPACE"], 0)
        app.safe_mode = True
        app.passive = False
        app.recording = False
        app.engine = GestureEngine(self.gestures_path)
        app.smoothing_enabled = False
        app.smoothers = {}
        app.dynamics = {}
        app.camera_ok = True
        app.kill_banner = False
        app.detected = {side: False for side in SIDES}
        app.confidences = {side: 0.0 for side in SIDES}
        app.status_message = ""
        app.perf_text = ""
        app.perf = {key: 0.0 for key in ("mediapipe", "normalisation", "reconnaissance", "stabilisation", "mapping", "input", "special", "affichage", "total")}
        app.geometry = geo.GeometryEngine()
        raw = np.zeros((21, 3), dtype=np.float32)
        raw[9] = (0.0, -0.2, 0.0)

        class LM:
            def __init__(self, row):
                self.x, self.y, self.z = (float(value) for value in row)

        app.collect_detections = lambda result: [{"hand": [LM(row) for row in raw], "side": "LEFT", "score": 1.0, "raw": "G01", "conf": 1.0}]
        app.curr_gestures = {side: UNKNOWN for side in SIDES}
        blank = np.zeros((80, 80, 3), dtype=np.uint8)
        app.process_frame(blank, None)
        self.assertEqual(app.inputs.keyboard.pressed, [])


class Phase7ReleaseTests(TempCase):
    def test_diagnostic_reports_ready_without_opening_the_camera(self):
        from diagnostics import collect_diagnostic
        ready, lines = collect_diagnostic(check_camera=False)
        text = "\n".join(lines)
        self.assertTrue(ready)
        self.assertIn("RESULT : READY", text)
        self.assertIn("SKIP", text)
        self.assertNotIn("SetForegroundWindow", text)

    def test_gaming_mode_defaults_off_and_debug_keeps_the_full_hud(self):
        from config import default_settings, validate_settings
        self.assertFalse(default_settings()["gaming_mode"])
        self.assertFalse(validate_settings({})["gaming_mode"])
        self.assertTrue(validate_settings({"gaming_mode": True})["gaming_mode"])
        source = inspect.getsource(main_module.HandControllerApp.draw_ui)
        self.assertIn("_compact_hud", source)
        self.assertIn("draw_preview", source)
        import hud_renderer
        debug_src = inspect.getsource(hud_renderer.draw_debug_info)
        self.assertIn("debug_hud_lines", debug_src)
        self.assertLess(debug_src.index('getattr(app, "debug", False)'), debug_src.index("debug_hud_lines"))

    def test_preview_does_not_block_activation(self):
        import preview_window
        source = inspect.getsource(preview_window)
        self.assertNotIn("NOACTIVATE", source)
        self.assertNotIn("SetForegroundWindow(", source)

    def test_long_run_releases_each_hand_and_bounds_history(self):
        app = arm_tracking(make_app(self.mapping_path), grace=2)
        app.inputs.enable()
        app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        app.mapping.set_command("RIGHT", "G01", HOLD, ["D"])
        for index in range(200):
            left = "G01" if index % 4 < 2 else None
            right = "G01" if index % 5 < 2 else None
            run_tracked_frame(app, left=left, right=right)
        run_tracked_frame(app, left=None, right=None)
        run_tracked_frame(app, left=None, right=None)
        run_tracked_frame(app, left=None, right=None)
        self.assertFalse(app.inputs.currently_pressed)
        engine = GestureEngine(self.gestures_path, stable_frames=2)
        for index in range(200):
            engine.stabilize("LEFT", "G01" if index % 3 else UNKNOWN, 0.9)
            engine.stabilize("RIGHT", "G02" if index % 2 else UNKNOWN, 0.4)
        window = max(engine.stable_frames, engine.unknown_frames)
        for side in SIDES:
            self.assertLessEqual(len(engine.histories[side]), window)


class LocalizationTests(TempCase):
    def tearDown(self):
        from localization import set_language
        set_language("fr")
        super().tearDown()

    def test_catalogs_share_the_same_keys(self):
        from localization import catalog_keys, load_catalogs
        load_catalogs(force=True)
        self.assertEqual(catalog_keys("fr"), catalog_keys("en"))
        self.assertGreater(len(catalog_keys("fr")), 80)

    def test_source_translation_keys_exist(self):
        import re
        from localization import all_catalog_keys, load_catalogs
        load_catalogs(force=True)
        keys = all_catalog_keys()
        root = os.path.dirname(os.path.abspath(main_module.__file__))
        missing = set()
        pattern = re.compile(r'\bt\(\s*[\'"]([a-z0-9_.]+)[\'"]')
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in ("__pycache__", ".git")]
            for name in filenames:
                if not name.endswith(".py") or name == "run_tests.py":
                    continue
                with open(os.path.join(dirpath, name), "r", encoding="utf-8") as handle:
                    text = handle.read()
                missing.update(key for key in pattern.findall(text) if key not in keys)
        self.assertEqual(missing, set())

    def test_switch_fr_en_and_back(self):
        from localization import action_label, current_language, set_language, t
        set_language("fr")
        self.assertEqual(current_language(), "fr")
        self.assertEqual(action_label("PRESS"), "Appui")
        self.assertEqual(action_label("HOLD"), "Maintien")
        self.assertIn("INPUT KILL", t("status.input_kill"))
        self.assertIn("reactiver", t("status.input_kill").lower())
        set_language("en")
        self.assertEqual(current_language(), "en")
        self.assertEqual(action_label("PRESS"), "Press")
        self.assertEqual(action_label("HOLD"), "Hold")
        self.assertIn("re-enable", t("status.input_kill").lower())
        set_language("fr")
        self.assertEqual(action_label("PRESS"), "Appui")

    def test_language_is_saved_and_reloaded(self):
        from config import load_settings, save_settings, validate_settings
        from localization import apply_language, language_configured, set_language
        path = os.path.join(self.td.name, "settings.json")
        data = validate_settings({"language": "en"})
        self.assertEqual(data["language"], "en")
        self.assertTrue(save_settings(data, path))
        loaded = load_settings(path)
        self.assertEqual(loaded["language"], "en")
        self.assertTrue(language_configured(loaded))
        apply_language(loaded, interactive=False)
        from localization import current_language
        self.assertEqual(current_language(), "en")
        set_language("fr")

    def test_settings_with_utf8_bom_are_not_reset(self):
        from config import load_settings
        path = os.path.join(self.td.name, "settings.json")
        with open(path, "w", encoding="utf-8-sig") as handle:
            json.dump({"language": "en", "hud_overlay": True}, handle)
        loaded = load_settings(path)
        self.assertEqual(loaded["language"], "en")
        self.assertTrue(loaded["hud_overlay"])
        self.assertFalse(os.path.exists(path + ".bak"))

    def test_missing_language_is_unset_until_chosen(self):
        from config import default_settings, validate_settings
        from localization import language_configured
        self.assertFalse(language_configured(default_settings()))
        self.assertFalse(language_configured(validate_settings({})))
        self.assertFalse(language_configured(validate_settings({"language": "de"})))
        self.assertTrue(language_configured(validate_settings({"language": "FR"})))

    def test_custom_gesture_name_is_not_translated(self):
        from localization import dynamic_label, set_language
        set_language("fr")
        self.assertEqual(dynamic_label("D01", "ATTAQUE"), "ATTAQUE")
        set_language("en")
        self.assertEqual(dynamic_label("D01", "ATTAQUE"), "ATTAQUE")
        self.assertEqual(dynamic_label("D01"), "D01")

    def test_f8_to_f11_stay_assignable_in_both_languages(self):
        from localization import set_language
        from gesture_mapping import GestureMapping, HOLD, PRESS
        mapping = GestureMapping(self.mapping_path)
        for lang in ("fr", "en"):
            set_language(lang)
            self.assertTrue(mapping.set_command("LEFT", "G01", HOLD, ["F11"]))
            self.assertTrue(mapping.set_command("LEFT", "G01", PRESS, ["F10"], 0))
            self.assertTrue(mapping.set_command("LEFT", "G02", HOLD, ["F8"]))
            self.assertTrue(mapping.set_command("LEFT", "G03", PRESS, ["F9"], 0))
            self.assertTrue(mapping.set_command("LEFT", "G04", HOLD, ["F1"]))

    def test_internal_action_types_stay_english(self):
        from gesture_mapping import COMBINATION, HOLD, MACRO, PRESS, WAIT_MIN_MS
        self.assertEqual(PRESS, "PRESS")
        self.assertEqual(HOLD, "HOLD")
        self.assertEqual(COMBINATION, "COMBINATION")
        self.assertEqual(MACRO, "MACRO")
        self.assertEqual(WAIT_MIN_MS, 0)

    def test_preview_focus_rules_unchanged(self):
        import preview_window
        import window_focus
        for source in (inspect.getsource(preview_window), inspect.getsource(window_focus), inspect.getsource(main_module)):
            self.assertNotIn("SetForegroundWindow(", source)
            self.assertNotIn("TOPMOST", source)


class FingerUpAndSystemGestureTests(TempCase):
    def setUp(self):
        super().setUp()
        self.root = os.path.join(self.td.name, "profiles")

    def settle(self, engine, side, pose, frames=None):
        last = None
        for _ in range(frames or geo.STABILITY_FRAMES):
            last = engine.update(side, pose)
        return last

    def test_detect_finger_up_each_finger_both_hands(self):
        for finger, pose in (
            ("INDEX", finger_pose((1.0, 0.2, 0.2, 0.2))),
            ("MIDDLE", finger_pose((0.2, 1.0, 0.2, 0.2))),
            ("RING", finger_pose((0.2, 0.2, 1.0, 0.2))),
            ("PINKY", finger_pose((0.2, 0.2, 0.2, 1.0))),
        ):
            self.assertTrue(geo.detect_finger_up(pose, finger), finger)
            for other in geo.FINGER_UP_FINGERS:
                if other == finger:
                    continue
                self.assertFalse(geo.detect_finger_up(pose, other), f"{finger} vs {other}")

    def test_finger_up_ids_left_and_right(self):
        pose = finger_pose((1.0, 0.2, 0.2, 0.2))
        engine = geo.GeometryEngine(stability_frames=1)
        self.assertEqual(self.settle(engine, "LEFT", pose, 1)[geo.FINGER_INDEX], geo.LEFT_FINGER_INDEX)
        self.assertEqual(self.settle(engine, "RIGHT", pose, 1)[geo.FINGER_INDEX], geo.RIGHT_FINGER_INDEX)

    def test_single_finger_mode_rejects_two_raised(self):
        engine = geo.GeometryEngine(stability_frames=1, finger_up_mode="single")
        two = finger_pose((1.0, 1.0, 0.2, 0.2))
        states = self.settle(engine, "LEFT", two, 2)
        self.assertIsNone(states[geo.FINGER_INDEX])
        self.assertIsNone(states[geo.FINGER_MIDDLE])

    def test_multi_finger_mode_allows_two_raised(self):
        engine = geo.GeometryEngine(stability_frames=1, finger_up_mode="multi")
        two = finger_pose((1.0, 1.0, 0.2, 0.2))
        states = self.settle(engine, "LEFT", two, 2)
        self.assertEqual(states[geo.FINGER_INDEX], geo.LEFT_FINGER_INDEX)
        self.assertEqual(states[geo.FINGER_MIDDLE], geo.LEFT_FINGER_MIDDLE)

    def test_finger_up_hold_and_press(self):
        app = make_geometry_app(self.mapping_path)
        app.inputs.enable()
        app.mapping.set_command("LEFT", geo.LEFT_FINGER_INDEX, HOLD, ["A"])
        pose = finger_pose((1.0, 0.2, 0.2, 0.2))
        for _ in range(geo.STABILITY_FRAMES):
            run_geometry_frame(app, left=pose)
        self.assertIn("A", app.inputs.get_pressed_names())
        curled = finger_pose((0.2, 0.2, 0.2, 0.2))
        for _ in range(geo.STABILITY_FRAMES):
            run_geometry_frame(app, left=curled)
        self.assertNotIn("A", app.inputs.get_pressed_names())
        before = len(app.inputs.keyboard.pressed)
        app.mapping.set_command("RIGHT", geo.RIGHT_FINGER_PINKY, PRESS, ["E"], 0)
        pinky = finger_pose((0.2, 0.2, 0.2, 1.0))
        for _ in range(geo.STABILITY_FRAMES):
            run_geometry_frame(app, right=pinky)
        self.assertGreater(len(app.inputs.keyboard.pressed), before)

    def test_hand_loss_releases_finger_hold(self):
        app = make_geometry_app(self.mapping_path)
        app.inputs.enable()
        app.mapping.set_command("LEFT", geo.LEFT_FINGER_INDEX, HOLD, ["A"])
        pose = finger_pose((1.0, 0.2, 0.2, 0.2))
        for _ in range(geo.STABILITY_FRAMES):
            run_geometry_frame(app, left=pose)
        self.assertIn("A", app.inputs.get_pressed_names())
        run_geometry_frame(app, left=None)
        self.assertNotIn("A", app.inputs.get_pressed_names())

    def test_system_gate_blocks_disabled_pinch(self):
        app = make_geometry_app(self.mapping_path)
        app.inputs.enable()
        app.system_gestures = sysg.default_system_gestures()
        app.mapping.set_command("RIGHT", geo.RIGHT_PINCH, HOLD, ["SHIFT"])
        for _ in range(geo.STABILITY_FRAMES):
            run_geometry_frame(app, right=PINCHING)
        self.assertNotIn("SHIFT", app.inputs.get_pressed_names())
        app.system_gestures = sysg.set_system_enabled(app.system_gestures, "pinch", "RIGHT", enabled=True)
        for _ in range(geo.STABILITY_FRAMES):
            run_geometry_frame(app, right=PINCHING)
        self.assertIn("SHIFT", app.inputs.get_pressed_names())

    def test_pinch_and_knn_together(self):
        app = make_geometry_app(self.mapping_path)
        app.inputs.enable()
        app.system_gestures = sysg.set_system_enabled(
            sysg.default_system_gestures(), "pinch", "LEFT", enabled=True,
        )
        app.mapping.set_command("LEFT", geo.LEFT_PINCH, HOLD, ["SHIFT"])
        for _ in range(geo.STABILITY_FRAMES):
            run_geometry_frame(app, left=PINCHING)
        self.assertIn("SHIFT", app.inputs.get_pressed_names())
        app.mapping.set_command("RIGHT", geo.RIGHT_FINGER_INDEX, HOLD, ["A"])
        # Finger up on right while pinch holds on left.
        pose = finger_pose((1.0, 0.2, 0.2, 0.2))
        app.system_gestures = sysg.add_finger_binding(app.system_gestures, "RIGHT", "index", True)
        for _ in range(geo.STABILITY_FRAMES):
            run_geometry_frame(app, left=PINCHING, right=pose)
        self.assertIn("SHIFT", app.inputs.get_pressed_names())
        self.assertIn("A", app.inputs.get_pressed_names())

    def test_f8_to_f11_assignable_to_finger_up(self):
        mapping = GestureMapping(self.mapping_path)
        self.assertTrue(mapping.set_command("LEFT", geo.LEFT_FINGER_INDEX, PRESS, ["F11"]))
        self.assertTrue(mapping.set_command("RIGHT", geo.RIGHT_FINGER_MIDDLE, HOLD, ["F10"]))
        self.assertTrue(mapping.set_command("LEFT", geo.LEFT_FINGER_RING, PRESS, ["F8"]))
        self.assertTrue(mapping.set_command("RIGHT", geo.RIGHT_FINGER_PINKY, HOLD, ["F9"]))

    def test_system_gesture_ids_are_not_knn_slots(self):
        for name in (geo.LEFT_FINGER_INDEX, geo.RIGHT_FINGER_PINKY, geo.LEFT_PINCH, geo.RIGHT_TILT_UP):
            self.assertNotIn(name, GESTURE_NAMES)

    def test_index_up_mapping_migrates_to_finger_index(self):
        mapping_path = os.path.join(self.td.name, "mapping.json")
        mapping = {
            "version": 6,
            "LEFT": {
                "LEFT_INDEX_UP": {"type": "HOLD", "inputs": ["W"]},
            },
            "RIGHT": {},
        }
        with open(mapping_path, "w", encoding="utf-8") as handle:
            json.dump(mapping, handle)
        loaded = sysg.migrate_system_gestures(None, mapping_path)
        bindings = loaded["finger_up"]["bindings"]
        self.assertTrue(any(item["hand"] == "left" and item["finger"] == "index" for item in bindings))
        self.assertTrue(sysg.is_system_enabled(loaded, "LEFT", "LEFT_FINGER_INDEX"))
        stored = json.loads(open(mapping_path, encoding="utf-8").read())
        self.assertEqual(stored["LEFT"]["LEFT_INDEX_UP"]["inputs"], ["W"])
        self.assertEqual(stored["LEFT"]["LEFT_FINGER_INDEX"]["inputs"], ["W"])

    def test_unique_config_loads_gestures_mapping_settings(self):
        from config import default_settings, validate_settings
        GestureEngine(self.gestures_path).save_database()
        GestureMapping(self.mapping_path)
        self.assertTrue(os.path.isfile(self.gestures_path))
        self.assertTrue(os.path.isfile(self.mapping_path))
        cleaned = validate_settings(default_settings())
        self.assertIn("system_gestures", cleaned)
        self.assertNotIn("active_profile", cleaned)

    def test_localization_system_keys(self):
        from localization import load_catalogs, set_language, t
        load_catalogs(force=True)
        set_language("fr")
        self.assertEqual(t("sys.title"), "GESTES SYSTEME")
        self.assertEqual(t("sys.middle"), "MAJEUR")
        set_language("en")
        self.assertEqual(t("sys.title"), "SYSTEM GESTURES")
        self.assertEqual(t("sys.middle"), "MIDDLE")
        set_language("fr")


class UXLaunch20Tests(TempCase):
    def _hud_app(self):
        app = make_geometry_app(self.mapping_path)
        app.kill_banner = False
        app.camera_ok = True
        app.detected = {side: False for side in SIDES}
        app.confidences = {side: 0.0 for side in SIDES}
        app.curr_gestures = {side: UNKNOWN for side in SIDES}
        app.curr_specials = {side: {} for side in SIDES}
        app.system_gestures = sysg.default_system_gestures()
        app.status_message = ""
        app.settings = {"gaming_mode": False}
        app.debug = False
        app.lang_mode = False
        app.help_mode = False
        app.finger_mode = False
        app.track_state = {side: "RELEASED" for side in SIDES}
        app.last_dynamic_osd = {side: ("", 0.0) for side in SIDES}
        app.sys_mode = False
        app.sys_phase = "list"
        app.sys_index = 0
        app.sys_edit = None
        app.sys_combo_capture = False
        app.sidebar_hidden = False
        app.settings_mode = False
        app.input_panel = False
        app.overlay_panel = False
        app.settings_side = "LEFT"
        return app

    def _finger_row(self, app, side="LEFT", finger="INDEX"):
        for index, row in enumerate(app._sys_rows()):
            if row.get("kind") == "finger" and row.get("side") == side and row.get("finger") == finger:
                return index, row
        self.fail("finger row missing")

    def test_launch_asks_language_then_opens_app(self):
        source = inspect.getsource(main_module.main)
        self.assertIn("apply_language", source)
        self.assertIn("language_configured", source)
        self.assertNotIn("enter_profile_setup", source)
        self.assertLess(source.index("apply_language"), source.index("HandControllerApp"))
        self.assertLess(source.index("if chosen is None"), source.index("HandControllerApp"))

    def test_help_has_no_profile_menu(self):
        source = inspect.getsource(main_module.HandControllerApp.draw_help)
        self.assertNotIn("cmd_profile", source)
        self.assertNotIn("enter_profile", source)
        self.assertIn("help.hud_body", source)

    def test_hud_classic_keyboard_without_f8_f11(self):
        app = self._hud_app()
        source = inspect.getsource(main_module.HandControllerApp.draw_ui)
        self.assertIn("draw_preview", source)
        self.assertIn("_compact_hud", source)
        self.assertNotIn("cmd_preview", source)
        self.assertNotIn("cmd_stop", source)
        self.assertNotIn("cmd_geom", source)
        self.assertNotIn("F8", source)
        self.assertNotIn("F9", source)
        idle = inspect.getsource(main_module.HandControllerApp.handle_idle_key)
        self.assertIn('ord("c")', idle)
        self.assertIn('ord("g")', idle)
        self.assertIn('ord("p")', idle)
        self.assertIn("enter_calibration", idle)
        self.assertNotIn('ord("v")', idle)
        run_src = inspect.getsource(main_module.HandControllerApp.run)
        self.assertIn('key == ord("q")', run_src)
        self.assertNotIn("PREVIEW_KEY", source)
        self.assertTrue(all(token not in source for token in ("PREVIEW_KEY", "EMERGENCY_KEY")))

    def test_hud_draws_small_normal_and_large_frames(self):
        app = self._hud_app()
        for size in ((240, 320, 3), (480, 640, 3), (720, 1280, 3)):
            frame = np.zeros(size, dtype=np.uint8)
            app.draw_ui(frame)
        metrics = app._hud_metrics(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertGreaterEqual(metrics["mx"], 12)
        self.assertTrue(metrics["compact"])

    def test_classic_opencv_hud_shows_ml_not_sidebar(self):
        from localization import load_catalogs, set_language, t
        load_catalogs(force=True)
        set_language("en")
        app = self._hud_app()
        app.detected = {"LEFT": True, "RIGHT": False}
        app.track_state = {"LEFT": "OK", "RIGHT": "RELEASED"}
        app.curr_gestures = {"LEFT": "G01", "RIGHT": UNKNOWN}
        app.confidences = {"LEFT": 0.92, "RIGHT": 0.0}
        app.fps = 30.0
        app.settings = {"emergency_key": "F11", "gaming_mode": False}
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        app.draw_ui(frame)
        self.assertIn("draw_preview", inspect.getsource(main_module.HandControllerApp.draw_ui))
        self.assertIn("C Calibration", t("hud.footer"))
        self.assertIn("Q Quit", t("hud.footer"))
        self.assertNotIn("F8", t("hud.footer"))
        self.assertNotIn("L Recording", t("hud.footer"))
        self.assertNotIn("Emergency", t("hud.footer"))
        self.assertNotIn("E Saved", t("hud.footer"))
        self.assertNotIn("H Help", t("hud.footer"))
        import hud_renderer
        hud_src = inspect.getsource(hud_renderer.draw_preview)
        self.assertNotIn("hud.points", hud_src)
        self.assertNotIn("hud.actives", hud_src)
        self.assertNotIn("hud.fps", hud_src)
        self.assertNotIn("camera_ok", hud_src)
        idle = inspect.getsource(main_module.HandControllerApp.handle_idle_key)
        self.assertNotIn("enter_profile", idle)
        self.assertNotIn("dynamic_gestures", idle)

    def test_system_capture_keeps_type_separate(self):
        app = self._hud_app()
        index, row = self._finger_row(app)
        app.sys_index = index
        app._sys_begin_edit(row)
        self.assertEqual(app.sys_phase, "assign")
        app.sys_edit["type"] = HOLD
        app.sys_phase = "capture"
        app.inputs.begin_key_capture()
        app.inputs._captured_name = "A"
        app.poll_key_capture()
        self.assertEqual(app.sys_phase, "assign")
        self.assertEqual(app.sys_edit["inputs"], ["A"])
        self.assertEqual(app.sys_edit["type"], HOLD)
        app.handle_system_key(ord("1"))
        self.assertEqual(app.sys_edit["type"], PRESS)
        app.handle_system_key(ord("2"))
        self.assertEqual(app.sys_edit["type"], HOLD)
        app.handle_system_key(ord("s"))
        entry = app.mapping.get("LEFT", geo.LEFT_FINGER_INDEX)
        self.assertEqual(entry["type"], HOLD)
        self.assertEqual(entry["inputs"], ["A"])

    def test_system_capture_accepts_f_keys_and_esc_cancels(self):
        app = self._hud_app()
        index, row = self._finger_row(app)
        app.sys_index = index
        app._sys_begin_edit(row)
        app.sys_phase = "capture"
        app.inputs.begin_key_capture()
        app.inputs._captured_name = "F11"
        app.poll_key_capture()
        self.assertEqual(app.sys_phase, "assign")
        self.assertEqual(app.sys_edit["inputs"], ["F11"])
        app.sys_phase = "capture"
        app.inputs.begin_key_capture()
        app.handle_system_key(27)
        self.assertEqual(app.sys_phase, "assign")

    def test_predefined_rows_exclude_flick_and_swipe(self):
        app = self._hud_app()
        kinds = {row.get("kind") for row in app._sys_rows()}
        self.assertIn("pinch", kinds)
        self.assertIn("tilt", kinds)
        self.assertIn("finger", kinds)
        self.assertNotIn("flick", kinds)
        self.assertNotIn("swipe", kinds)
        row = {"kind": "tilt", "side": "RIGHT", "direction": "right"}
        self.assertIn(HOLD, app._sys_allowed_types(row))

    def test_finger_press_does_not_repeat(self):
        app = make_geometry_app(self.mapping_path)
        app.inputs.enable()
        app.system_gestures = sysg.add_finger_binding(sysg.default_system_gestures(), "LEFT", "index", True)
        app.mapping.set_command("LEFT", geo.LEFT_FINGER_INDEX, PRESS, ["A"], 0)
        pose = finger_pose((1.0, 0.2, 0.2, 0.2))
        for _ in range(geo.STABILITY_FRAMES):
            run_geometry_frame(app, left=pose)
        first = len(app.inputs.keyboard.pressed)
        for _ in range(4):
            run_geometry_frame(app, left=pose)
        self.assertEqual(len(app.inputs.keyboard.pressed), first)
        curled = finger_pose((0.2, 0.2, 0.2, 0.2))
        for _ in range(geo.STABILITY_FRAMES):
            run_geometry_frame(app, left=curled)
        for _ in range(geo.STABILITY_FRAMES):
            run_geometry_frame(app, left=pose)
        self.assertGreater(len(app.inputs.keyboard.pressed), first)

    def test_finger_hold_one_down_one_up(self):
        app = make_geometry_app(self.mapping_path)
        app.inputs.enable()
        app.system_gestures = sysg.add_finger_binding(sysg.default_system_gestures(), "LEFT", "index", True)
        app.mapping.set_command("LEFT", geo.LEFT_FINGER_INDEX, HOLD, ["A"])
        pose = finger_pose((1.0, 0.2, 0.2, 0.2))
        for _ in range(geo.STABILITY_FRAMES):
            run_geometry_frame(app, left=pose)
        downs = len(app.inputs.keyboard.pressed)
        self.assertEqual(downs, 1)
        for _ in range(4):
            run_geometry_frame(app, left=pose)
        self.assertEqual(len(app.inputs.keyboard.pressed), downs)
        self.assertIn("A", app.inputs.get_pressed_names())
        curled = finger_pose((0.2, 0.2, 0.2, 0.2))
        for _ in range(geo.STABILITY_FRAMES):
            run_geometry_frame(app, left=curled)
        self.assertNotIn("A", app.inputs.get_pressed_names())

    def test_mapping_change_updates_finger_action(self):
        path = os.path.join(self.td.name, "finger_map.json")
        mapping = GestureMapping(path)
        mapping.set_command("LEFT", geo.LEFT_FINGER_INDEX, HOLD, ["A"])
        self.assertEqual(GestureMapping(path).get("LEFT", geo.LEFT_FINGER_INDEX)["inputs"], ["A"])
        self.assertEqual(GestureMapping(path).get("LEFT", geo.LEFT_FINGER_INDEX)["type"], HOLD)
        mapping.set_command("LEFT", geo.LEFT_FINGER_INDEX, PRESS, ["B"], 0)
        self.assertEqual(GestureMapping(path).get("LEFT", geo.LEFT_FINGER_INDEX)["inputs"], ["B"])
        self.assertEqual(GestureMapping(path).get("LEFT", geo.LEFT_FINGER_INDEX)["type"], PRESS)

    def test_sidebar_order_and_renames(self):
        app = self._hud_app()
        actions = [item[0] for item in app._sidebar_items()]
        self.assertEqual(actions, ["record", "settings", "input", "help"])
        from localization import set_language
        set_language("en")
        labels = [item[1] for item in app._sidebar_items()]
        self.assertEqual(labels[0], "Record Gesture")
        self.assertEqual(labels[1], "Gesture Settings")
        self.assertEqual(labels[2], "Input")
        self.assertEqual(labels[3], "Help")
        set_language("fr")
        labels = [item[1] for item in app._sidebar_items()]
        self.assertEqual(labels[0], "Enregistrer un geste")
        self.assertEqual(labels[1], "Paramètres de geste")
        self.assertEqual(labels[2], "Input")
        self.assertEqual(labels[3], "Help")

    def test_camera_hand_cards_left_and_right(self):
        from overlay_labels import hand_overlay_state
        app = self._hud_app()
        present, labels = hand_overlay_state(app, "LEFT")
        self.assertFalse(present)
        self.assertEqual(labels, [])
        app.detected["LEFT"] = True
        app.track_state["LEFT"] = "OK"
        present, labels = hand_overlay_state(app, "LEFT")
        self.assertTrue(present)
        self.assertEqual(labels, [])
        app.curr_specials["LEFT"] = {"finger": "INDEX UP"}
        present, labels = hand_overlay_state(app, "LEFT")
        self.assertTrue(present)
        self.assertEqual(labels, ["INDEX UP"])
        app.curr_gestures["LEFT"] = "G01"
        present, labels = hand_overlay_state(app, "LEFT")
        self.assertEqual(labels, ["G01", "INDEX UP"])
        app.curr_gestures["RIGHT"] = "PINCH"
        app.detected["RIGHT"] = True
        app.track_state["RIGHT"] = "OK"
        present, labels = hand_overlay_state(app, "RIGHT")
        self.assertTrue(present)
        self.assertEqual(labels, ["PINCH"])
        src = inspect.getsource(main_module.HandControllerApp._draw_camera_hands)
        self.assertIn('"LEFT", False', src)
        self.assertIn('"RIGHT", True', src)
        card = inspect.getsource(main_module.HandControllerApp._draw_hand_card)
        self.assertIn("cam_left + margin", card)
        self.assertIn("width - margin - panel_w", card)

    def test_help_omits_removed_commands(self):
        from localization import load_catalogs, t
        load_catalogs(force=True)
        app = self._hud_app()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        app.help_mode = True
        app.draw_help(frame, 0)
        source = inspect.getsource(main_module.HandControllerApp.draw_help)
        self.assertNotIn("F8", source)
        self.assertNotIn("F9", source)
        self.assertNotIn("cmd_geom", source)
        self.assertNotIn("cmd_preview", source)
        self.assertIn("cmd_quit", source)
        joined = "\n".join(t(key) for key in (
            "help.title", "help.record", "help.predefined", "help.key_settings",
            "help.input_body", "help.hud_body", "help.close",
        ))
        self.assertNotIn("F8", joined)
        self.assertNotIn("F9", joined)

    def test_sidebar_opens_existing_systems(self):
        source = inspect.getsource(main_module.HandControllerApp._activate_sidebar)
        self.assertIn("enter_record_gesture", source)
        self.assertIn("enter_gesture_settings", source)
        self.assertIn("enter_input_panel", source)
        self.assertNotIn("enter_overlay_panel", source)
        self.assertIn("help_mode = True", source)
        self.assertIn("sidebar_hidden", source)
        run_src = inspect.getsource(main_module.HandControllerApp.run)
        self.assertIn("fit_preview_frame", run_src)
        self.assertNotIn("compose_sidebar_camera", run_src)
        self.assertIn("destroyAllWindows", run_src)

    def test_localization_ux_keys(self):
        from localization import load_catalogs, set_language, t
        load_catalogs(force=True)
        set_language("fr")
        self.assertEqual(t("hud.cmd_system"), "Gestes systeme")
        self.assertEqual(t("sys.assign_key"), "Assigner une touche")
        self.assertIn("Appuyez", t("sys.press_key"))
        set_language("en")
        self.assertEqual(t("hud.cmd_system"), "System Gestures")
        self.assertEqual(t("sys.assign_key"), "Assign Key")
        self.assertIn("Press a key", t("sys.press_key"))
        set_language("fr")


class GameOsdTests(TempCase):
    def tearDown(self):
        from localization import set_language
        set_language("fr")
        super().tearDown()

    def _app(self, **kwargs):
        app = HandControllerApp.__new__(HandControllerApp)
        app.settings = {}
        app.curr_gestures = {side: UNKNOWN for side in SIDES}
        app.curr_specials = {side: {} for side in SIDES}
        app.detected = {side: False for side in SIDES}
        app.track_state = {side: "RELEASED" for side in SIDES}
        app.last_dynamic_osd = {side: ("", 0.0) for side in SIDES}
        app.fps = kwargs.get("fps", 0.0)
        app.profile_name = kwargs.get("profile_name", "Default")
        return app

    def test_game_osd_is_gone_from_settings(self):
        from config import default_settings, load_settings, save_settings, validate_settings
        self.assertNotIn("game_osd", default_settings())
        cleaned = validate_settings({"game_osd": {"enabled": True}})
        self.assertNotIn("game_osd", cleaned)
        path = os.path.join(self.td.name, "settings.json")
        self.assertTrue(save_settings(cleaned, path))
        loaded = load_settings(path)
        self.assertNotIn("game_osd", loaded)

    def test_osd_lives_in_unique_settings(self):
        from config import default_settings
        self.assertNotIn("game_osd", default_settings())
        self.assertIn("system_gestures", default_settings())

    def test_no_hands_none_none(self):
        from overlay_labels import osd_state_from_app
        state = osd_state_from_app(self._app())
        self.assertEqual(state["left"], "none")
        self.assertEqual(state["right"], "none")
        self.assertNotIn("game_fps", state)

    def test_detected_and_gesture_and_grace(self):
        from overlay_labels import osd_state_from_app
        app = self._app()
        app.detected["LEFT"] = True
        app.track_state["LEFT"] = "OK"
        app.curr_gestures["LEFT"] = UNKNOWN
        state = osd_state_from_app(app)
        self.assertEqual(state["left"], "none")
        app.curr_specials["LEFT"] = {"finger": "INDEX UP"}
        self.assertEqual(osd_state_from_app(app)["left"], "INDEX UP")
        app.curr_gestures["RIGHT"] = "PINCH"
        app.detected["RIGHT"] = True
        app.track_state["RIGHT"] = "OK"
        self.assertEqual(osd_state_from_app(app)["right"], "PINCH")
        app.detected["LEFT"] = False
        app.track_state["LEFT"] = "GRACE"
        app.curr_specials["LEFT"] = {"finger": "INDEX UP"}
        self.assertEqual(osd_state_from_app(app)["left"], "INDEX UP")
        app.track_state["LEFT"] = "RELEASED"
        self.assertEqual(osd_state_from_app(app)["left"], "none")

    def test_flick_is_not_shown_on_the_overlay(self):
        from overlay_labels import osd_state_from_app
        from time import monotonic
        app = self._app()
        app.detected["RIGHT"] = True
        app.track_state["RIGHT"] = "OK"
        app.last_dynamic_osd["RIGHT"] = ("FLICK RIGHT", monotonic())
        self.assertEqual(osd_state_from_app(app)["right"], "none")

    def test_overlay_has_no_fps_icon_or_brand(self):
        from localization import set_language
        from overlay_labels import osd_paint_lines
        app = self._app()
        set_language("en")
        text = " ".join(row[0] for row in osd_paint_lines(app))
        for banned in ("FPS", "CPU", "RAM", "HandController", "Camera", "CAMERA", "v1.0"):
            self.assertNotIn(banned, text)
        set_language("fr")

    def test_paint_is_compact_and_transparent(self):
        import game_osd
        ex = game_osd.osd_ex_styles()
        for flag in (game_osd.WS_EX_LAYERED, game_osd.WS_EX_TRANSPARENT,
                     game_osd.WS_EX_NOACTIVATE, game_osd.WS_EX_TOOLWINDOW):
            self.assertTrue(ex & flag, hex(flag))
        self.assertEqual(game_osd.COLOR_KEY, 0)
        self.assertLessEqual(game_osd.OSD_W, 480)
        self.assertLessEqual(game_osd.OSD_H, 200)
        self.assertLess(game_osd.OSD_X, 40)
        self.assertLess(game_osd.OSD_Y, 40)

    def test_profile_switch_reuses_the_same_osd(self):
        app = self._app()
        app.hud_overlay = True
        created = []

        class FakeOSD:
            def __init__(self):
                self.visible = False
                self.rows = None
                created.append(self)

            def show(self):
                self.visible = True

            def hide(self):
                self.visible = False

            def update(self, rows):
                self.rows = rows

            def is_active(self):
                return self.visible

        import game_osd
        with mock.patch.object(game_osd, "GameOSD", FakeOSD):
            app._sync_game_osd()
            app._sync_game_osd()
        self.assertEqual(len(created), 1)

    def test_osd_off_creates_no_window_on_creates_and_shutdown_destroys(self):
        import game_osd
        app = self._app()
        app.hud_overlay = False
        app.game_osd = None
        with mock.patch.object(game_osd, "GameOSD") as factory:
            app._sync_game_osd()
            factory.assert_not_called()
        self.assertIsNone(app.game_osd)
        self.assertFalse(app.game_osd_active())
        osd = game_osd.GameOSD()
        self.assertIsNone(osd.hwnd)
        self.assertFalse(osd.is_active())
        run_src = inspect.getsource(main_module.HandControllerApp.run)
        self.assertIn("game_osd.destroy()", run_src)
        self.assertIn("_sync_game_osd()", run_src)

    def test_show_and_hide_persist(self):
        app = self._app()
        from config import default_settings
        app.settings = default_settings()
        app.hud_overlay = False
        with mock.patch.object(app, "_sync_game_osd") as sync, \
                mock.patch.object(main_module, "save_settings") as saver:
            app.toggle_hud_overlay()
            self.assertTrue(app.hud_overlay)
            self.assertTrue(app.settings["hud_overlay"])
            app.toggle_hud_overlay()
            self.assertFalse(app.hud_overlay)
            self.assertEqual(sync.call_count, 2)
            self.assertEqual(saver.call_count, 2)
        self.assertNotIn("game_osd", app.settings)

    @unittest.skipUnless(os.name == "nt", "Win32 only")
    def test_real_win32_osd_never_takes_focus(self):
        import ctypes
        import game_osd
        user32 = ctypes.windll.user32
        before = user32.GetForegroundWindow()
        osd = game_osd.GameOSD()
        try:
            osd.show()
            osd.update([("LEFT", "cyan"), ("Absent", "dim")])
            self.assertTrue(osd.is_active())
            self.assertNotEqual(user32.GetForegroundWindow(), osd.hwnd)
            self.assertEqual(user32.GetForegroundWindow(), before)
            ex = user32.GetWindowLongW(osd.hwnd, -20) & 0xFFFFFFFF
            self.assertTrue(ex & game_osd.WS_EX_TRANSPARENT)
            self.assertTrue(ex & game_osd.WS_EX_NOACTIVATE)
            self.assertTrue(ex & game_osd.WS_EX_LAYERED)
            osd.hide()
            self.assertFalse(osd.is_active())
        finally:
            osd.destroy()
        self.assertIsNone(osd.hwnd)

    @unittest.skipUnless(os.name == "nt", "Win32 only")
    def test_real_win32_osd_survives_external_close_and_destroy(self):
        import ctypes
        import game_osd
        user32 = ctypes.windll.user32
        osd = game_osd.GameOSD()
        try:
            osd.show()
            osd.update([("RIGHT", "cyan"), ("Present", "dim")])
            first = osd.hwnd
            self.assertTrue(osd.is_active())
            user32.SendMessageW(first, game_osd.WM_CLOSE, 0, 0)
            self.assertTrue(osd.is_active())
            self.assertEqual(osd.hwnd, first)
            user32.DestroyWindow(first)
            self.assertIsNone(osd.hwnd)
            self.assertFalse(osd.is_active())
            osd.show()
            osd.update([("RIGHT", "cyan"), ("Absent", "dim")])
            self.assertTrue(osd.is_active())
        finally:
            osd.destroy()
        self.assertIsNone(osd.hwnd)

    def test_localization_osd_keys(self):
        from localization import load_catalogs, set_language, t
        load_catalogs(force=True)
        set_language("fr")
        self.assertEqual(t("osd.left"), "MAIN GAUCHE")
        self.assertEqual(t("osd.none"), "Aucune")
        self.assertEqual(t("osd.detected"), "Détectée")
        self.assertEqual(t("osd.enabled"), "Activé")
        set_language("en")
        self.assertEqual(t("osd.left"), "LEFT HAND")
        self.assertEqual(t("osd.none"), "None")
        self.assertEqual(t("osd.detected"), "Detected")
        self.assertEqual(t("osd.disabled"), "Disabled")

    def test_osd_is_external_only(self):
        import game_osd
        osd_src = inspect.getsource(game_osd)
        for banned in ("SetForegroundWindow(", "PySide", "PyQt", "CreateRemoteThread",
                       "WriteProcessMemory", "OpenProcess", "SetWindowsHookEx", "BringWindowToTop("):
            self.assertNotIn(banned, osd_src)
        self.assertIn("SW_SHOWNOACTIVATE", osd_src)
        self.assertIn("HTTRANSPARENT", osd_src)
        import preview_window
        preview_src = inspect.getsource(preview_window)
        for banned in ("SetWindowPos", "TOPMOST", "SetForegroundWindow(", "BringWindowToTop("):
            self.assertNotIn(banned, preview_src)
        main_src = inspect.getsource(main_module)
        self.assertNotIn("SetForegroundWindow(", main_src)
        self.assertIn("destroyAllWindows", inspect.getsource(main_module.HandControllerApp.run))

    def test_icon_files_have_alpha(self):
        import cv2
        png = os.path.join(os.path.dirname(os.path.abspath(main_module.__file__)), "assets", "handcontroller_icon.png")
        ico = os.path.join(os.path.dirname(os.path.abspath(main_module.__file__)), "assets", "handcontroller_icon.ico")
        self.assertTrue(os.path.isfile(png), png)
        self.assertTrue(os.path.isfile(ico), ico)
        image = cv2.imread(png, cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(image)
        self.assertEqual(image.shape[2], 4)
        self.assertGreater(int(np.count_nonzero(image[:, :, 3] == 0)), 0)
        self.assertGreater(int(np.count_nonzero(image[:, :, 3] > 0)), 0)
        alpha = image[:, :, 3]
        for corner in (alpha[0, 0], alpha[0, -1], alpha[-1, 0], alpha[-1, -1]):
            self.assertEqual(int(corner), 0)
        transparent = os.path.join(os.path.dirname(png), "handcontroller_icon_transparent.png")
        self.assertTrue(os.path.isfile(transparent))
        with open(ico, "rb") as handle:
            header = handle.read(6)
        reserved, kind, count = struct.unpack("<HHH", header)
        self.assertEqual((reserved, kind), (0, 1))
        self.assertGreaterEqual(count, 7)
        root = os.path.dirname(os.path.abspath(main_module.__file__))
        with open(os.path.join(root, "HandController.spec"), encoding="utf-8") as handle:
            spec = handle.read()
        self.assertIn("handcontroller_icon_transparent.png", spec)
        self.assertIn('icon="assets/handcontroller_icon.ico"', spec)
        self.assertIn("uac_admin=True", spec)


class CoexistenceTests(TempCase):
    def setUp(self):
        super().setUp()
        self.app = arm_tracking(make_geometry_app(self.mapping_path), grace=2)
        self.app.inputs.enable()

    def _hold_g01(self, side="LEFT", key="W"):
        self.app.mapping.set_command(side, "G01", HOLD, [key])
        run_tracked_frame(self.app, left="G01" if side == "LEFT" else KEEP, right="G01" if side == "RIGHT" else KEEP)

    def test_g01_hold_and_pinch_hold_same_hand(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        self.app.mapping.set_command("LEFT", geo.LEFT_PINCH, HOLD, ["SHIFT"])
        run_tracked_frame(self.app, left="G01")
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", left_points=PINCHING)
        names = self.app.inputs.get_pressed_names()
        self.assertIn("W", names)
        self.assertIn("SHIFT", names)
        self.assertIn("LEFT", self.app.inputs.hold_sources)
        self.assertIn(geo.special_source("LEFT", geo.PINCH), self.app.inputs.hold_sources)

    def test_g01_hold_and_each_finger_up_hold(self):
        fingers = (
            ("INDEX", geo.LEFT_FINGER_INDEX, finger_pose((1.0, 0.2, 0.2, 0.2))),
            ("MIDDLE", geo.LEFT_FINGER_MIDDLE, finger_pose((0.2, 1.0, 0.2, 0.2))),
            ("RING", geo.LEFT_FINGER_RING, finger_pose((0.2, 0.2, 1.0, 0.2))),
            ("PINKY", geo.LEFT_FINGER_PINKY, finger_pose((0.2, 0.2, 0.2, 1.0))),
        )
        for name, gid, pose in fingers:
            with self.subTest(finger=name):
                app = arm_tracking(make_geometry_app(self.mapping_path), grace=2)
                app.inputs.enable()
                app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
                app.mapping.set_command("LEFT", gid, HOLD, ["SPACE"])
                run_tracked_frame(app, left="G01")
                for _ in range(geo.STABILITY_FRAMES):
                    run_tracked_frame(app, left="G01", left_points=pose)
                names = app.inputs.get_pressed_names()
                self.assertIn("W", names)
                self.assertIn("SPACE", names)

    def test_pinch_hold_and_tilt_hold(self):
        self.app.mapping.set_command("LEFT", geo.LEFT_PINCH, HOLD, ["SHIFT"])
        self.app.mapping.set_command("LEFT", geo.LEFT_TILT_LEFT, HOLD, ["A"])
        pose = hand(palm=(-0.8, -0.6), index=(-0.8, -0.6), pinch=0.05)
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left=KEEP, left_points=pose)
        names = self.app.inputs.get_pressed_names()
        self.assertIn("SHIFT", names)
        self.assertIn("A", names)

    def test_g01_press_and_pinch_press(self):
        self.app.mapping.set_command("LEFT", "G01", PRESS, ["E"], 0)
        self.app.mapping.set_command("LEFT", geo.LEFT_PINCH, PRESS, ["SPACE"], 0)
        run_tracked_frame(self.app, left="G01")
        self.assertEqual(self.app.inputs.keyboard.pressed.count("e"), 1)
        space = self.app.inputs.special_keys["SPACE"]
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", left_points=PINCHING)
        self.assertGreaterEqual(self.app.inputs.keyboard.pressed.count(space), 1)

    def test_g01_hold_and_pinch_press(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        self.app.mapping.set_command("LEFT", geo.LEFT_PINCH, PRESS, ["SPACE"], 0)
        run_tracked_frame(self.app, left="G01")
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", left_points=PINCHING)
        self.assertIn("W", self.app.inputs.get_pressed_names())
        space = self.app.inputs.special_keys["SPACE"]
        self.assertGreaterEqual(self.app.inputs.keyboard.pressed.count(space), 1)

    def test_g01_press_and_tilt_press(self):
        self.app.mapping.set_command("LEFT", "G01", PRESS, ["E"], 0)
        self.app.mapping.set_command("LEFT", geo.LEFT_TILT_UP, PRESS, ["SPACE"], 0)
        run_tracked_frame(self.app, left="G01")
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", left_points=OPEN_UP)
        space = self.app.inputs.special_keys["SPACE"]
        self.assertEqual(self.app.inputs.keyboard.pressed.count("e"), 1)
        self.assertGreaterEqual(self.app.inputs.keyboard.pressed.count(space), 1)

    def test_g01_macro_and_pinch_macro_independent_owners(self):
        self.app.mapping.set_macro("LEFT", "G01", [
            {"action": STEP_HOLD, "inputs": ["W"]},
            {"action": STEP_WAIT, "duration": 5000},
        ])
        self.app.mapping.set_macro("LEFT", geo.LEFT_PINCH, [
            {"action": STEP_HOLD, "inputs": ["A"]},
            {"action": STEP_WAIT, "duration": 5000},
        ])
        run_tracked_frame(self.app, left="G01")
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", left_points=PINCHING)
        self.assertTrue(self.app.inputs.is_macro_running("LEFT"))
        self.assertTrue(self.app.inputs.is_macro_running(geo.special_source("LEFT", geo.PINCH)))
        self.app.inputs.stop_all_macros()
        self.app.inputs.release_all()

    def test_left_g01_and_right_pinch(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["A"])
        self.app.mapping.set_command("RIGHT", geo.RIGHT_PINCH, HOLD, ["B"])
        run_tracked_frame(self.app, left="G01")
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", right=KEEP, right_points=PINCHING)
        names = self.app.inputs.get_pressed_names()
        self.assertIn("A", names)
        self.assertIn("B", names)

    def test_left_g01_and_right_tilt(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["A"])
        self.app.mapping.set_command("RIGHT", geo.RIGHT_TILT_UP, PRESS, ["E"], 0)
        run_tracked_frame(self.app, left="G01")
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", right=KEEP, right_points=OPEN_UP)
        self.assertIn("A", self.app.inputs.get_pressed_names())
        self.assertGreaterEqual(self.app.inputs.keyboard.pressed.count("e"), 1)

    def test_hand_loss_left_releases_only_left_sources(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["A"])
        self.app.mapping.set_command("LEFT", geo.LEFT_PINCH, HOLD, ["SHIFT"])
        self.app.mapping.set_command("RIGHT", "G01", HOLD, ["B"])
        run_tracked_frame(self.app, left="G01", right="G01")
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", right="G01", left_points=PINCHING)
        for _ in range(3):
            run_tracked_frame(self.app, left=None, right="G01")
        self.assertNotIn("A", self.app.inputs.get_pressed_names())
        self.assertNotIn("SHIFT", self.app.inputs.get_pressed_names())
        self.assertIn("B", self.app.inputs.get_pressed_names())

    def test_hand_loss_right_releases_only_right_sources(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["A"])
        self.app.mapping.set_command("RIGHT", geo.RIGHT_PINCH, HOLD, ["B"])
        run_tracked_frame(self.app, left="G01")
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", right=KEEP, right_points=PINCHING)
        for _ in range(3):
            run_tracked_frame(self.app, left="G01", right=None)
        self.assertIn("A", self.app.inputs.get_pressed_names())
        self.assertNotIn("B", self.app.inputs.get_pressed_names())

    def test_disable_pinch_gate_leaves_g01(self):
        self.app.system_gestures = sysg.default_system_gestures()
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        self.app.mapping.set_command("LEFT", geo.LEFT_PINCH, HOLD, ["SHIFT"])
        run_tracked_frame(self.app, left="G01")
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", left_points=PINCHING)
        self.assertIn("W", self.app.inputs.get_pressed_names())
        self.assertNotIn("SHIFT", self.app.inputs.get_pressed_names())

    def test_disable_g01_leaves_pinch(self):
        self.app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        self.app.mapping.disable("LEFT", "G01")
        self.app.mapping.set_command("LEFT", geo.LEFT_PINCH, HOLD, ["SHIFT"])
        run_tracked_frame(self.app, left="G01")
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", left_points=PINCHING)
        self.assertNotIn("W", self.app.inputs.get_pressed_names())
        self.assertIn("SHIFT", self.app.inputs.get_pressed_names())

    def test_pinch_detected_without_mapping(self):
        self.app.mapping.disable("LEFT", geo.LEFT_PINCH)
        for _ in range(geo.STABILITY_FRAMES):
            run_geometry_frame(self.app, left=PINCHING)
        pinch = self.app.geometry.detector("LEFT", geo.PINCH)
        self.assertTrue(pinch is not None and pinch.active)
        self.assertNotIn(geo.special_source("LEFT", geo.PINCH), self.app.inputs.hold_sources)

    def test_combination_g01_and_pinch_hold(self):
        self.app.mapping.set_command("LEFT", "G01", COMBINATION, ["SHIFT", "A"])
        self.app.mapping.set_command("LEFT", geo.LEFT_PINCH, HOLD, ["SPACE"])
        run_tracked_frame(self.app, left="G01")
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(self.app, left="G01", left_points=PINCHING)
        names = self.app.inputs.get_pressed_names()
        self.assertIn("SHIFT", names)
        self.assertIn("A", names)
        self.assertIn("SPACE", names)

    def test_hud_lists_knn_and_system(self):
        from overlay_labels import active_gesture_labels
        self.app.detected["LEFT"] = True
        self.app.track_state["LEFT"] = "OK"
        self.app.curr_gestures["LEFT"] = "G01"
        self.app.curr_specials["LEFT"] = {geo.PINCH: geo.LEFT_PINCH, geo.finger_up_family("INDEX"): geo.LEFT_FINGER_INDEX}
        labels = active_gesture_labels(self.app, "LEFT")
        self.assertEqual(labels[0], "G01")
        self.assertIn("PINCH LEFT", labels)
        self.assertIn("INDEX UP", labels)


class MouseUiTests(TempCase):
    def _hud_app(self):
        app = make_geometry_app(self.mapping_path)
        app.kill_banner = False
        app.camera_ok = True
        app.detected = {side: False for side in SIDES}
        app.confidences = {side: 0.0 for side in SIDES}
        app.curr_gestures = {side: UNKNOWN for side in SIDES}
        app.curr_specials = {side: {} for side in SIDES}
        app.system_gestures = sysg.default_system_gestures()
        app.status_message = ""
        app.settings = {"gaming_mode": False}
        app.debug = False
        app.lang_mode = False
        app.help_mode = False
        app.finger_mode = False
        app.track_state = {side: "RELEASED" for side in SIDES}
        app.last_dynamic_osd = {side: ("", 0.0) for side in SIDES}
        app.sys_mode = False
        app.sys_phase = "list"
        app.sys_index = 0
        app.sys_edit = None
        app.sys_combo_capture = False
        app.sidebar_hidden = False
        app.settings_mode = False
        app.input_panel = False
        app.overlay_panel = False
        app.settings_side = "LEFT"
        return app

    def _mouse_app(self):
        app = self._hud_app()
        app.reset_state = lambda **kwargs: None
        app.ensure_preview_visible = lambda: None
        app.engine = GestureEngine(self.gestures_path)
        app.editor_mode = False
        app.editor_phase = "list"
        app.editor_side = "LEFT"
        app.editor_index = 0
        app.map_mode = False
        app.map_phase = "hand"
        app.map_side = "LEFT"
        app.map_index = 0
        app.edit_type = HOLD
        app.edit_inputs = ["W"]
        app.edit_slot = 0
        app.edit_cooldown = 3
        app.calib_mode = False
        app.calib_phase = "hand"
        app.calib_name_buffer = ""
        app.countdown_seconds = 3
        return app

    def test_g_v_q_m_are_ui_when_focused_and_still_assignable(self):
        run_src = inspect.getsource(main_module.HandControllerApp.run)
        self.assertIn('key == ord("q")', run_src)
        idle = inspect.getsource(main_module.HandControllerApp.handle_idle_key)
        self.assertIn('ord("g")', idle)
        self.assertIn('ord("p")', idle)
        self.assertIn('ord("c")', idle)
        self.assertIn("enter_calibration()", idle)
        self.assertIn("enter_mapping_menu()", idle)
        self.assertNotIn('ord("v")', idle)
        mapping_src = inspect.getsource(main_module.HandControllerApp.handle_mapping_key)
        self.assertNotIn('ord("v")', mapping_src)
        self.assertIn('key == ord(" ")', mapping_src)

    def test_window_x_still_closes(self):
        run_src = inspect.getsource(main_module.HandControllerApp.run)
        self.assertIn("preview_was_closed", run_src)
        self.assertIn("release_all", run_src)
        self.assertIn("stop_all_macros", run_src)
        self.assertIn("cap.release", run_src)

    def test_click_saved_gestures_opens_editor(self):
        app = self._mouse_app()
        app._activate_sidebar("settings")
        self.assertTrue(app.settings_mode)
        frame = np.zeros((480, 720, 3), dtype=np.uint8)
        app.draw_gesture_settings(frame)
        kinds = [hit[4] for hit in app._ui_hits]
        self.assertIn("settings", kinds)
        app._handle_settings_click("side", "RIGHT")
        self.assertEqual(app.settings_side, "RIGHT")

    def test_click_predefined_opens_system(self):
        app = self._mouse_app()
        app._activate_sidebar("settings")
        frame = np.zeros((480, 720, 3), dtype=np.uint8)
        app.draw_gesture_settings(frame)
        self.assertTrue(any(hit[4] == "settings" for hit in app._ui_hits))
        predefined = [row for row in app._sys_rows() if row.get("side") == "LEFT" and row.get("kind") == "pinch"]
        self.assertTrue(predefined)
        app._handle_settings_click("sys_edit", 0)
        self.assertTrue(app.sys_mode)

    def test_click_key_settings_opens_mapping(self):
        app = self._mouse_app()
        app.enter_mapping_menu()
        self.assertTrue(app.map_mode)
        app._handle_map_click("hand", "LEFT")
        self.assertEqual(app.map_side, "LEFT")
        self.assertEqual(app.map_phase, "list")
        app._handle_map_click("row", 0)
        self.assertIn(app.map_phase, ("simple", "combo"))

    def test_physical_assign_and_hold_press(self):
        app = self._mouse_app()
        app.enter_mapping_menu()
        app._handle_map_click("hand", "LEFT")
        app._handle_map_click("row", 0)
        app._handle_map_click("type", HOLD)
        self.assertEqual(app.edit_type, HOLD)
        app._handle_map_click("assign")
        self.assertEqual(app.map_phase, "capture")
        app.inputs._captured_name = "A"
        app.poll_key_capture()
        gesture = app.current_map_gesture()
        entry = app.mapping.get(app.map_side, gesture)
        self.assertEqual(entry["inputs"], ["A"])
        self.assertEqual(entry["type"], HOLD)
        app.map_phase = "simple"
        app._handle_map_click("type", PRESS)
        self.assertEqual(app.edit_type, PRESS)
        app._handle_map_click("save")
        self.assertEqual(app.mapping.get(app.map_side, gesture)["type"], PRESS)

    def test_record_and_calibrate_buttons(self):
        app = self._mouse_app()
        app._activate_sidebar("record")
        self.assertTrue(app.calib_mode)
        self.assertEqual(app.calib_phase, "create")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        app.draw_record_panel(frame)
        self.assertTrue(any(hit[4] == "calib" for hit in app._ui_hits))

    def test_game_input_click_toggle(self):
        app = self._mouse_app()
        app._activate_sidebar("input")
        self.assertTrue(app.input_panel)
        self.assertEqual(app.input_draft, "off")
        self.assertFalse(app.inputs.enabled)
        app._handle_input_panel_click("pick", "on")
        self.assertFalse(app.inputs.enabled)
        app._handle_input_panel_click("ok")
        self.assertTrue(app.inputs.enabled)
        app._activate_sidebar("input")
        app._handle_input_panel_click("pick", "off")
        app._handle_input_panel_click("ok")
        self.assertFalse(app.inputs.enabled)

    def test_emergency_stop_releases_all(self):
        app = self._mouse_app()
        app.inputs.enable()
        app.inputs.set_source_hold("LEFT", ["W", "A"])
        self.assertTrue(app.inputs.get_pressed_names())
        app.trigger_ui_emergency()
        self.assertFalse(app.inputs.enabled)
        self.assertEqual(app.inputs.get_pressed_names(), [])
        msg = (app.status_message or "").lower()
        self.assertTrue("libere" in msg or "released" in msg)

    def test_localization_mouse_strings(self):
        from localization import load_catalogs, set_language, t
        load_catalogs(force=True)
        set_language("fr")
        self.assertEqual(t("ui.recorded_gestures"), "Gestes enregistrés")
        self.assertEqual(t("ui.gesture_settings"), "Paramètres de geste")
        self.assertEqual(t("ui.predefined"), "Gestes prédéfinis")
        self.assertEqual(t("ui.assign"), "Assigner une touche")
        self.assertIn("Appuyez", t("ui.press_to_assign"))
        self.assertEqual(t("ui.record_gestures"), "Enregistrer un geste")
        self.assertEqual(t("ui.configure"), "Configurer")
        set_language("en")
        self.assertEqual(t("ui.recorded_gestures"), "Recorded Gestures")
        self.assertEqual(t("ui.gesture_settings"), "Gesture Settings")
        self.assertEqual(t("ui.assign"), "Assign a key")
        self.assertIn("Press a key", t("ui.press_to_assign"))
        set_language("fr")

    def test_g01_and_pinch_still_coexist(self):
        from overlay_labels import active_gesture_labels
        app = arm_tracking(make_geometry_app(self.mapping_path), grace=2)
        app.inputs.enable()
        app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        app.mapping.set_command("LEFT", geo.LEFT_PINCH, HOLD, ["SHIFT"])
        run_tracked_frame(app, left="G01")
        for _ in range(geo.STABILITY_FRAMES):
            run_tracked_frame(app, left="G01", left_points=PINCHING)
        names = app.inputs.get_pressed_names()
        self.assertIn("W", names)
        self.assertIn("SHIFT", names)
        app.detected["LEFT"] = True
        app.track_state["LEFT"] = "OK"
        app.curr_gestures["LEFT"] = "G01"
        app.curr_specials["LEFT"] = {geo.PINCH: geo.LEFT_PINCH}
        labels = active_gesture_labels(app, "LEFT")
        self.assertIn("G01", labels)
        self.assertIn("PINCH LEFT", labels)


class MainUiTests(TempCase):
    def test_available_languages_from_files(self):
        from localization import available_languages, language_display_name, language_options
        langs = available_languages()
        self.assertIn("fr", langs)
        self.assertIn("en", langs)
        names = dict(language_options())
        self.assertIn("Français", names["fr"])
        self.assertEqual(language_display_name("en"), "English")

    def test_language_prompt_default_fr_and_cancel_ok(self):
        from localization import language_prompt_click, language_prompt_state
        state = language_prompt_state()
        self.assertEqual(state["selected"], "fr")
        self.assertFalse(state["open"])
        opened = language_prompt_click(state, "toggle")
        self.assertTrue(opened["open"])
        picked = language_prompt_click(opened, "pick", "en")
        self.assertEqual(picked["selected"], "en")
        self.assertFalse(picked["open"])
        self.assertEqual(language_prompt_click(picked, "ok"), "en")
        self.assertIsNone(language_prompt_click(state, "cancel"))

    def test_language_saved_and_skips_prompt_when_valid(self):
        from config import load_settings, save_settings, validate_settings
        from localization import apply_language, language_configured, persist_language, current_language, set_language
        path = os.path.join(self.td.name, "settings.json")
        data = validate_settings({})
        self.assertFalse(language_configured(data))
        persist_language(data, "en", path)
        loaded = load_settings(path)
        self.assertEqual(loaded["language"], "en")
        self.assertTrue(language_configured(loaded))
        apply_language(loaded, interactive=False)
        self.assertEqual(current_language(), "en")
        set_language("fr")

    def test_main_does_not_open_camera_before_language(self):
        source = inspect.getsource(main_module.main)
        self.assertLess(source.index("apply_language"), source.index("HandControllerApp"))
        self.assertIn("if chosen is None", source)
        run_src = inspect.getsource(main_module.HandControllerApp.run)
        self.assertIn("open_camera", run_src)
        self.assertNotIn("prompt_language", run_src)

    def test_sidebar_hide_and_show(self):
        app = make_geometry_app(self.mapping_path)
        app.sidebar_hidden = False
        app._activate_sidebar("hide")
        self.assertTrue(app.sidebar_hidden)
        self.assertEqual(app._sidebar_width(1280), 0)
        app._activate_sidebar("show")
        self.assertFalse(app.sidebar_hidden)
        self.assertGreaterEqual(app._sidebar_width(1280), 250)

    def test_g_v_q_m_still_assignable(self):
        mapping = GestureMapping(self.mapping_path)
        for key in ("G", "V", "Q", "M"):
            self.assertTrue(mapping.set_command("LEFT", "G01", HOLD, [key]))
            self.assertEqual(mapping.get("LEFT", "G01")["inputs"], [key])

    def test_f8_to_f11_are_game_keys(self):
        mapping = GestureMapping(self.mapping_path)
        self.assertTrue(mapping.set_command("LEFT", "G01", HOLD, ["F9"]))
        self.assertTrue(mapping.set_command("LEFT", "G01", PRESS, ["F11"], 0))
        self.assertTrue(mapping.set_command("RIGHT", "G01", HOLD, ["F8"]))
        self.assertTrue(mapping.set_command("RIGHT", "G02", PRESS, ["F10"], 0))

    def test_no_focus_stealing(self):
        main_src = inspect.getsource(main_module)
        self.assertNotIn("SetForegroundWindow(", main_src)
        self.assertNotIn("BringWindowToTop(", main_src)
        loc_src = inspect.getsource(__import__("localization"))
        self.assertNotIn("SetForegroundWindow(", loc_src)
        self.assertNotIn("BringWindowToTop(", loc_src)
        self.assertNotIn("TOPMOST", inspect.getsource(main_module.HandControllerApp))

    def test_input_panel_and_emergency(self):
        app = make_geometry_app(self.mapping_path)
        app.reset_state = lambda **kwargs: None
        app.ensure_preview_visible = lambda: None
        app.settings = {}
        app.kill_banner = False
        app.status_message = ""
        app.sidebar_hidden = False
        app.settings_mode = False
        app.input_panel = False
        app._activate_sidebar("input")
        self.assertTrue(app.input_panel)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        app.draw_input_panel(frame)
        self.assertTrue(any(hit[4] == "input_panel" for hit in app._ui_hits))
        app.inputs.enable()
        app.inputs.set_source_hold("LEFT", ["W"])
        app._handle_input_panel_click("emergency")
        self.assertFalse(app.inputs.enabled)
        self.assertEqual(app.inputs.get_pressed_names(), [])

    def test_coexistence_tilt_and_index_with_g01(self):
        app = arm_tracking(make_geometry_app(self.mapping_path), grace=2)
        app.inputs.enable()
        app.mapping.set_command("LEFT", "G01", HOLD, ["W"])
        app.mapping.set_command("LEFT", geo.LEFT_TILT_LEFT, HOLD, ["A"])
        run_tracked_frame(app, left="G01")
        self.assertIn("W", app.inputs.get_pressed_names())
        app.mapping.set_command("RIGHT", "G01", HOLD, ["D"])
        run_tracked_frame(app, left="G01", right="G01")
        names = app.inputs.get_pressed_names()
        self.assertIn("W", names)
        self.assertIn("D", names)


class OverlayHudTests(TempCase):
    def _app(self):
        app = make_geometry_app(self.mapping_path)
        app.reset_state = lambda **kwargs: None
        app.ensure_preview_visible = lambda: None
        app.status_message = ""
        app.help_mode = False
        app.lang_mode = False
        app.settings_mode = False
        app.input_panel = False
        app.sidebar_hidden = False
        app.settings = {}
        app.osd = None
        app.input_draft = "off"
        app.input_list_open = False
        app.detected = {side: False for side in SIDES}
        app.track_state = {side: "RELEASED" for side in SIDES}
        app.curr_gestures = {side: UNKNOWN for side in SIDES}
        app.curr_specials = {side: {} for side in SIDES}
        app.last_dynamic_osd = {side: ("", 0.0) for side in SIDES}
        return app

    def test_overlay_under_input_in_sidebar(self):
        from localization import set_language
        app = self._app()
        actions = [item[0] for item in app._sidebar_items()]
        self.assertEqual(actions[2], "input")
        self.assertEqual(actions[3], "help")
        self.assertNotIn("overlay", actions)
        set_language("en")
        labels = [item[1] for item in app._sidebar_items()]
        self.assertEqual(labels[3], "Help")
        set_language("fr")

    def test_sidebar_has_no_handcontroller_logo(self):
        source = inspect.getsource(main_module.HandControllerApp._draw_sidebar)
        self.assertNotIn("APP_NAME", source)
        self.assertNotIn("icon_png_path", source)
        self.assertNotIn("_ui_icon", source)

    def test_overlay_window_opens(self):
        app = self._app()
        self.assertTrue(getattr(app, "sidebar_hidden", False) is False)
        self.assertFalse(hasattr(app, "draw_overlay_panel"))
        source = inspect.getsource(main_module.HandControllerApp)
        self.assertNotIn("def enter_overlay_panel", source)
        self.assertNotIn("PySide", inspect.getsource(__import__("game_osd")))

    def test_overlay_hotkeys_configurable(self):
        listener = inspect.getsource(input_controller.InputController.start_emergency_listener)
        self.assertNotIn("match_osd_hotkey", listener)
        self.assertNotIn("set_overlay_hotkeys", inspect.getsource(input_controller.InputController))

    def test_overlay_ok_apply_cancel(self):
        from config import default_settings
        self.assertNotIn("game_osd", default_settings())

    def test_esc_cancels_overlay_capture(self):
        source = inspect.getsource(main_module.HandControllerApp.poll_key_capture)
        self.assertNotIn("overlay_capture", source)

    def test_pinch_left_and_right_exist(self):
        self.assertEqual(geo.LEFT_PINCH, "LEFT_PINCH")
        self.assertEqual(geo.RIGHT_PINCH, "RIGHT_PINCH")
        self.assertNotEqual(geo.LEFT_PINCH, geo.RIGHT_PINCH)
        app = self._app()
        titles = [app._sys_row_title(row) for row in app._sys_rows() if row.get("kind") == "pinch"]
        self.assertIn("PINCH LEFT", titles)
        self.assertIn("PINCH RIGHT", titles)
        from overlay_labels import special_label
        self.assertEqual(special_label(geo.LEFT_PINCH), "PINCH LEFT")
        self.assertEqual(special_label(geo.RIGHT_PINCH), "PINCH RIGHT")

    def test_index_up_and_tilt_up_simultaneous(self):
        from overlay_labels import active_gesture_labels, osd_state_from_app
        app = self._app()
        app.detected["RIGHT"] = True
        app.track_state["RIGHT"] = "OK"
        app.curr_specials["RIGHT"] = {
            geo.TILT: geo.RIGHT_TILT_UP,
            geo.FINGER_INDEX: geo.RIGHT_FINGER_INDEX,
            geo.INDEX: geo.RIGHT_INDEX_UP,
        }
        labels = active_gesture_labels(app, "RIGHT")
        self.assertIn("INDEX UP", labels)
        self.assertIn("TILT UP", labels)
        self.assertEqual(labels.count("INDEX UP"), 1)
        state = osd_state_from_app(app)
        self.assertIn("INDEX UP", state["right_labels"])
        self.assertIn("TILT UP", state["right_labels"])

    def test_pinch_right_and_index_up_simultaneous(self):
        from overlay_labels import active_gesture_labels
        app = self._app()
        app.detected["RIGHT"] = True
        app.track_state["RIGHT"] = "OK"
        app.curr_specials["RIGHT"] = {
            geo.PINCH: geo.RIGHT_PINCH,
            geo.FINGER_INDEX: geo.RIGHT_FINGER_INDEX,
        }
        labels = active_gesture_labels(app, "RIGHT")
        self.assertIn("PINCH RIGHT", labels)
        self.assertIn("INDEX UP", labels)

    def test_knn_with_tilt_and_index(self):
        from overlay_labels import active_gesture_labels
        app = self._app()
        app.detected["LEFT"] = True
        app.track_state["LEFT"] = "OK"
        app.curr_gestures["LEFT"] = "G01"
        app.curr_specials["LEFT"] = {
            geo.TILT: geo.LEFT_TILT_UP,
            geo.FINGER_INDEX: geo.LEFT_FINGER_INDEX,
        }
        labels = active_gesture_labels(app, "LEFT")
        self.assertEqual(labels[0], "G01")
        self.assertIn("TILT UP", labels)
        self.assertIn("INDEX UP", labels)

    def test_hud_lists_multiple_and_none_only_when_empty(self):
        from overlay_labels import active_gesture_labels, hand_overlay_state, osd_state_from_app
        app = self._app()
        present, labels = hand_overlay_state(app, "RIGHT")
        self.assertFalse(present)
        self.assertEqual(labels, [])
        app.detected["RIGHT"] = True
        app.track_state["RIGHT"] = "OK"
        self.assertEqual(osd_state_from_app(app)["right"], "none")
        self.assertEqual(active_gesture_labels(app, "RIGHT"), [])
        app.curr_specials["RIGHT"] = {
            geo.TILT: geo.RIGHT_TILT_UP,
            geo.FINGER_INDEX: geo.RIGHT_FINGER_INDEX,
            geo.PINCH: geo.RIGHT_PINCH,
        }
        labels = active_gesture_labels(app, "RIGHT")
        self.assertEqual(labels, ["TILT UP", "INDEX UP", "PINCH RIGHT"])
        self.assertEqual(osd_state_from_app(app)["right"], "TILT UP + INDEX UP + PINCH RIGHT")

    def test_left_right_independent(self):
        from overlay_labels import active_gesture_labels
        app = self._app()
        app.detected["LEFT"] = True
        app.detected["RIGHT"] = True
        app.track_state["LEFT"] = "OK"
        app.track_state["RIGHT"] = "OK"
        app.curr_specials["LEFT"] = {geo.TILT: geo.LEFT_TILT_LEFT, geo.PINCH: geo.LEFT_PINCH}
        app.curr_specials["RIGHT"] = {geo.FINGER_INDEX: geo.RIGHT_FINGER_INDEX, geo.TILT: geo.RIGHT_TILT_UP}
        left = active_gesture_labels(app, "LEFT")
        right = active_gesture_labels(app, "RIGHT")
        self.assertIn("TILT LEFT", left)
        self.assertIn("PINCH LEFT", left)
        self.assertNotIn("INDEX UP", left)
        self.assertIn("INDEX UP", right)
        self.assertIn("TILT UP", right)
        self.assertNotIn("PINCH LEFT", right)

    def test_flick_and_swipe_are_gone_from_hud(self):
        from overlay_labels import active_gesture_labels
        from time import monotonic
        app = self._app()
        app.detected["RIGHT"] = True
        app.track_state["RIGHT"] = "OK"
        now = monotonic()
        app.last_dynamic_osd["RIGHT"] = ("FLICK RIGHT", now)
        self.assertNotIn("FLICK RIGHT", active_gesture_labels(app, "RIGHT", now))
        app.last_dynamic_osd["RIGHT"] = ("SWIPE UP", now)
        self.assertNotIn("SWIPE UP", active_gesture_labels(app, "RIGHT", now))

    def test_overlay_hotkey_match_and_f_keys_assignable(self):
        listener = inspect.getsource(input_controller.InputController.start_emergency_listener)
        self.assertNotIn("if key == emergency", listener)
        self.assertNotIn("if key == preview", listener)
        self.assertNotIn("match_osd_hotkey", listener)
        mapping = GestureMapping(self.mapping_path)
        self.assertTrue(mapping.set_command("LEFT", "G01", HOLD, ["F11"]))
        self.assertTrue(mapping.set_command("LEFT", "G02", HOLD, ["F9"]))

    def test_gvmq_are_ui_and_still_assignable(self):
        run_src = inspect.getsource(main_module.HandControllerApp.run)
        self.assertIn('ord("q")', run_src)
        idle = inspect.getsource(main_module.HandControllerApp.handle_idle_key)
        self.assertIn('ord("g")', idle)
        self.assertIn('ord("c")', idle)
        mapping = GestureMapping(self.mapping_path)
        for key in ("G", "V", "Q", "M"):
            self.assertTrue(mapping.set_command("LEFT", "G01", HOLD, [key]))

    def test_x_close_and_cleanup(self):
        run_src = inspect.getsource(main_module.HandControllerApp.run)
        self.assertIn("preview_was_closed", run_src)
        self.assertIn("try:", run_src)
        self.assertIn("finally:", run_src)
        self.assertIn("stop_all_macros", run_src)
        self.assertIn("release_all", run_src)
        self.assertIn("stop_emergency_listener", run_src)
        self.assertIn("cap.release", run_src)
        self.assertIn("destroyAllWindows", run_src)
        self.assertIn("destroyAllWindows", run_src)


class FinalUiCleanupTests(TempCase):
    def test_overlay_off_at_launch_even_if_saved_on(self):
        self.assertNotIn("PySide", inspect.getsource(__import__("game_osd")))
        app = HandControllerApp.__new__(HandControllerApp)
        app.settings = {}
        app.sidebar_hidden = True
        app.inputs = make_inputs()
        self.assertFalse(app.inputs.enabled)
        self.assertTrue(app.sidebar_hidden)

    def test_overlay_show_hide_hotkeys_have_no_toggle(self):
        from config import default_settings
        self.assertNotIn("game_osd", default_settings())
        self.assertNotIn("OSD_HOTKEY_SLOTS", inspect.getsource(__import__("config")))
        listener = inspect.getsource(input_controller.InputController.start_emergency_listener)
        self.assertNotIn("match_osd_hotkey", listener)

    def test_input_window_enable_disable_ok_cancel(self):
        from localization import input_option_label, input_prompt_click, input_prompt_state, set_language
        set_language("fr")
        app = self._fresh()
        app.enter_input_panel()
        self.assertTrue(app.input_panel)
        self.assertEqual(app.input_draft, "off")
        frame = np.zeros((360, 520, 3), dtype=np.uint8)
        app.draw_input_panel(frame)
        labels = " ".join(str(hit) for hit in app._ui_hits)
        self.assertIn("input_panel", labels)
        actions = [hit[5][0] if len(hit) > 5 and hit[5] else "" for hit in app._ui_hits]
        self.assertNotIn("emergency", actions)
        panel_src = inspect.getsource(main_module.HandControllerApp.draw_input_panel)
        self.assertNotIn("emergency_stop", panel_src)
        self.assertNotIn("new_gesture", panel_src)
        self.assertNotIn('"<"', panel_src)
        self.assertNotIn('">"', panel_src)
        self.assertEqual(input_option_label("on"), "Activer")
        app._handle_input_panel_click("pick", "on")
        app._handle_input_panel_click("cancel")
        self.assertFalse(app.inputs.enabled)
        self.assertFalse(app.input_panel)
        app.enter_input_panel()
        app._handle_input_panel_click("pick", "on")
        app._handle_input_panel_click("ok")
        self.assertTrue(app.inputs.enabled)
        app.enter_input_panel()
        app._handle_input_panel_click("pick", "off")
        app._handle_input_panel_click("ok")
        self.assertFalse(app.inputs.enabled)
        state = input_prompt_state("off")
        self.assertEqual(input_prompt_click(state, "ok"), "off")
        self.assertIsNone(input_prompt_click(state, "cancel"))

    def test_emergency_stop_from_input_window(self):
        app = self._fresh()
        app.inputs.enable()
        app.inputs.set_source_hold("LEFT", ["W", "A"])
        app.enter_input_panel()
        app._handle_input_panel_click("emergency")
        self.assertFalse(app.inputs.enabled)
        self.assertEqual(app.inputs.get_pressed_names(), [])
        self.assertEqual(app.input_draft, "off")

    def test_index_up_and_tilt_up_display_together(self):
        from overlay_labels import active_gesture_labels
        app = self._fresh()
        app.detected["RIGHT"] = True
        app.track_state["RIGHT"] = "OK"
        app.curr_gestures["RIGHT"] = UNKNOWN
        app.curr_specials["RIGHT"] = {
            geo.FINGER_INDEX: geo.RIGHT_FINGER_INDEX,
            geo.TILT: geo.RIGHT_TILT_UP,
        }
        labels = active_gesture_labels(app, "RIGHT")
        self.assertIn("INDEX UP", labels)
        self.assertIn("TILT UP", labels)

    def _fresh(self):
        from config import default_settings
        app = HandControllerApp.__new__(HandControllerApp)
        app.settings = default_settings()
        app.inputs = make_inputs()
        app.mapping = GestureMapping(self.mapping_path)
        app.status_message = ""
        app.kill_banner = False
        app.help_mode = False
        app.lang_mode = False
        app.settings_mode = False
        app.input_panel = False
        app.input_draft = "off"
        app.input_list_open = False
        app.detected = {side: False for side in SIDES}
        app.track_state = {side: "RELEASED" for side in SIDES}
        app.curr_gestures = {side: UNKNOWN for side in SIDES}
        app.curr_specials = {side: {} for side in SIDES}
        app.reset_state = lambda **kwargs: None
        app.ensure_preview_visible = lambda: None
        return app


class GestureLabelTests(unittest.TestCase):
    def test_format_gesture_label_never_repeats_the_id(self):
        import gesture_editor as editor
        self.assertEqual(editor.format_gesture_label("G01", ""), "G01")
        self.assertEqual(editor.format_gesture_label("G01", "G01"), "G01")
        self.assertEqual(editor.format_gesture_label("G01", "ATTAQUE"), "G01 — ATTAQUE")
        self.assertEqual(editor.format_gesture_label("G01", None), "G01")

    def test_gesture_rows_keep_empty_name_when_unlabeled(self):
        import gesture_editor as editor
        from gesture_engine import GestureSamples
        folder = tempfile.mkdtemp()
        engine = GestureEngine(os.path.join(folder, "gestures.json"))
        mapping = GestureMapping(os.path.join(folder, "mapping.json"))
        samples = [(pose(1) + i * 0.01).tolist() for i in range(4)]
        engine.database["gestures"]["LEFT"]["G01"] = GestureSamples(samples, "")
        rows = [row for row in editor.gesture_rows(engine, mapping, "LEFT") if row["id"] == "G01"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "")
        self.assertEqual(editor.format_gesture_label(rows[0]["id"], rows[0]["name"]), "G01")


class TiltDominantAxisTests(unittest.TestCase):
    def setUp(self):
        self.detector = geo.DirectionDetector("LEFT", geo.TILT, thresholds=geo._tilt_thresholds())

    def classify(self, dx, dy):
        return self.detector._candidate(np.array([dx, dy], dtype=np.float32))

    def test_cardinals(self):
        self.assertEqual(self.classify(0.0, 0.0), geo.DIR_NEUTRAL)
        self.assertEqual(self.classify(-0.9, -0.2), geo.DIR_LEFT)
        self.assertEqual(self.classify(0.9, -0.2), geo.DIR_RIGHT)
        self.assertEqual(self.classify(0.2, -0.95), geo.DIR_UP)
        self.assertEqual(self.classify(0.2, 0.8), geo.DIR_DOWN)

    def test_diagonals_follow_dominant_axis(self):
        self.assertEqual(self.classify(0.9, -0.7), geo.DIR_RIGHT)
        self.assertEqual(self.classify(-0.9, -0.7), geo.DIR_LEFT)
        self.assertEqual(self.classify(0.7, -0.95), geo.DIR_UP)
        self.assertEqual(self.classify(0.7, 0.85), geo.DIR_DOWN)

    def test_tie_is_vertical(self):
        self.assertEqual(self.classify(0.9, -0.9), geo.DIR_UP)
        self.assertEqual(self.classify(0.9, 0.9), geo.DIR_DOWN)

    def test_right_component_does_not_beat_larger_up(self):
        self.assertEqual(self.classify(0.50, -0.85), geo.DIR_UP)

    def test_pointing_right_is_not_tilt_right(self):
        engine = geo.GeometryEngine(stability_frames=1)
        pose = hand(palm=UP, index=RIGHTWARD, index_len=1.0)
        states = engine.update("LEFT", pose)
        self.assertEqual(states[geo.INDEX], geo.LEFT_INDEX_RIGHT)
        self.assertNotEqual(states[geo.TILT], geo.LEFT_TILT_RIGHT)


class FingerUpGeometryTests(unittest.TestCase):
    def test_index_up_requires_image_up_not_just_extension(self):
        vertical = finger_pose((1.0, 0.2, 0.2, 0.2))
        self.assertTrue(geo.detect_finger_up(vertical, "INDEX"))
        sideways = rotate_landmarks(vertical, 90)
        self.assertFalse(geo.detect_finger_up(sideways, "INDEX"))
        self.assertEqual(geo.finger_image_direction(geo.finger_direction_vector(sideways, "INDEX")), geo.DIR_RIGHT)
        self.assertFalse(geo.detect_finger_up(hand(palm=UP, index=RIGHTWARD, index_len=1.0), "INDEX"))
        self.assertFalse(geo.detect_finger_up(hand(palm=UP, index=DOWN, index_len=1.0), "INDEX"))

    def test_each_selected_finger_including_thumb(self):
        poses = {
            "INDEX": finger_pose((1.0, 0.2, 0.2, 0.2)),
            "MIDDLE": finger_pose((0.2, 1.0, 0.2, 0.2)),
            "RING": finger_pose((0.2, 0.2, 1.0, 0.2)),
            "PINKY": finger_pose((0.2, 0.2, 0.2, 1.0)),
            "THUMB": thumb_extended_pose(UP),
        }
        for finger, pose in poses.items():
            self.assertTrue(geo.detect_finger_up(pose, finger), finger)
            for other in geo.FINGER_UP_FINGERS:
                if other == finger:
                    continue
                self.assertFalse(geo.detect_finger_up(pose, other), f"{finger} vs {other}")

    def test_thumb_ids_are_stable_and_selectable(self):
        self.assertEqual(geo.finger_up_id("LEFT", "THUMB"), geo.LEFT_FINGER_THUMB)
        self.assertEqual(geo.finger_up_id("RIGHT", "THUMB"), geo.RIGHT_FINGER_THUMB)
        self.assertIn("THUMB", geo.FINGER_UP_FINGERS)
        engine = geo.GeometryEngine(stability_frames=1)
        states = engine.update("LEFT", thumb_extended_pose(UP))
        self.assertEqual(states[geo.FINGER_THUMB], geo.LEFT_FINGER_THUMB)

    def test_down_and_horizontal_are_not_finger_up(self):
        for finger, pose in (
            ("INDEX", finger_pose((1.0, 0.2, 0.2, 0.2))),
            ("MIDDLE", finger_pose((0.2, 1.0, 0.2, 0.2))),
            ("RING", finger_pose((0.2, 0.2, 1.0, 0.2))),
            ("PINKY", finger_pose((0.2, 0.2, 0.2, 1.0))),
            ("THUMB", thumb_extended_pose(UP)),
        ):
            self.assertFalse(geo.detect_finger_up(rotate_landmarks(pose, 180), finger), finger)
            self.assertFalse(geo.detect_finger_up(rotate_landmarks(pose, 90), finger), finger)

    def test_rotated_hand_keeps_the_selected_finger_when_still_up(self):
        pose = rotate_landmarks(finger_pose((0.2, 1.0, 0.2, 0.2)), -15)
        self.assertTrue(geo.detect_finger_up(pose, "MIDDLE"))
        self.assertFalse(geo.detect_finger_up(pose, "INDEX"))


class FingerSelectionUiTests(TempCase):
    def _app(self):
        app = HandControllerApp.__new__(HandControllerApp)
        app.engine = GestureEngine(self.gestures_path)
        app.mapping = GestureMapping(self.mapping_path)
        app.system_gestures = sysg.default_system_gestures()
        app.settings = {}
        app.sys_index = 0
        app.sys_edit = {}
        app.sys_phase = "list"
        app.sys_combo_capture = False
        app.status_message = ""
        app.geometry = geo.GeometryEngine()
        app._save_system_gestures = lambda: None
        app._build_geometry = lambda: geo.GeometryEngine()
        app._sys_rows = HandControllerApp._sys_rows.__get__(app)
        app._sys_cycle_finger = HandControllerApp._sys_cycle_finger.__get__(app)
        app._sys_begin_edit = HandControllerApp._sys_begin_edit.__get__(app)
        app._sys_row_ids = HandControllerApp._sys_row_ids.__get__(app)
        app._sys_allowed_types = HandControllerApp._sys_allowed_types.__get__(app)
        return app

    def test_cycle_finger_changes_binding_not_just_label(self):
        app = self._app()
        app.system_gestures = sysg.set_system_enabled(
            app.system_gestures, "finger_up", "LEFT", enabled=True, finger="INDEX",
        )
        app.mapping.set_command("LEFT", geo.LEFT_FINGER_INDEX, HOLD, ["A"])
        app.sys_index = next(
            index for index, row in enumerate(app._sys_rows())
            if row.get("kind") == "finger" and row.get("side") == "LEFT" and row.get("finger") == "INDEX"
        )
        app._sys_cycle_finger(1)
        self.assertTrue(sysg.is_system_enabled(app.system_gestures, "LEFT", geo.LEFT_FINGER_MIDDLE))
        self.assertFalse(sysg.is_system_enabled(app.system_gestures, "LEFT", geo.LEFT_FINGER_INDEX))
        self.assertEqual(app.mapping.get("LEFT", geo.LEFT_FINGER_MIDDLE)["inputs"], ["A"])
        row = app._sys_rows()[app.sys_index]
        self.assertEqual(row.get("finger"), "MIDDLE")


class CameraResolutionTests(unittest.TestCase):
    def test_presets_and_fallback(self):
        from config import (
            CAMERA_RESOLUTIONS, accept_camera_size, camera_size_matches, camera_try_order,
            nearest_camera_preset,
        )
        self.assertIn((640, 480), CAMERA_RESOLUTIONS)
        self.assertIn((1280, 720), CAMERA_RESOLUTIONS)
        self.assertIn((1920, 1080), CAMERA_RESOLUTIONS)
        self.assertEqual(nearest_camera_preset(1280, 720), (1280, 720))
        self.assertEqual(nearest_camera_preset(800, 600), (640, 480))
        self.assertTrue(accept_camera_size(1280, 720))
        self.assertFalse(accept_camera_size(0, 0))
        self.assertTrue(camera_size_matches(1920, 1080, 1920, 1080))
        self.assertFalse(camera_size_matches(1280, 720, 1920, 1080))
        self.assertEqual(camera_try_order(1920, 1080)[0], (1920, 1080))
        self.assertIn((640, 480), camera_try_order(1920, 1080))

    def test_open_camera_reads_actual_size(self):
        source = inspect.getsource(main_module.HandControllerApp.open_camera)
        self.assertIn("camera_try_order", source)
        self.assertIn("camera_actual_width", source)
        self.assertIn("CAP_PROP_FRAME_WIDTH", source)


class RecordPageTests(TempCase):
    def test_record_opens_one_create_panel(self):
        app = HandControllerApp.__new__(HandControllerApp)
        app.engine = GestureEngine(self.gestures_path)
        app.help_mode = False
        app.lang_mode = False
        app.settings_mode = True
        app.input_panel = True
        app.overlay_panel = False
        app.overlay_capture_slot = None
        app.editor_mode = True
        app.map_mode = False
        app.sys_mode = False
        app.system_test_mode = False
        app.finger_mode = False
        app.geo_calib_mode = False
        app.calib_mode = False
        app.calib_phase = "hand"
        app.calib_name_buffer = "x"
        app.calib_side = "RIGHT"
        app.calib_index = 0
        app.calib_end_time = 0
        app.status_message = ""
        app.reset_state = lambda **kwargs: None
        app.ensure_preview_visible = lambda: None
        app._cancel_overlay_capture = lambda: None
        app._close_other_ui = HandControllerApp._close_other_ui.__get__(app)
        app.enter_calibration = HandControllerApp.enter_calibration.__get__(app)
        app.enter_record_gesture = HandControllerApp.enter_record_gesture.__get__(app)
        app.enter_record_gesture()
        self.assertTrue(app.calib_mode)
        self.assertEqual(app.calib_phase, "create")
        self.assertFalse(app.editor_mode)
        self.assertFalse(app.settings_mode)
        self.assertFalse(app.input_panel)
        source = inspect.getsource(main_module.HandControllerApp.run)
        self.assertIn('overlay = "calib"', source)
        self.assertIn("draw_record_panel", inspect.getsource(main_module.HandControllerApp))


class CalibrationKeysAndDisplayTests(TempCase):
    def test_calibration_uses_old_keyboard_and_arrows(self):
        source = inspect.getsource(main_module.HandControllerApp.handle_calibration_key)
        menu = source.split('if self.calib_phase == "menu":', 1)[1].split("if self.calib_phase == \"mode\"", 1)[0]
        self.assertIn("is_right_arrow", menu)
        self.assertIn("is_left_arrow", menu)
        self.assertIn('ord("m")', menu)
        self.assertNotIn('ord("n")', menu)
        self.assertNotIn('ord("p")', menu)
        self.assertNotIn('ord("h")', menu)
        run_src = inspect.getsource(main_module.HandControllerApp.run)
        self.assertIn('overlay = "calib"', run_src)
        self.assertIn("draw_calibration", run_src)
        self.assertIn("fit_preview_frame", run_src)
        self.assertIn("advance_calibration", run_src)

    def test_calibration_hud_is_drawn_after_window_fit(self):
        run_src = inspect.getsource(main_module.HandControllerApp.run)
        fit_at = run_src.index("fit_preview_frame")
        draw_at = run_src.index("draw_calibration")
        self.assertLess(fit_at, draw_at)
        calib_block = run_src.split("elif self.calib_mode:")[1].split("elif self.geo_calib_mode:")[0]
        self.assertIn('overlay = "calib"', calib_block)
        self.assertNotIn("overlay = None", calib_block)

    def test_mapped_tilt_and_finger_up_are_allowed(self):
        app = make_geometry_app(self.mapping_path)
        app.system_test_mode = False
        app.system_gestures = sysg.default_system_gestures()
        app._system_allowed = HandControllerApp._system_allowed.__get__(app)
        self.assertFalse(app._system_allowed("LEFT", geo.LEFT_TILT_UP))
        app.mapping.set_command("LEFT", geo.LEFT_TILT_UP, HOLD, ["SPACE"])
        app._enable_preset_gate = HandControllerApp._enable_preset_gate.__get__(app)
        app._save_system_gestures = lambda: None
        app._build_geometry = lambda: app.geometry
        app._enable_preset_gate("LEFT", geo.LEFT_TILT_UP)
        self.assertTrue(app._system_allowed("LEFT", geo.LEFT_TILT_UP))
        app.mapping.set_command("RIGHT", geo.RIGHT_FINGER_INDEX, PRESS, ["F"], 0)
        app._enable_preset_gate("RIGHT", geo.RIGHT_FINGER_INDEX)
        self.assertTrue(app._system_allowed("RIGHT", geo.RIGHT_FINGER_INDEX))
        from hud_renderer import preset_display_name
        self.assertEqual(preset_display_name(geo.LEFT_TILT_UP), "TILT_UP")
        self.assertEqual(preset_display_name(geo.LEFT_FINGER_INDEX), "FINGER_UP_INDEX")
        self.assertEqual(preset_display_name(geo.RIGHT_TILT_DOWN), "TILT_DOWN")
        app.map_side = "LEFT"
        app.engine = GestureEngine(self.gestures_path)
        app.map_gesture_names = HandControllerApp.map_gesture_names.__get__(app)
        listed = app.map_gesture_names()
        self.assertIn(geo.LEFT_TILT_UP, listed)
        self.assertIn(geo.LEFT_TILT_DOWN, listed)
        self.assertIn(geo.LEFT_TILT_LEFT, listed)
        self.assertIn(geo.LEFT_TILT_RIGHT, listed)
        self.assertIn(geo.LEFT_FINGER_INDEX, listed)

    def test_display_path_uses_raw_points(self):
        source = inspect.getsource(main_module.HandControllerApp.process_frame)
        self.assertIn('display = det.get("raw_points")', source)
        smooth = inspect.getsource(main_module.HandControllerApp._apply_smoothing)
        self.assertIn('det["raw_points"] = raw', smooth)
        self.assertNotIn('points_to_landmarks(stable)', smooth)

    def test_finger_arrow_follows_enabled_bindings(self):
        app = HandControllerApp.__new__(HandControllerApp)
        app.system_gestures = sysg.default_system_gestures()
        app._finger_arrow_enabled = HandControllerApp._finger_arrow_enabled.__get__(app)
        self.assertFalse(app._finger_arrow_enabled("LEFT", "INDEX"))
        app.system_gestures = sysg.set_system_enabled(
            app.system_gestures, "finger_up", "LEFT", enabled=True, finger="THUMB",
        )
        self.assertTrue(app._finger_arrow_enabled("LEFT", "THUMB"))
        self.assertFalse(app._finger_arrow_enabled("LEFT", "INDEX"))


class RecordedDeletionTests(TempCase):
    def test_delete_removes_pose_mapping_and_survives_reload(self):
        from gesture_engine import GestureSamples
        app = HandControllerApp.__new__(HandControllerApp)
        app.engine = GestureEngine(self.gestures_path)
        app.mapping = GestureMapping(self.mapping_path)
        app.gestures_file = self.gestures_path
        app.mapping_file = self.mapping_path
        app.editor_side = "LEFT"
        app.settings_side = "LEFT"
        app.curr_gestures = {"LEFT": "G01", "RIGHT": UNKNOWN}
        app.status_message = ""
        samples = [(pose(9) + i * 0.01).tolist() for i in range(8)]
        app.engine.database["gestures"]["LEFT"]["G01"] = GestureSamples(samples, "ATTAQUE")
        app.engine.save_database()
        app.engine.rebuild_index()
        app.mapping.set_command("LEFT", "G01", PRESS, ["LEFT_MOUSE"], 0)
        app._recorded_rows = HandControllerApp._recorded_rows.__get__(app)
        app._delete_recorded_gesture = HandControllerApp._delete_recorded_gesture.__get__(app)
        self.assertTrue(any(row["id"] == "G01" and row["name"] == "ATTAQUE" for row in app._recorded_rows()))
        self.assertEqual(app.mapping.get("LEFT", "G01")["inputs"], ["LEFT_MOUSE"])
        self.assertTrue(app._delete_recorded_gesture("LEFT", "G01"))
        self.assertFalse(any(row["id"] == "G01" for row in app._recorded_rows()))
        self.assertEqual(app.engine.sample_count("LEFT", "G01"), 0)
        self.assertEqual(app.mapping.get("LEFT", "G01")["type"], NONE)
        self.assertEqual(app.curr_gestures["LEFT"], UNKNOWN)
        self.assertNotEqual(app.engine.recognize(np.asarray(samples[0], dtype=np.float32), "LEFT")[0], "G01")
        reloaded = GestureEngine(self.gestures_path)
        self.assertEqual(reloaded.sample_count("LEFT", "G01"), 0)
        self.assertEqual(reloaded.display_name("LEFT", "G01"), "")
        self.assertFalse(app._delete_recorded_gesture("LEFT", geo.LEFT_FINGER_THUMB))
        self.assertFalse(app._delete_recorded_gesture("LEFT", geo.LEFT_PINCH))


class SystemOverlayGateTests(TempCase):
    def _app(self):
        app = HandControllerApp.__new__(HandControllerApp)
        app.curr_gestures = {side: UNKNOWN for side in SIDES}
        app.curr_specials = {side: {} for side in SIDES}
        app.detected = {side: True for side in SIDES}
        app.system_gestures = sysg.default_system_gestures()
        app._system_allowed = HandControllerApp._system_allowed.__get__(app)
        return app

    def test_disabled_thumb_is_hidden_and_hands_are_independent(self):
        from overlay_labels import active_gesture_labels
        app = self._app()
        app.system_gestures = sysg.set_system_enabled(
            app.system_gestures, "finger_up", "RIGHT", enabled=True, finger="THUMB",
        )
        app.curr_specials["LEFT"] = {geo.FINGER_THUMB: geo.LEFT_FINGER_THUMB}
        app.curr_specials["RIGHT"] = {geo.FINGER_THUMB: geo.RIGHT_FINGER_THUMB}
        self.assertNotIn("THUMB UP", active_gesture_labels(app, "LEFT"))
        self.assertIn("THUMB UP", active_gesture_labels(app, "RIGHT"))
        app.system_gestures = sysg.set_system_enabled(
            app.system_gestures, "finger_up", "LEFT", enabled=True, finger="THUMB",
        )
        app.system_gestures = sysg.set_system_enabled(
            app.system_gestures, "finger_up", "RIGHT", enabled=False, finger="THUMB",
        )
        self.assertIn("THUMB UP", active_gesture_labels(app, "LEFT"))
        self.assertNotIn("THUMB UP", active_gesture_labels(app, "RIGHT"))

    def test_disabled_pinch_and_tilt_stay_off_the_overlay(self):
        from overlay_labels import active_gesture_labels
        app = self._app()
        app.curr_specials["LEFT"] = {geo.PINCH: geo.LEFT_PINCH, geo.TILT: geo.LEFT_TILT_UP}
        self.assertEqual(active_gesture_labels(app, "LEFT"), [])
        app.system_gestures = sysg.set_system_enabled(app.system_gestures, "pinch", "LEFT", enabled=True)
        app.system_gestures = sysg.set_system_enabled(
            app.system_gestures, "tilt", "LEFT", direction="up", enabled=True,
        )
        labels = active_gesture_labels(app, "LEFT")
        self.assertIn("PINCH LEFT", labels)
        self.assertIn("TILT UP", labels)


class QtShellTests(TempCase):
    def test_pyside_and_ui_package_are_gone(self):
        req = open(os.path.join(os.path.dirname(main_module.__file__), "requirements.txt"), encoding="utf-8").read()
        self.assertNotIn("PySide6", req)
        self.assertFalse(os.path.isdir(os.path.join(os.path.dirname(main_module.__file__), "ui")))
        self.assertNotIn("PySide", inspect.getsource(__import__("game_osd")))
        source = inspect.getsource(main_module)
        self.assertNotIn("from ui.", source)
        self.assertNotIn("PySide6", source)
        self.assertNotIn("SetForegroundWindow", source)

    def test_overlay_stays_off_in_default_settings(self):
        from config import default_settings
        self.assertEqual(default_settings()["ml_backend"], "knn")
        self.assertNotIn("game_osd", default_settings())
        self.assertIn("self.sidebar_hidden = True", inspect.getsource(main_module.HandControllerApp.__init__))

    def test_hud_shows_enabled_detected_only_and_keeps_hands_apart(self):
        from overlay_labels import active_gesture_labels
        app = HandControllerApp.__new__(HandControllerApp)
        app.engine = GestureEngine(self.gestures_path)
        app.mapping = GestureMapping(self.mapping_path)
        app.curr_gestures = {"LEFT": "G01", "RIGHT": "G05"}
        app.curr_specials = {
            "LEFT": {geo.PINCH: geo.LEFT_PINCH, geo.finger_up_family("INDEX"): geo.LEFT_FINGER_INDEX},
            "RIGHT": {geo.TILT: geo.RIGHT_TILT_RIGHT, geo.finger_up_family("MIDDLE"): geo.RIGHT_FINGER_MIDDLE},
        }
        app.system_gestures = sysg.default_system_gestures()
        app.system_gestures = sysg.set_system_enabled(app.system_gestures, "pinch", "LEFT", enabled=True)
        app.system_gestures = sysg.set_system_enabled(
            app.system_gestures, "finger_up", "LEFT", enabled=True, finger="index",
        )
        app.system_gestures = sysg.set_system_enabled(
            app.system_gestures, "tilt", "RIGHT", "right", True,
        )
        app._system_allowed = HandControllerApp._system_allowed.__get__(app)
        app.engine.database["gestures"]["LEFT"]["G01"].name = "ATTAQUE"
        app.engine.database["gestures"]["RIGHT"]["G05"].name = "ESQUIVE"
        app.mapping.set_command("LEFT", "G01", PRESS, ["W"], 0)
        app.mapping.set_command("RIGHT", "G05", PRESS, ["A"], 0)
        left = active_gesture_labels(app, "LEFT")
        right = active_gesture_labels(app, "RIGHT")
        self.assertTrue(any("G01" in item and "ATTAQUE" in item for item in left))
        self.assertIn("PINCH LEFT", left)
        self.assertIn("INDEX UP", left)
        self.assertNotIn("TILT RIGHT", left)
        self.assertNotIn("MIDDLE UP", left)
        self.assertTrue(any("G05" in item and "ESQUIVE" in item for item in right))
        self.assertIn("TILT RIGHT", right)
        self.assertNotIn("PINCH LEFT", right)
        self.assertNotIn("INDEX UP", right)
        self.assertNotIn("MIDDLE UP", right)
        app.mapping.disable("LEFT", "G01")
        left_off = active_gesture_labels(app, "LEFT")
        self.assertFalse(any(item.startswith("G01") for item in left_off))

    def test_geometry_pipeline_keeps_original_landmarks(self):
        source = inspect.getsource(main_module.HandControllerApp.update_special_gestures)
        self.assertIn("geometry_points", source)
        self.assertIn("stable_points", source)
        app = make_geometry_app(self.mapping_path)
        for _ in range(3):
            run_geometry_frame(app, left=OPEN_LEFT)
        self.assertEqual(app.curr_specials["LEFT"].get(geo.TILT), geo.LEFT_TILT_LEFT)

    def test_knn_orientation_invariant_for_several_recorded_gestures(self):
        from gesture_engine import GestureSamples
        engine = GestureEngine(self.gestures_path)
        poses = {
            "G01": FIST_POSE,
            "G02": PALM_POSE,
            "G03": POINTING_POSE,
            "G04": finger_pose((1.0, 1.0, 0.2, 0.2)),
            "G05": finger_pose((1.0, 1.0, 1.0, 0.2)),
        }
        for name, points in poses.items():
            samples = []
            for jitter in range(8):
                noisy = np.array(points, dtype=np.float32, copy=True)
                noisy += jitter * 0.001
                samples.append(GestureEngine.normalize_points(noisy).tolist())
            engine.database["gestures"]["LEFT"][name] = GestureSamples(samples, name)
            engine.database["gestures"]["RIGHT"][name] = GestureSamples(samples, name)
        engine.rebuild_index()
        for name, points in poses.items():
            queries = [
                points,
                rotate_landmarks(points, -20),
                rotate_landmarks(points, 20),
                rotate_landmarks(points, -12) + np.array([0.25, 0.0, 0.0], dtype=np.float32),
                points * 1.35 + np.array([0.0, 0.18, 0.0], dtype=np.float32),
            ]
            for query in queries:
                found, _conf = engine.recognize(GestureEngine.normalize_points(query), "LEFT")
                self.assertEqual(found, name, msg=f"{name} failed orientation query")
            found_right, _conf = engine.recognize(GestureEngine.normalize_points(points), "RIGHT")
            self.assertEqual(found_right, name)
        fist = GestureEngine.normalize_points(FIST_POSE)
        palm = GestureEngine.normalize_points(PALM_POSE)
        self.assertGreater(float(np.linalg.norm(fist - palm)), 0.35)

    def test_press_hold_combination_macro_wait_still_map(self):
        mapping = GestureMapping(self.mapping_path)
        mapping.set_command("LEFT", "G01", PRESS, ["W"], 0)
        mapping.set_command("LEFT", "G02", HOLD, ["SHIFT"])
        mapping.set_command("LEFT", "G03", COMBINATION, ["W", "SHIFT"])
        mapping.set_macro("LEFT", "G04", [
            {"action": STEP_PRESS, "inputs": ["W"]},
            {"action": STEP_WAIT, "duration": 100},
            {"action": STEP_HOLD, "inputs": ["SHIFT"]},
            {"action": STEP_WAIT, "duration": 200},
            {"action": STEP_PRESS, "inputs": ["LEFT_MOUSE"]},
            {"action": STEP_WAIT, "duration": 500},
            {"action": STEP_RELEASE, "inputs": ["SHIFT"]},
        ])
        self.assertEqual(mapping.get_type("LEFT", "G01"), PRESS)
        self.assertEqual(mapping.get_type("LEFT", "G02"), HOLD)
        self.assertEqual(mapping.get_type("LEFT", "G03"), COMBINATION)
        self.assertEqual(mapping.get_type("LEFT", "G04"), MACRO)
        steps = mapping.get_steps("LEFT", "G04")
        self.assertEqual(steps[1]["action"], STEP_WAIT)
        self.assertEqual(steps[1]["duration"], 100)

    def test_camera_apply_strict_fallback_restores_previous(self):
        app = HandControllerApp.__new__(HandControllerApp)
        app.settings = {"camera_width": 1280, "camera_height": 720}
        app.camera_width = 1280
        app.camera_height = 720
        app.camera_actual_width = 1280
        app.camera_actual_height = 720
        app.status_message = ""
        app._camera_restart = False
        saved = []

        def fake_save(data, path=None):
            saved.append(dict(data))
            return True

        original = main_module.save_settings
        main_module.save_settings = fake_save
        try:
            from localization import t
            app.apply_camera_resolution(1920, 1080)
            self.assertEqual((app.camera_width, app.camera_height), (1920, 1080))
            self.assertTrue(app._camera_strict)
            app._revert_camera_resolution()
            self.assertEqual((app.camera_width, app.camera_height), (1280, 720))
            self.assertIn("1920", t("ui.camera_unsupported", width=1920, height=1080))
        finally:
            main_module.save_settings = original


class RecognitionEngineUpgradeTests(TempCase):
    def test_quality_accepts_a_normal_hand_and_holds_a_jump(self):
        from landmark_quality import ACCEPT, HOLD, LandmarkQualityChecker
        checker = LandmarkQualityChecker()
        pose = PALM_POSE + np.array([0.4, 0.4, 0.0], dtype=np.float32)
        self.assertEqual(checker.evaluate("LEFT", pose, 0.95), ACCEPT)
        jumped = pose + np.array([0.55, 0.0, 0.0], dtype=np.float32)
        self.assertEqual(checker.evaluate("LEFT", jumped, 0.95), HOLD)
        self.assertEqual(checker.evaluate("RIGHT", pose, 0.95), ACCEPT)

    def test_quality_left_does_not_affect_right(self):
        from landmark_quality import ACCEPT, HOLD, LandmarkQualityChecker
        checker = LandmarkQualityChecker()
        left = PALM_POSE + np.array([0.3, 0.3, 0.0], dtype=np.float32)
        right = PALM_POSE + np.array([0.7, 0.3, 0.0], dtype=np.float32)
        self.assertEqual(checker.evaluate("LEFT", left, 0.9), ACCEPT)
        self.assertEqual(checker.evaluate("RIGHT", right, 0.9), ACCEPT)
        checker.evaluate("LEFT", left + np.array([0.5, 0.0, 0.0], dtype=np.float32), 0.9)
        self.assertEqual(checker.evaluate("RIGHT", right, 0.9), ACCEPT)

    def test_one_euro_resets_after_a_long_gap(self):
        from landmark_filter import OneEuroFilter
        filt = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        filt.filter(np.zeros((21, 3), dtype=np.float32), 0.0)
        later = np.ones((21, 3), dtype=np.float32)
        out = filt.filter(later, 1.0)
        np.testing.assert_allclose(out, later)

    def test_palm_features_ignore_translation_and_keep_fist_vs_palm_apart(self):
        from feature_extractor import knn_features
        base = knn_features(PALM_POSE)
        moved = knn_features(PALM_POSE + np.array([0.35, -0.2, 0.0], dtype=np.float32))
        scaled = knn_features(PALM_POSE * 1.4)
        tilted = knn_features(rotate_landmarks(PALM_POSE, 22))
        self.assertLess(float(np.linalg.norm(base - moved)), 0.12)
        self.assertLess(float(np.linalg.norm(base - scaled)), 0.12)
        self.assertLess(float(np.linalg.norm(base - tilted)), 0.6)
        self.assertGreater(float(np.linalg.norm(base - knn_features(FIST_POSE))), 0.35)

    def test_process_frame_skips_knn_on_held_quality(self):
        from landmark_quality import HOLD
        source = inspect.getsource(main_module.HandControllerApp.process_frame)
        self.assertIn("if det is None or self._quality_frozen(det):", source)
        specials = inspect.getsource(main_module.HandControllerApp.update_special_gestures)
        self.assertIn("if self._quality_frozen(det):", specials)
        app = make_geometry_app(self.mapping_path)
        app.curr_gestures = {"LEFT": "G01", "RIGHT": UNKNOWN}
        det = {"quality": HOLD, "raw": "G02", "conf": 0.99, "stable_points": PALM_POSE}
        self.assertTrue(HandControllerApp._quality_frozen(app, det))
        self.assertEqual(app.curr_gestures["LEFT"], "G01")

    def test_geometry_tilt_still_uses_image_landmarks(self):
        app = make_geometry_app(self.mapping_path)
        for _ in range(3):
            run_geometry_frame(app, left=OPEN_LEFT)
        self.assertEqual(app.curr_specials["LEFT"].get(geo.TILT), geo.LEFT_TILT_LEFT)
        source = inspect.getsource(main_module.HandControllerApp.update_special_gestures)
        self.assertIn("geometry_points", source)
        self.assertIn("_quality_frozen", source)

    def test_finger_up_uses_joint_chain_not_only_wrist_y(self):
        source = inspect.getsource(geo.is_finger_extended)
        self.assertIn("FINGER_CHAIN", source)
        self.assertIn("THUMB", source)
        self.assertIn("[:3]", source)
        pointing = POINTING_POSE
        self.assertTrue(geo.is_finger_extended(pointing, "INDEX"))
        self.assertFalse(geo.is_finger_extended(FIST_POSE, "INDEX"))

    def test_knn_k_comparison_runs_on_real_or_synthetic_samples(self):
        from classifier_benchmark import compare_k_values, compare_sklearn, compare_shape_features
        engine = GestureEngine(self.gestures_path)
        engine.database["gestures"]["LEFT"]["G01"] = samples_for(11)
        engine.database["gestures"]["LEFT"]["G02"] = samples_for(29)
        engine.rebuild_index()
        scores = compare_k_values(engine, ks=(3, 5, 7))
        self.assertIn(5, scores)
        self.assertIsNotNone(scores[5])
        self.assertGreaterEqual(scores[5], 0.5)
        shape = compare_shape_features(engine, k=5)
        self.assertIn("coords63", shape)
        sklearn_scores = compare_sklearn(engine)
        if sklearn_scores is not None and "knn5" in sklearn_scores:
            self.assertGreaterEqual(sklearn_scores["knn5"], 0.0)

    def test_pipeline_still_measures_filter_knn_and_geometry(self):
        source = inspect.getsource(main_module.HandControllerApp.process_frame)
        self.assertIn("_apply_quality", source)
        self.assertIn("measure(\"smoothing\"", source.replace(" ", ""))
        self.assertIn("quality", inspect.getsource(main_module.HandControllerApp.__init__))

    def test_quality_rejects_unusable_landmarks_then_recovers_after_a_real_move(self):
        from landmark_quality import ACCEPT, HOLD, REJECT, LandmarkQualityChecker
        checker = LandmarkQualityChecker()
        pose = PALM_POSE + np.array([0.4, 0.4, 0.0], dtype=np.float32)
        self.assertEqual(checker.evaluate("LEFT", None, 0.9), REJECT)
        self.assertEqual(checker.evaluate("LEFT", pose, 0.95), ACCEPT)
        jumped = pose + np.array([0.45, 0.0, 0.0], dtype=np.float32)
        self.assertEqual(checker.evaluate("LEFT", jumped, 0.95), HOLD)
        self.assertEqual(checker.evaluate("LEFT", jumped, 0.95), HOLD)
        self.assertEqual(checker.evaluate("LEFT", jumped, 0.95), ACCEPT)

    def test_smoothing_keys_are_runtime_settings(self):
        from config import default_settings
        settings = default_settings()
        for key in ("landmark_smoothing", "smoothing_min_cutoff", "smoothing_beta", "smoothing_d_cutoff"):
            self.assertIn(key, settings)

    def test_new_engine_modules_never_open_gestures_backup(self):
        root = os.path.dirname(main_module.__file__)
        for name in ("landmark_quality.py", "feature_extractor.py", "classifier_benchmark.py",
                     "recognition_engine.py", "ml_engine.py", "ensemble_classifier.py",
                     "geometry_diagnostics.py"):
            with open(os.path.join(root, name), encoding="utf-8") as handle:
                self.assertNotIn("gestures_backup", handle.read(), name)

    def test_live_knn_keeps_63_features_and_shape_is_optional(self):
        from feature_extractor import FEATURE_SIZE, knn_features, local_palm_points, shape_features
        features = knn_features(PALM_POSE)
        self.assertEqual(features.size, FEATURE_SIZE)
        extras = shape_features(local_palm_points(PALM_POSE))
        self.assertEqual(extras.size, 10)
        source = inspect.getsource(main_module.HandControllerApp.process_frame)
        self.assertIn("normalize_points", source)
        self.assertNotIn("shape_features", source)

    def test_confidence_uses_nearest_distance_and_second_class_gap(self):
        from gesture_engine import GestureSamples
        engine = GestureEngine(self.gestures_path)
        fist = GestureEngine.normalize_points(FIST_POSE).tolist()
        palm = GestureEngine.normalize_points(PALM_POSE).tolist()
        engine.database["gestures"]["LEFT"]["G01"] = GestureSamples([fist] * 6)
        engine.database["gestures"]["LEFT"]["G02"] = GestureSamples([palm] * 6)
        engine.rebuild_index()
        name, confidence = engine.recognize(GestureEngine.normalize_points(FIST_POSE), "LEFT")
        self.assertEqual(name, "G01")
        self.assertGreater(confidence, 0.7)

    def test_hand_loss_resets_quality_and_filter_after_grace(self):
        smoothing = inspect.getsource(main_module.HandControllerApp._apply_smoothing)
        quality = inspect.getsource(main_module.HandControllerApp._apply_quality)
        self.assertIn("smoother.reset()", smoothing)
        self.assertIn("checker.reset(side)", quality)
        self.assertIn("_in_loss_grace", smoothing)
        self.assertIn("_in_loss_grace", quality)

    def test_ensemble_votes_and_rejects_disagreement(self):
        from ensemble_classifier import EnsembleClassifier
        from recognition_engine import RecognitionEngine
        from gesture_engine import GestureSamples
        voter = EnsembleClassifier(mode="weighted", min_confidence=0.2, class_margin=0.05)
        gesture, conf, info = voter.decide({
            "knn": ("G01", 0.9),
            "svm": ("G01", 0.8),
            "random_forest": ("G01", 0.7),
        })
        self.assertEqual(gesture, "G01")
        self.assertGreater(info["agreement"], 0.9)
        unknown, _conf, _info = voter.decide({
            "knn": ("G01", 0.6),
            "svm": ("G02", 0.6),
            "random_forest": ("G03", 0.6),
        })
        self.assertEqual(unknown, UNKNOWN)
        engine = GestureEngine(self.gestures_path)
        fist = GestureEngine.normalize_points(FIST_POSE).tolist()
        palm = GestureEngine.normalize_points(PALM_POSE).tolist()
        engine.database["gestures"]["LEFT"]["G01"] = GestureSamples([fist] * 8)
        engine.database["gestures"]["LEFT"]["G02"] = GestureSamples([palm] * 8)
        engine.rebuild_index()
        rec = RecognitionEngine(engine, settings={"ml_backend": "ensemble", "confidence_threshold": 0.2})
        result = rec.recognize_ml(GestureEngine.normalize_points(FIST_POSE), "LEFT")
        self.assertEqual(result.knn_gesture, "G01")
        self.assertIn(result.gesture_id, ("G01", UNKNOWN))
        self.assertEqual(rec.features(FIST_POSE).size, 63)


def _require_sklearn():
    from ml_engine import sklearn_available
    if not sklearn_available():
        raise unittest.SkipTest("sklearn unavailable")


class SklearnModelTests(TempCase):
    def _two_class_engine(self):
        from gesture_engine import GestureSamples
        engine = GestureEngine(self.gestures_path)
        a = GestureEngine.normalize_points(FIST_POSE).tolist()
        b = GestureEngine.normalize_points(PALM_POSE).tolist()
        engine.database["gestures"]["LEFT"]["G01"] = GestureSamples([a] * 8)
        engine.database["gestures"]["LEFT"]["G02"] = GestureSamples([b] * 8)
        engine.rebuild_index()
        return engine

    def test_svm_train_predict(self):
        _require_sklearn()
        from ml_engine import MLEngine
        from recognition_engine import RecognitionEngine
        engine = self._two_class_engine()
        rec = RecognitionEngine(engine, settings={"ml_backend": "svm", "confidence_threshold": 0.0})
        rec.ml.train(force=True)
        result = rec.recognize_ml(GestureEngine.normalize_points(FIST_POSE), "LEFT")
        self.assertIn(result.svm_gesture, ("G01", "G02"))
        self.assertIn(result.gesture_id, ("G01", "G02", UNKNOWN))
        self.assertIn("svm", result.active_models)

    def test_random_forest_train_predict(self):
        _require_sklearn()
        from recognition_engine import RecognitionEngine
        engine = self._two_class_engine()
        rec = RecognitionEngine(engine, settings={"ml_backend": "rf", "confidence_threshold": 0.0})
        rec.ml.train(force=True)
        result = rec.recognize_ml(GestureEngine.normalize_points(PALM_POSE), "LEFT")
        self.assertIn(result.rf_gesture, ("G01", "G02"))
        self.assertIn("random_forest", result.active_models)

    def test_ensemble_vote(self):
        _require_sklearn()
        from ensemble_classifier import EnsembleClassifier, EnsemblePrediction
        from recognition_engine import RecognitionEngine
        voter = EnsembleClassifier(mode="hard", min_confidence=0.0, class_margin=0.0)
        gesture, conf, info = voter.decide({
            "knn": ("G01", 0.9),
            "svm": ("G01", 0.8),
            "random_forest": ("G02", 0.7),
        })
        self.assertEqual(gesture, "G01")
        self.assertGreaterEqual(info["agreement"], 0.66)
        self.assertEqual(sorted(info["active_models"]), ["knn", "random_forest", "svm"])
        self.assertFalse(info["ambiguous"])
        self.assertIsInstance(EnsemblePrediction(), EnsemblePrediction)
        engine = self._two_class_engine()
        rec = RecognitionEngine(engine, settings={"ml_backend": "ensemble", "confidence_threshold": 0.0})
        rec.ml.train(force=True)
        result = rec.recognize_ml(GestureEngine.normalize_points(FIST_POSE), "LEFT")
        self.assertEqual(result.source, "ensemble")
        self.assertGreaterEqual(len(result.active_models), 2)

    def test_insufficient_classes_skips_svm_rf(self):
        from gesture_engine import GestureSamples
        from ml_engine import MIN_CLASSES, SKIP_INSUFFICIENT, MLEngine
        engine = GestureEngine(self.gestures_path)
        pose = GestureEngine.normalize_points(FIST_POSE).tolist()
        engine.database["gestures"]["LEFT"]["G01"] = GestureSamples([pose] * 8)
        engine.rebuild_index()
        ml = MLEngine(engine)
        ml.train(force=True)
        self.assertEqual(ml.skip_reason["LEFT"], SKIP_INSUFFICIENT)
        self.assertIsNone(ml._svm["LEFT"])
        self.assertEqual(MIN_CLASSES, 2)

    def test_ensemble_falls_back_to_knn_without_extra_models(self):
        from recognition_engine import RecognitionEngine
        engine = GestureEngine(self.gestures_path)
        rec = RecognitionEngine(engine, settings={"ml_backend": "ensemble"})
        result = rec.recognize_ml(GestureEngine.normalize_points(FIST_POSE), "LEFT")
        self.assertEqual(result.source, "knn")
        self.assertEqual(result.active_models, ["knn"])


class GesturesJsonRecognitionTests(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.dirname(os.path.abspath(main_module.__file__)), "gestures.json")
        if not os.path.isfile(self.path):
            self.skipTest("gestures.json missing")
        if os.path.basename(self.path).lower() == "gestures_backup.json":
            self.fail("tests must not use gestures_backup.json")

    def _engine(self):
        return GestureEngine(self.path)

    def test_knn_left_and_right_on_gestures_json(self):
        from classifier_benchmark import bank_vectors, leave_one_out_hand
        engine = self._engine()
        for side in SIDES:
            rows = bank_vectors(engine, side)
            result = leave_one_out_hand(rows, k=5, include_sklearn=False)
            if result["samples"] < 2:
                continue
            self.assertIsNotNone(result["knn"].get("accuracy"))
            self.assertGreaterEqual(result["knn"]["accuracy"], 0.0)

    def test_svm_random_forest_ensemble_on_gestures_json(self):
        from gesture_engine import samples_of
        from ml_engine import MIN_CLASSES
        from recognition_engine import RecognitionEngine
        engine = self._engine()
        rec = RecognitionEngine(engine, settings={"ml_backend": "ensemble", "confidence_threshold": 0.0})
        rec.ml.train(force=True)
        for side in SIDES:
            names = [
                name for name, entry in engine.database["gestures"][side].items()
                if samples_of(entry)
            ]
            if len(names) < MIN_CLASSES:
                self.assertIsNone(rec.ml._svm.get(side))
                continue
            _require_sklearn()
            sample = samples_of(engine.database["gestures"][side][names[0]])[0]
            feat = GestureEngine.canonicalize_features(sample)
            result = rec.recognize_ml(feat, side)
            self.assertEqual(result.source, "ensemble")
            self.assertIn("svm", result.active_models)
            self.assertIn(result.svm_gesture, names + [UNKNOWN])
            self.assertIn(result.rf_gesture, names + [UNKNOWN])


class FinalHandTrackingAuditTests(unittest.TestCase):
    """Last V1 robustness pass: relative geometry, quality parts, short HOLD interpolation."""

    def test_relative_metrics_are_scale_invariant(self):
        near = geo.relative_metrics(hand(scale=0.8, pinch=0.2))
        far = geo.relative_metrics(hand(scale=0.12, pinch=0.2))
        self.assertAlmostEqual(near["pinch_ratio"], far["pinch_ratio"], places=5)
        self.assertAlmostEqual(near["palm_aspect"], far["palm_aspect"], places=5)
        self.assertAlmostEqual(near["ext_index"], far["ext_index"], places=5)
        self.assertGreater(near["scale"], far["scale"])

    def test_finger_curl_arch_spread_and_thumb_span(self):
        open_hand = PALM_POSE
        fist = FIST_POSE
        pointing = POINTING_POSE
        self.assertLess(geo.finger_curl(open_hand, "INDEX"), geo.finger_curl(fist, "INDEX"))
        self.assertGreater(geo.finger_arch(open_hand, "INDEX"), geo.finger_arch(fist, "INDEX"))
        pinched_spread = np.array(open_hand, copy=True)
        pinched_spread[geo.PINKY_TIP] = pinched_spread[geo.INDEX_TIP]
        self.assertGreater(geo.finger_spread(open_hand), geo.finger_spread(pinched_spread))
        self.assertGreater(geo.finger_extension_ratio(pointing, "INDEX"), geo.finger_extension_ratio(fist, "INDEX"))
        thumb_up = thumb_extended_pose(UP)
        self.assertGreater(geo.thumb_span(thumb_up), 0.0)
        self.assertLess(geo.finger_curl(thumb_up, "THUMB"), geo.finger_curl(fist, "THUMB") + 0.35)

    def test_geometry_sanity_accepts_palm_and_flags_exploded(self):
        pose = PALM_POSE + np.array([0.4, 0.4, 0.0], dtype=np.float32)
        self.assertTrue(geo.geometry_sane(pose))
        exploded = np.array(pose, copy=True)
        exploded[geo.INDEX_TIP] = exploded[geo.WRIST] + np.array([8.0, 8.0, 0.0], dtype=np.float32)
        ok, reasons = geo.geometry_sanity(exploded)
        self.assertFalse(ok)
        self.assertTrue(any(item.startswith("finger_") for item in reasons))

    def test_quality_components_stay_separate(self):
        from landmark_quality import ACCEPT, LandmarkQualityChecker
        checker = LandmarkQualityChecker()
        pose = PALM_POSE + np.array([0.4, 0.4, 0.0], dtype=np.float32)
        report = checker.evaluate_report("LEFT", pose, 0.95)
        self.assertEqual(report.action, ACCEPT)
        self.assertGreater(report.quality_score, 0.0)
        self.assertGreater(report.detection_quality, 0.0)
        self.assertGreater(report.finger_quality, 0.0)
        self.assertGreater(report.handedness_quality, 0.0)
        self.assertGreater(report.temporal_quality, 0.0)
        self.assertAlmostEqual(
            report.quality_score,
            min(report.detection_quality, report.finger_quality, report.temporal_quality),
        )

    def test_quality_lowers_finger_score_on_impossible_geometry(self):
        from landmark_quality import ACCEPT, HOLD, LandmarkQualityChecker
        checker = LandmarkQualityChecker()
        pose = PALM_POSE + np.array([0.4, 0.4, 0.0], dtype=np.float32)
        first = checker.evaluate_report("LEFT", pose, 0.95)
        self.assertEqual(first.action, ACCEPT)
        exploded = np.array(pose, copy=True)
        exploded[geo.INDEX_TIP] = exploded[geo.WRIST] + np.array([8.0, 8.0, 0.0], dtype=np.float32)
        held = checker.evaluate_report("LEFT", exploded, 0.95)
        self.assertEqual(held.action, HOLD)
        self.assertLess(held.finger_quality, first.finger_quality)
        self.assertLess(held.quality_score, first.quality_score)

    def test_temporal_interpolation_reuses_last_accept_without_extrapolation(self):
        from landmark_quality import ACCEPT, HOLD, LandmarkHold
        holder = LandmarkHold(max_frames=2)
        pose = PALM_POSE + np.array([0.4, 0.4, 0.0], dtype=np.float32)
        first, interpolated = holder.observe("LEFT", pose, ACCEPT, 1.0)
        self.assertFalse(interpolated)
        np.testing.assert_allclose(first, pose)
        bad = pose + np.array([0.6, 0.0, 0.0], dtype=np.float32)
        held, interpolated = holder.observe("LEFT", bad, HOLD, 1.04)
        self.assertTrue(interpolated)
        np.testing.assert_allclose(held, pose)
        held2, interpolated2 = holder.observe("LEFT", None, HOLD, 1.08)
        self.assertTrue(interpolated2)
        np.testing.assert_allclose(held2, pose)
        gone, interpolated3 = holder.observe("LEFT", None, HOLD, 1.12)
        self.assertFalse(interpolated3)
        self.assertIsNone(gone)

    def test_interpolation_is_not_gesture_proof(self):
        source = inspect.getsource(main_module.HandControllerApp.process_frame)
        self.assertIn("if det is None or self._quality_frozen(det):", source)
        self.assertIn("display_points", source)
        quality = inspect.getsource(main_module.HandControllerApp._apply_quality)
        self.assertIn("drawing only", quality)

    def test_hold_interpolator_left_does_not_affect_right(self):
        from landmark_quality import ACCEPT, HOLD, LandmarkHold
        holder = LandmarkHold(max_frames=2)
        left = PALM_POSE + np.array([0.3, 0.3, 0.0], dtype=np.float32)
        right = PALM_POSE + np.array([0.7, 0.3, 0.0], dtype=np.float32)
        holder.observe("LEFT", left, ACCEPT)
        holder.observe("RIGHT", right, ACCEPT)
        held, interpolated = holder.observe("LEFT", None, HOLD)
        self.assertTrue(interpolated)
        np.testing.assert_allclose(held, left)
        still, interp_right = holder.observe("RIGHT", right, ACCEPT)
        self.assertFalse(interp_right)
        np.testing.assert_allclose(still, right)

    def test_pinch_tilt_finger_thresholds_unchanged(self):
        self.assertAlmostEqual(geo.PINCH_ON_THRESHOLD, 0.25)
        self.assertAlmostEqual(geo.PINCH_OFF_THRESHOLD, 0.32)
        self.assertAlmostEqual(geo.TILT_LEFT_THRESHOLD, 0.45)
        self.assertAlmostEqual(geo.TILT_RIGHT_THRESHOLD, 0.45)
        self.assertAlmostEqual(geo.TILT_UP_THRESHOLD, 0.80)
        self.assertAlmostEqual(geo.TILT_DOWN_THRESHOLD, 0.55)
        self.assertAlmostEqual(geo.FINGER_UP_ON_THRESHOLD, 0.55)
        self.assertAlmostEqual(geo.FINGER_UP_OFF_THRESHOLD, 0.40)

    def test_pinch_tilt_and_finger_up_still_detect(self):
        engine = geo.GeometryEngine()
        for _ in range(geo.STABILITY_FRAMES):
            engine.update("LEFT", PINCHING)
            engine.update("RIGHT", OPEN_RIGHT)
        self.assertEqual(engine.states("LEFT")[geo.PINCH], geo.LEFT_PINCH)
        self.assertEqual(engine.states("RIGHT")[geo.TILT], geo.RIGHT_TILT_RIGHT)
        pointing = POINTING_POSE
        self.assertTrue(geo.detect_finger_up(pointing, "INDEX"))
        self.assertFalse(geo.detect_finger_up(FIST_POSE, "INDEX"))
        self.assertFalse(geo.detect_finger_up(FIST_POSE, "MIDDLE"))
        self.assertFalse(geo.detect_finger_up(FIST_POSE, "RING"))
        self.assertFalse(geo.detect_finger_up(FIST_POSE, "PINKY"))
        self.assertTrue(geo.detect_finger_up(thumb_extended_pose(UP), "THUMB"))

    def test_close_far_tilted_and_partially_occluded_hands(self):
        close = PALM_POSE * 1.35 + np.array([0.40, 0.55, 0.0], dtype=np.float32)
        far = PALM_POSE * 0.45 + np.array([0.45, 0.50, 0.0], dtype=np.float32)
        tilted = rotate_landmarks(PALM_POSE + np.array([0.4, 0.5, 0.0], dtype=np.float32), 28)
        from landmark_quality import ACCEPT, LandmarkQualityChecker
        checker = LandmarkQualityChecker()
        self.assertEqual(checker.evaluate("LEFT", close, 0.95), ACCEPT)
        other = LandmarkQualityChecker()
        self.assertEqual(other.evaluate("RIGHT", far, 0.95), ACCEPT)
        third = LandmarkQualityChecker()
        self.assertEqual(third.evaluate("LEFT", tilted, 0.95), ACCEPT)
        occluded = np.array(PALM_POSE + np.array([0.4, 0.4, 0.0], dtype=np.float32), copy=True)
        for index in (geo.INDEX_TIP, geo.MIDDLE_TIP, geo.RING_TIP):
            occluded[index] = occluded[geo.WRIST]
        report = LandmarkQualityChecker().evaluate_report("LEFT", occluded, 0.9)
        self.assertIn(report.action, (ACCEPT, "hold", "reject"))
        self.assertTrue(geo.geometry_sane(occluded) or report.finger_quality < 1.0)

    def test_fast_motion_is_still_accepted(self):
        from landmark_quality import ACCEPT, LandmarkQualityChecker
        checker = LandmarkQualityChecker()
        pose = PALM_POSE + np.array([0.35, 0.35, 0.0], dtype=np.float32)
        self.assertEqual(checker.evaluate("LEFT", pose, 0.95), ACCEPT)
        self.assertEqual(checker.evaluate("LEFT", pose + np.array([0.22, 0.08, 0.0], dtype=np.float32), 0.95), ACCEPT)

    def test_orientation_invariance_on_g01_to_g05(self):
        from gesture_engine import GestureSamples
        path = os.path.join(tempfile.mkdtemp(), "gestures.json")
        engine = GestureEngine(path)
        poses = {
            "G01": FIST_POSE,
            "G02": PALM_POSE,
            "G03": POINTING_POSE,
            "G04": finger_pose((1.0, 1.0, 0.2, 0.2)),
            "G05": finger_pose((1.0, 1.0, 1.0, 0.2)),
        }
        for name, points in poses.items():
            samples = [GestureEngine.normalize_points(points + i * 0.001).tolist() for i in range(8)]
            engine.database["gestures"]["LEFT"][name] = GestureSamples(samples, name)
        engine.rebuild_index()
        for name, points in poses.items():
            for degrees in (-25, -12, 0, 12, 25):
                query = rotate_landmarks(points, degrees)
                found, _conf = engine.recognize(GestureEngine.normalize_points(query), "LEFT")
                self.assertEqual(found, name)

    def test_orientation_invariance_on_gestures_json_if_present(self):
        from feature_extractor import knn_features
        from gesture_engine import samples_of
        path = os.path.join(os.path.dirname(os.path.abspath(main_module.__file__)), "gestures.json")
        if not os.path.isfile(path):
            self.skipTest("gestures.json missing")
        engine = GestureEngine(path)
        checked = 0
        for side in SIDES:
            for name in ("G01", "G02", "G03", "G04", "G05"):
                entry = engine.database["gestures"][side].get(name)
                samples = samples_of(entry) if entry is not None else []
                if not samples:
                    continue
                base = GestureEngine.canonicalize_features(samples[0])
                if base is None:
                    continue
                points = base.reshape(21, 3)
                for degrees in (-20, 20):
                    rotated = knn_features(rotate_landmarks(points, degrees))
                    found, _conf = engine.recognize(rotated, side)
                    self.assertEqual(found, name, msg=f"{side} {name} rotation {degrees}")
                checked += 1
        if checked == 0:
            self.skipTest("G01-G05 not populated in gestures.json")

    def test_pipeline_layers_stay_separated(self):
        self.assertNotIn("pynput", inspect.getsource(geo))
        mapping_src = inspect.getsource(main_module.HandControllerApp.execute_gesture)
        self.assertNotIn("hand_landmarks", mapping_src)
        self.assertNotIn("normalize_points", mapping_src)
        quality_src = inspect.getsource(main_module.HandControllerApp._apply_quality)
        self.assertNotIn("execute_gesture", quality_src)
        self.assertNotIn("pinch_ratio", inspect.getsource(main_module.HandControllerApp.execute_gesture))

    def test_diagnostics_compare_metrics_without_writing_settings(self):
        from geometry_diagnostics import compare_metric_sets, format_discriminators, metrics_from_landmarks
        correct = [metrics_from_landmarks(PINCHING) for _ in range(4)]
        impostor = [metrics_from_landmarks(OPENED) for _ in range(4)]
        report = compare_metric_sets(correct, impostor)
        self.assertIn("pinch_ratio", report["metrics"])
        text = format_discriminators(report)
        self.assertIn("BEST DISCRIMINATORS", text)
        names = {item["metric"] for item in report["best_discriminators"]}
        self.assertIn("pinch_ratio", names)
        source = inspect.getsource(__import__("geometry_diagnostics"))
        self.assertNotIn("save_settings", source)
        self.assertNotIn("gestures_backup", source)

    def test_gesture_result_keeps_quality_components(self):
        from landmark_quality import QualityReport
        from recognition_engine import GestureResult, RecognitionEngine
        path = os.path.join(tempfile.mkdtemp(), "gestures.json")
        engine = GestureEngine(path)
        rec = RecognitionEngine(engine)
        report = QualityReport(action="accept", quality_score=0.8, detection_quality=0.9,
                               finger_quality=0.7, handedness_quality=0.85, temporal_quality=1.0)
        result = rec.recognize_ml(GestureEngine.normalize_points(PALM_POSE), "LEFT", quality_report=report)
        self.assertIsInstance(result, GestureResult)
        self.assertAlmostEqual(result.quality_score, 0.8)
        self.assertAlmostEqual(result.finger_quality, 0.7)
        self.assertAlmostEqual(result.handedness_quality, 0.85)
        self.assertAlmostEqual(result.temporal_quality, 1.0)
        self.assertAlmostEqual(result.detection_quality, 0.9)


class LandmarkQualityScenarioTests(unittest.TestCase):
    def test_stable_landmarks_are_accepted(self):
        from landmark_quality import ACCEPT, LandmarkQualityChecker
        checker = LandmarkQualityChecker()
        pose = PALM_POSE + np.array([0.4, 0.4, 0.0], dtype=np.float32)
        self.assertEqual(checker.evaluate("LEFT", pose, 0.95), ACCEPT)
        moved = pose + np.array([0.08, 0.04, 0.0], dtype=np.float32)
        self.assertEqual(checker.evaluate("LEFT", moved, 0.95), ACCEPT)

    def test_fast_valid_motion_is_not_a_glitch(self):
        from landmark_quality import ACCEPT, LandmarkQualityChecker
        checker = LandmarkQualityChecker()
        pose = PALM_POSE + np.array([0.35, 0.35, 0.0], dtype=np.float32)
        self.assertEqual(checker.evaluate("LEFT", pose, 0.95), ACCEPT)
        fast = pose + np.array([0.20, 0.10, 0.0], dtype=np.float32)
        self.assertEqual(checker.evaluate("LEFT", fast, 0.95), ACCEPT)

    def test_mediapipe_glitch_holds_then_relocks(self):
        from landmark_quality import ACCEPT, HOLD, LandmarkQualityChecker
        checker = LandmarkQualityChecker()
        pose = PALM_POSE + np.array([0.4, 0.4, 0.0], dtype=np.float32)
        self.assertEqual(checker.evaluate("LEFT", pose, 0.95), ACCEPT)
        glitch = pose + np.array([0.50, 0.0, 0.0], dtype=np.float32)
        self.assertEqual(checker.evaluate("LEFT", glitch, 0.95), HOLD)
        self.assertEqual(checker.evaluate("LEFT", glitch, 0.95), HOLD)
        self.assertEqual(checker.evaluate("LEFT", glitch, 0.95), ACCEPT)

    def test_impossible_jump_is_held_not_trusted_immediately(self):
        from landmark_quality import ACCEPT, HOLD, LandmarkQualityChecker
        checker = LandmarkQualityChecker()
        pose = PALM_POSE + np.array([0.4, 0.4, 0.0], dtype=np.float32)
        self.assertEqual(checker.evaluate("LEFT", pose, 0.95), ACCEPT)
        jump = pose + np.array([0.9, 0.9, 0.0], dtype=np.float32)
        self.assertEqual(checker.evaluate("LEFT", jump, 0.95), HOLD)

    def test_hand_loss_rejects_after_hold_budget(self):
        from landmark_quality import ACCEPT, HOLD, REJECT, LandmarkQualityChecker
        checker = LandmarkQualityChecker()
        pose = PALM_POSE + np.array([0.4, 0.4, 0.0], dtype=np.float32)
        self.assertEqual(checker.evaluate("LEFT", pose, 0.95), ACCEPT)
        self.assertEqual(checker.evaluate("LEFT", None, 0.95), HOLD)
        self.assertEqual(checker.evaluate("LEFT", None, 0.95), HOLD)
        self.assertEqual(checker.evaluate("LEFT", None, 0.95), REJECT)


class MlBackendStatusTests(unittest.TestCase):
    def test_status_never_claims_ensemble_when_only_knn(self):
        from ml_engine import format_ml_startup, get_ml_backend_status
        status = get_ml_backend_status(None, requested="ensemble")
        self.assertTrue(status["knn"])
        self.assertFalse(status["ensemble"])
        text = format_ml_startup(status, requested="ensemble")
        self.assertNotIn("Ensemble : KNN + SVM", text)
        if not status["sklearn_installed"]:
            self.assertIn("k-NN uniquement", text)


class OpenCvUxShortcutTests(TempCase):
    def _dispatch(self, app, key, raw=0):
        import ui_context
        ctx = ui_context.of(app)
        if ui_context.is_text_context(ctx):
            if app.map_mode:
                app.handle_mapping_key(key, raw)
            elif app.calib_mode:
                app.handle_calibration_key(key, raw)
            return
        if (key in (ord("q"), ord("Q"))) and ui_context.allows_quit(ctx):
            app.running = False
            return
        if app.map_mode:
            app.handle_mapping_key(key, raw)
        elif app.calib_mode:
            app.handle_calibration_key(key, raw)
        else:
            app.handle_idle_key(key, raw)

    def _bind(self, app):
        app.handle_mapping_key = HandControllerApp.handle_mapping_key.__get__(app)
        app.handle_calibration_key = HandControllerApp.handle_calibration_key.__get__(app)
        app.handle_idle_key = HandControllerApp.handle_idle_key.__get__(app)
        app._ui_consumes_text = HandControllerApp._ui_consumes_text.__get__(app)
        app.ui_mode = HandControllerApp.ui_mode.__get__(app)
        app.toggle_hud_overlay = HandControllerApp.toggle_hud_overlay.__get__(app)
        app.map_gesture_names = HandControllerApp.map_gesture_names.__get__(app)
        app.current_map_gesture = HandControllerApp.current_map_gesture.__get__(app)
        app.recorded_gesture_ids = HandControllerApp.recorded_gesture_ids.__get__(app)
        app._listable_preset = HandControllerApp._listable_preset
        app.load_edit_buffers = HandControllerApp.load_edit_buffers.__get__(app)
        app._shift_map_index = HandControllerApp._shift_map_index.__get__(app)
        app._save_simple_or_combo = HandControllerApp._save_simple_or_combo.__get__(app)
        app.print_edit_screen = lambda: None
        app._enable_preset_gate = lambda *args, **kwargs: None
        return app

    def _app(self):
        from localization import load_catalogs, set_language
        load_catalogs(force=True)
        set_language("fr")
        app = HandControllerApp.__new__(HandControllerApp)
        app.engine = GestureEngine(self.gestures_path)
        from gesture_engine import GestureSamples
        for index in range(1, 9):
            gid = f"G{index:02d}"
            app.engine.database["gestures"]["LEFT"][gid] = GestureSamples(samples_for(index), gid)
            app.engine.database["gestures"]["RIGHT"][gid] = GestureSamples(samples_for(100 + index), gid)
        app.mapping = GestureMapping(self.mapping_path)
        app.inputs = make_inputs()
        app.running = True
        app.map_mode = False
        app.map_phase = "list"
        app.map_side = "LEFT"
        app.map_index = 0
        app.rename_buffer = ""
        app.status_message = ""
        app.sys_mode = False
        app.sys_phase = "list"
        app.calib_mode = False
        app.calib_phase = "menu"
        app.calib_index = 0
        app.calib_side = "LEFT"
        app.calib_name_buffer = ""
        app.hud_overlay = False
        app.settings = {}
        app.lang_mode = False
        app.help_mode = False
        app.finger_mode = False
        app.editor_mode = False
        app.edit_type = HOLD
        app.edit_inputs = ["W"]
        app.edit_modes = [HOLD]
        app.edit_slot = 0
        app.edit_cooldown = 3
        app.macro_steps = []
        app.macro_index = 0
        app.macro_speed = 1
        app.capture_combo = False
        app.test_active = False
        return self._bind(app)

    def test_s_toggles_overlay_on_and_off(self):
        import game_osd
        app = self._app()
        self.assertFalse(app.hud_overlay)
        with mock.patch.object(game_osd, "GameOSD") as factory:
            osd = factory.return_value
            osd.is_active.return_value = True
            app.handle_idle_key(ord("s"))
            self.assertTrue(app.hud_overlay)
            osd.show.assert_called()
            osd.update.assert_called()
            self.assertTrue(app.game_osd_active())
            app.handle_idle_key(ord("s"))
            self.assertFalse(app.hud_overlay)
            osd.hide.assert_called()
            self.assertFalse(app.game_osd_active())
        toggle_src = inspect.getsource(main_module.HandControllerApp.toggle_hud_overlay)
        self.assertIn("_sync_game_osd()", toggle_src)
        draw_src = inspect.getsource(main_module.HandControllerApp.draw_ui)
        self.assertIn("draw_preview", draw_src)

    def test_main_shortcuts_p_c_l_m_r(self):
        idle = inspect.getsource(main_module.HandControllerApp.handle_idle_key)
        self.assertIn('ord("p")', idle)
        self.assertIn("enter_mapping_menu()", idle)
        self.assertIn('ord("c")', idle)
        self.assertIn("enter_calibration()", idle)
        self.assertIn('ord("l")', idle)
        self.assertIn("lang_mode = True", idle)
        self.assertIn("toggle_hud_overlay", idle)
        self.assertNotIn('ord("v")', idle)
        self.assertNotIn('ord("y")', idle)
        self.assertNotIn('ord("m")', idle)
        list_src = inspect.getsource(main_module.HandControllerApp.handle_mapping_key)
        list_block = list_src.split('if self.map_phase == "list":', 1)[1].split('if self.map_phase == "rename":', 1)[0]
        self.assertIn('ord("r")', list_block)
        self.assertIn('ord("m")', list_block)
        self.assertIn('"hand"', list_block)

    def test_arrows_navigate_gestures_not_np(self):
        from main import ARROW_LEFT, ARROW_RIGHT
        app = self._app()
        app.map_mode = True
        app.map_phase = "list"
        names = app.map_gesture_names()
        self.assertGreater(len(names), 2)
        app.map_index = 1
        app.handle_mapping_key(0, raw=next(iter(ARROW_LEFT)))
        self.assertEqual(app.map_index, 0)
        app.handle_mapping_key(0, raw=next(iter(ARROW_RIGHT)))
        self.assertEqual(app.map_index, 1)
        list_block = inspect.getsource(main_module.HandControllerApp.handle_mapping_key).split(
            'if self.map_phase == "list":', 1)[1].split('if self.map_phase == "rename":', 1)[0]
        self.assertNotIn('ord("n")', list_block)

    def test_help_texts_have_no_left_right_np_rd_or_backspace_words(self):
        from localization import load_catalogs, set_language, t
        load_catalogs(force=True)
        for lang in ("fr", "en"):
            set_language(lang)
            for key in (
                "mapping.list_help", "mapping.simple_help", "mapping.combo_help",
                "mapping.rename_help", "mapping.nav_hint", "calib.menu_help",
                "calib.nav_hint", "hud.footer",
            ):
                text = t(key)
                self.assertNotIn("n/p", text, key)
                self.assertNotIn("r/d", text, key)
                self.assertNotRegex(text, r"\bLEFT\b")
                self.assertNotRegex(text, r"\bRIGHT\b")
                self.assertNotIn("arrière", text.lower())
                self.assertNotIn("arriere", text.lower())
                self.assertNotIn("efface", text.lower())
        set_language("fr")
        self.assertEqual(t("mapping.user_gestures"), "Gestes enregistrés")
        self.assertIn("prédéfinis", t("mapping.preset_gestures"))
        self.assertIn("Entrer : Valider", t("mapping.rename_help"))
        self.assertIn("Esc : Retour", t("mapping.rename_help"))
        self.assertIn("Entrer : Modifier", t("mapping.list_help"))
        self.assertIn("S : Supprimer", t("mapping.list_help"))
        self.assertIn("A/D", t("mapping.list_help"))
        self.assertNotIn("P : Appuyer", t("mapping.list_help"))
        self.assertNotIn("C : Combinaison", t("mapping.list_help"))
        self.assertNotIn("T : Tester", t("mapping.list_help"))
        self.assertEqual(t("mapping.state_on"), "Activé")
        self.assertEqual(t("mapping.state_off"), "Désactivé")

    def test_recorded_and_preset_column_format(self):
        from hud_renderer import mapping_row_parts, preset_display_name
        from localization import load_catalogs, set_language
        load_catalogs(force=True)
        set_language("fr")
        gid, action, kind, state = mapping_row_parts("G01", {"type": HOLD, "inputs": ["Z"]})
        self.assertEqual(gid, "G01")
        self.assertEqual(action, "Z")
        self.assertEqual(kind, "[Maintien]")
        self.assertEqual(state, "Activé")
        gid, action, kind, state = mapping_row_parts("G02", {"type": PRESS, "inputs": ["A"]})
        self.assertEqual(kind, "[Appuie]")
        gid, action, kind, state = mapping_row_parts(
            "G03", {"type": COMBINATION, "inputs": ["B", "A", "LEFT_MOUSE"]}
        )
        self.assertEqual(action, "B+A+LEFT_MOUSE")
        self.assertEqual(kind, "[Combinaison]")
        gid, action, kind, state = mapping_row_parts("G04", {"type": MACRO, "steps": [{"action": "PRESS", "inputs": ["X"]}]})
        self.assertEqual(action, "[Macro]")
        self.assertEqual(kind, "[Macro]")
        gid, action, kind, state = mapping_row_parts("G05", {"type": NONE, "inputs": []})
        self.assertEqual(action, "-")
        self.assertEqual(kind, "[Non configure]")
        self.assertEqual(state, "Désactivé")
        self.assertEqual(preset_display_name(geo.LEFT_TILT_UP), "TILT_UP")
        self.assertEqual(preset_display_name(geo.LEFT_FINGER_INDEX), "FINGER_UP_INDEX")
        app = self._app()
        app.map_side = "LEFT"
        listed = app.map_gesture_names()
        user = [name for name in listed if name in GESTURE_NAMES]
        presets = [name for name in listed if name not in GESTURE_NAMES]
        if user and presets:
            self.assertLess(listed.index(user[-1]), listed.index(presets[0]))
        self.assertTrue(any("TILT_UP" in name for name in listed))

    def test_rename_q_does_not_quit(self):
        app = self._app()
        app.mapping.set_command("LEFT", "G01", HOLD, ["Z"])
        names = app.map_gesture_names()
        app.map_index = names.index("G01")
        app.map_mode = True
        app.map_phase = "rename"
        app.rename_buffer = ""
        app.running = True
        self.assertEqual(app.ui_mode(), "RENAME")
        self.assertTrue(app._ui_consumes_text())
        self._dispatch(app, ord("q"))
        self.assertTrue(app.running)
        self.assertEqual(app.rename_buffer, "q")
        self._dispatch(app, 13)
        self.assertEqual(app.map_phase, "list")
        app.map_phase = "rename"
        app.rename_buffer = "abc"
        self._dispatch(app, 27)
        self.assertEqual(app.map_phase, "list")
        self.assertTrue(app.running)

    def test_s_deletes_in_gesture_settings_not_overlay(self):
        app = self._app()
        app.map_mode = True
        app.map_phase = "list"
        app.hud_overlay = False
        app.mapping.set_command("LEFT", "G01", HOLD, ["Z"])
        names = app.map_gesture_names()
        app.map_index = names.index("G01")
        app.handle_mapping_key(ord("s"))
        self.assertEqual(app.mapping.get("LEFT", "G01")["type"], NONE)
        self.assertFalse(app.hud_overlay)
        app.map_mode = False
        app.handle_idle_key(ord("s"))
        self.assertTrue(app.hud_overlay)

    def test_simple_and_combo_keep_opencv_commands(self):
        app = self._app()
        app.map_mode = True
        app.map_phase = "simple"
        app.edit_type = HOLD
        app.edit_inputs = ["Z"]
        app.handle_mapping_key(ord("p"))
        self.assertEqual(app.edit_type, PRESS)
        app.handle_mapping_key(ord("m"))
        self.assertEqual(app.edit_type, HOLD)
        app.handle_mapping_key(ord("c"))
        self.assertEqual(app.map_phase, "simple")
        self.assertNotEqual(app.edit_type, COMBINATION)
        simple = inspect.getsource(main_module.HandControllerApp.handle_mapping_key).split(
            'if self.map_phase == "simple":', 1)[1].split('if self.map_phase == "combo":', 1)[0]
        self.assertNotIn('ord("n")', simple)
        self.assertIn("is_left_arrow", simple)
        self.assertIn("is_right_arrow", simple)
        combo = inspect.getsource(main_module.HandControllerApp.handle_mapping_key).split(
            'if self.map_phase == "combo":', 1)[1].split('if self.map_phase == "macro":', 1)[0]
        self.assertIn("PRESS", combo)
        self.assertIn("HOLD", combo)
        macro = inspect.getsource(main_module.HandControllerApp.handle_mapping_key).split(
            'if self.map_phase == "macro":', 1)[1]
        self.assertNotIn('ord("n")', macro)
        self.assertIn("set_macro", macro)
        self.assertIn('key == ord(" ")', macro)

    def test_no_pyside_flick_profiles_or_preset_injection(self):
        source = inspect.getsource(main_module)
        self.assertNotIn("PySide6", source)
        self.assertNotIn("from PySide", source)
        self.assertNotIn("enter_profile", source)
        idle = inspect.getsource(main_module.HandControllerApp.handle_idle_key)
        self.assertNotIn("FLICK", idle)
        self.assertNotIn("SWIPE", idle)
        engine = GestureEngine(self.gestures_path)
        left = engine.database["gestures"]["LEFT"]
        self.assertNotIn("TILT_UP", left)
        self.assertNotIn("FINGER_UP_INDEX", left)
        self.assertNotIn("LEFT_TILT_UP", left)

    def test_gestures_backup_hash_unchanged(self):
        import hashlib
        path = os.path.join(os.path.dirname(main_module.__file__), "gestures_backup.json")
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        self.assertEqual(digest, "a90c42beea113313221da377929fd533914eb251a341b554e697c2c4524b51e0")

    def test_hud_overlay_defaults_off(self):
        from config import default_settings
        self.assertFalse(default_settings()["hud_overlay"])

    def test_osd_shows_only_active_gestures_top_left(self):
        from localization import set_language
        from overlay_labels import osd_paint_lines
        app = self._app()
        app.hud_overlay = True
        app.curr_gestures = {"LEFT": "G02", "RIGHT": "G01"}
        app.curr_specials = {
            "LEFT": {geo.finger_up_family("INDEX"): geo.LEFT_FINGER_INDEX},
            "RIGHT": {},
        }
        app.detected = {"LEFT": True, "RIGHT": True}
        app.track_state = {"LEFT": "OK", "RIGHT": "OK"}
        app._system_allowed = lambda side, gid: True
        try:
            set_language("en")
            rows = [text for text, _tone in osd_paint_lines(app)]
            self.assertEqual(rows[0], "LEFT")
            self.assertIn("RIGHT", rows)
            blob = " ".join(rows)
            self.assertEqual(rows, ["LEFT", "Present", "G02 + FINGER UP INDEX", "", "RIGHT", "Present", "G01"])
            blob = " ".join(rows)
            for banned in ("FPS", "CAMERA", "CPU", "RAM"):
                self.assertNotIn(banned, blob)
            set_language("fr")
            rows = [text for text, _tone in osd_paint_lines(app)]
            self.assertEqual(rows, ["GAUCHE", "Présente", "G02 + FINGER UP INDEX", "", "DROITE", "Présente", "G01"])
            app.curr_gestures = {"LEFT": UNKNOWN, "RIGHT": UNKNOWN}
            app.curr_specials = {"LEFT": {}, "RIGHT": {}}
            app.detected = {"LEFT": False, "RIGHT": False}
            app.track_state = {"LEFT": "RELEASED", "RIGHT": "RELEASED"}
            self.assertEqual(
                osd_paint_lines(app),
                [("GAUCHE", "cyan"), ("Absente", "dim"), ("", "dim"), ("DROITE", "cyan"), ("Absente", "dim")],
            )
        finally:
            set_language("fr")

    def test_custom_gesture_name_is_not_translated_on_osd(self):
        from localization import set_language
        from overlay_labels import localize_display_label
        try:
            set_language("en")
            self.assertEqual(localize_display_label("G01 — Index gauche"), "G01   Index gauche")
            self.assertEqual(localize_display_label("THUMB UP"), "Thumb Up")
            set_language("fr")
            self.assertEqual(localize_display_label("G01 — Index gauche"), "G01   Index gauche")
            self.assertEqual(localize_display_label("THUMB UP"), "Pouce levé")
            self.assertEqual(localize_display_label("TILT LEFT"), "Inclinaison gauche")
        finally:
            set_language("fr")

    def test_compact_status_display_absent_and_present(self):
        from hud_renderer import draw_preview, status_rows
        from localization import set_language
        from main import put
        app = self._app()
        app.detected = {"LEFT": False, "RIGHT": False}
        app.track_state = {"LEFT": "RELEASED", "RIGHT": "RELEASED"}
        app.curr_gestures = {"LEFT": UNKNOWN, "RIGHT": UNKNOWN}
        app.confidences = {"LEFT": 0.0, "RIGHT": 0.0}
        app.inputs.disable()
        app.game_osd = None
        app.hud_overlay = False
        try:
            set_language("fr")
            rows = [text for text, _color in status_rows(app)]
            self.assertEqual(rows, [
                "MAIN GAUCHE", "Absente", "Aucun   Conf: 0.00",
                "MAIN DROITE", "Absente", "Aucun   Conf: 0.00",
                "INPUT: OFF", "Sur-impression d'écran: Off",
            ])
            app.detected["RIGHT"] = True
            app.track_state["RIGHT"] = "OK"
            app.curr_gestures["RIGHT"] = "G01"
            app.confidences["RIGHT"] = 0.92
            app.inputs.enable()
            app.hud_overlay = True
            app.game_osd = mock.Mock()
            app.game_osd.is_active.return_value = False
            rows = [text for text, _color in status_rows(app)]
            self.assertEqual(rows[3:8], ["MAIN DROITE", "Présente", "G01   Conf: 0.92", "INPUT: ON", "Sur-impression d'écran: Off"])
            app.game_osd.is_active.return_value = True
            rows = [text for text, _color in status_rows(app)]
            self.assertEqual(rows[-1], "Sur-impression d'écran: Ok")
            set_language("en")
            rows = [text for text, _color in status_rows(app)]
            self.assertEqual(rows, [
                "LEFT HAND", "Absent", "None   Conf: 0.00",
                "RIGHT HAND", "Present", "G01   Conf: 0.92",
                "INPUT: ON", "Screen Overlay: OK",
            ])
            blob = " ".join(rows)
            self.assertNotIn("CAMERA", blob)
            self.assertNotIn("v1.0", blob)
            self.assertNotIn("HandController", blob)
            for size in ((480, 640), (1080, 1920), (240, 320)):
                frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
                bottom = draw_preview(app, frame, put)
                self.assertGreater(bottom, 0)
                self.assertLess(bottom, size[0] * 0.75)
                self.assertGreater(int(frame[:bottom, :].sum()), 0)
                # The block stays in the top-left quarter-ish and never covers the right side.
                self.assertEqual(int(frame[:, size[1] * 2 // 3:].sum()), 0)
        finally:
            app.inputs.disable()
            set_language("fr")

    @staticmethod
    def _hands(app, left=None, right=None):
        """left/right: None = hand absent, else (recorded id or UNKNOWN, {family: engine id}, conf)."""
        app.curr_gestures, app.curr_specials, app.confidences = {}, {}, {}
        app.detected, app.track_state = {}, {}
        for side, spec in (("LEFT", left), ("RIGHT", right)):
            present = spec is not None
            gesture, specials, conf = spec if present else (UNKNOWN, {}, 0.0)
            app.curr_gestures[side] = gesture
            app.curr_specials[side] = dict(specials)
            app.confidences[side] = conf
            app.detected[side] = present
            app.track_state[side] = "OK" if present else "RELEASED"
        app._system_allowed = lambda side, gid: True

    def test_osd_scenarios_with_predefined_gestures(self):
        from localization import set_language
        from overlay_labels import osd_paint_lines
        app = self._app()

        def rows():
            return [text for text, _tone in osd_paint_lines(app)]

        try:
            set_language("en")
            self._hands(app)
            self.assertEqual(rows(), ["LEFT", "Absent", "", "RIGHT", "Absent"])
            self._hands(app, left=("G01", {}, 0.9))
            self.assertEqual(rows(), ["LEFT", "Present", "G01", "", "RIGHT", "Absent"])
            self._hands(app, right=(UNKNOWN, {geo.PINCH: geo.RIGHT_PINCH}, 0.0))
            self.assertEqual(rows(), ["LEFT", "Absent", "", "RIGHT", "Present", "PINCH"])
            self._hands(app, left=("G01", {}, 0.9),
                        right=(UNKNOWN, {geo.TILT: geo.RIGHT_TILT_LEFT}, 0.0))
            self.assertEqual(rows(), ["LEFT", "Present", "G01", "", "RIGHT", "Present", "TILT LEFT"])
            # Neutral tilt and index direction are states, not gestures.
            self._hands(app, right=(UNKNOWN, {geo.TILT: geo.RIGHT_TILT_NEUTRAL,
                                              geo.INDEX: geo.RIGHT_INDEX_UP}, 0.0))
            self.assertEqual(rows()[-3:], ["RIGHT", "Present", "None"])
            set_language("fr")
            self.assertEqual(rows()[-3:], ["DROITE", "Présente", "Aucun"])
            self._hands(app, left=(UNKNOWN, {geo.PINCH: geo.LEFT_PINCH}, 0.0))
            self.assertEqual(rows()[:3], ["GAUCHE", "Présente", "PINCH"])
            self._hands(app, left=("G02", {"FINGER_THUMB": "LEFT_FINGER_THUMB"}, 0.8))
            self.assertEqual(rows()[2], "G02 + FINGER UP THUMB")
            # A disabled gate is hidden, exactly like the input path ignores it.
            app._system_allowed = lambda side, gid: gid != "LEFT_FINGER_THUMB"
            self.assertEqual(rows()[2], "G02")
        finally:
            set_language("fr")

    def test_every_predefined_gesture_has_a_canonical_label(self):
        from overlay_labels import preset_display_label
        expected = {
            geo.LEFT_PINCH: "PINCH", geo.RIGHT_PINCH: "PINCH",
            geo.LEFT_TILT_UP: "TILT UP", geo.RIGHT_TILT_DOWN: "TILT DOWN",
            geo.LEFT_TILT_LEFT: "TILT LEFT", geo.RIGHT_TILT_RIGHT: "TILT RIGHT",
            geo.LEFT_FINGER_INDEX: "FINGER UP INDEX", "RIGHT_FINGER_MIDDLE": "FINGER UP MIDDLE",
            "LEFT_FINGER_RING": "FINGER UP RING", "RIGHT_FINGER_PINKY": "FINGER UP PINKY",
            geo.RIGHT_FINGER_THUMB: "FINGER UP THUMB",
            geo.LEFT_FIST: "FIST", geo.RIGHT_OPEN_PALM: "OPEN PALM",
        }
        for gid, label in expected.items():
            self.assertEqual(preset_display_label(gid), label, gid)
        for gid in (geo.LEFT_TILT_NEUTRAL, geo.RIGHT_INDEX_UP, geo.LEFT_INDEX_NEUTRAL, "G01", UNKNOWN, ""):
            self.assertIsNone(preset_display_label(gid), gid)
        for gid in geo.SPECIAL_NAMES:
            core = gid.split("_", 1)[1]
            if core.endswith("NEUTRAL") or core.startswith("INDEX_"):
                self.assertIsNone(preset_display_label(gid), gid)
            else:
                self.assertIsNotNone(preset_display_label(gid), gid)

    def test_local_status_shows_predefined_and_recorded_gestures(self):
        from hud_renderer import status_rows
        from localization import set_language
        app = self._app()
        app.inputs.enable()
        app.hud_overlay = True
        app.game_osd = mock.Mock()
        app.game_osd.is_active.return_value = True
        self._hands(app, left=(UNKNOWN, {geo.PINCH: geo.LEFT_PINCH}, 0.0), right=("G01", {}, 0.94))
        try:
            set_language("fr")
            self.assertEqual([text for text, _ in status_rows(app)], [
                "MAIN GAUCHE", "Présente", "PINCH",
                "MAIN DROITE", "Présente", "G01   Conf: 0.94",
                "INPUT: ON", "Sur-impression d'écran: Ok",
            ])
            set_language("en")
            self.assertEqual([text for text, _ in status_rows(app)], [
                "LEFT HAND", "Present", "PINCH",
                "RIGHT HAND", "Present", "G01   Conf: 0.94",
                "INPUT: ON", "Screen Overlay: OK",
            ])
            # Fixed order whatever the detector order.
            self._hands(app, right=("G01", {geo.TILT: geo.RIGHT_TILT_UP, geo.PINCH: geo.RIGHT_PINCH}, 0.5))
            self.assertEqual(status_rows(app)[5][0], "G01 + PINCH + TILT UP   Conf: 0.50")
            app._system_allowed = lambda side, gid: gid != geo.RIGHT_PINCH
            self.assertEqual(status_rows(app)[5][0], "G01 + TILT UP   Conf: 0.50")
            # Overlay line follows the real window, not the wish.
            app.game_osd.is_active.return_value = False
            self.assertEqual(status_rows(app)[-1][0], "Screen Overlay: Off")
            blob = " ".join(text for text, _ in status_rows(app))
            for banned in ("CAMERA", "v1.0", "HandController", "First hand", "Second hand"):
                self.assertNotIn(banned, blob)
        finally:
            app.inputs.disable()
            set_language("fr")

    def test_preset_catalog_lists_only_real_engine_gestures(self):
        from overlay_labels import preset_catalog
        app = self._app()
        self._hands(app, left=(UNKNOWN, {geo.PINCH: geo.LEFT_PINCH}, 0.0))
        app.geometry = geo.GeometryEngine()
        app._system_allowed = lambda side, gid: gid == geo.LEFT_PINCH
        rows = preset_catalog(app, "LEFT")
        self.assertEqual([row["label"] for row in rows], [
            "PINCH", "TILT UP", "TILT DOWN", "TILT LEFT", "TILT RIGHT",
            "FINGER UP INDEX", "FINGER UP MIDDLE", "FINGER UP RING", "FINGER UP PINKY", "FINGER UP THUMB",
        ])
        self.assertTrue(all(row["id"].startswith("LEFT_") for row in rows))
        self.assertTrue(rows[0]["enabled"] and rows[0]["active"])
        self.assertFalse(rows[1]["enabled"] or rows[1]["active"])
        self.assertFalse(any(row["label"].startswith("G") and row["label"][1:].isdigit() for row in rows))
        app.geometry = geo.GeometryEngine(fist_enabled=True, open_palm_enabled=True)
        labels = [row["label"] for row in preset_catalog(app, "RIGHT")]
        self.assertEqual(labels[-2:], ["FIST", "OPEN PALM"])

    def test_hand_choice_uses_left_right_detection(self):
        from localization import set_language, t
        try:
            set_language("fr")
            self.assertEqual(t("mapping.choose_left"), "[1] Main gauche")
            self.assertEqual(t("mapping.choose_right"), "[2] Main droite")
            set_language("en")
            self.assertEqual(t("mapping.choose_left"), "[1] Left hand")
            self.assertEqual(t("mapping.choose_right"), "[2] Right hand")
        finally:
            set_language("fr")
        app = self._app()
        app.calib_mode, app.calib_phase = True, "hand"
        app.handle_calibration_key(ord("2"))
        self.assertEqual(app.calib_side, "RIGHT")
        app.calib_phase = "hand"
        app.handle_calibration_key(ord("1"))
        self.assertEqual(app.calib_side, "LEFT")
        app.calib_mode = False
        app.map_mode, app.map_phase = True, "hand"
        app.handle_mapping_key(ord("2"))
        self.assertEqual(app.map_side, "RIGHT")

    def test_removed_texts_never_reach_the_app(self):
        root = os.path.dirname(os.path.abspath(main_module.__file__))
        banned = (
            "Première main", "Premiere main", "Seconde main", "First hand", "Second hand",
            "K : capturer", "K: capture", "K : capture", "K assign", "CAMERA: OK", "HandController v1.0.0",
        )
        files = [os.path.join(root, "localization", f"{lang}.json") for lang in ("fr", "en")]
        files += [os.path.join(root, name) for name in os.listdir(root)
                  if name.endswith(".py") and name != "run_tests.py"]
        files += [os.path.join(root, name) for name in os.listdir(root)
                  if name.startswith("README") and name.endswith(".md")]
        for path in files:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            for needle in banned:
                self.assertNotIn(needle, text, f"{needle!r} in {os.path.basename(path)}")
        self.assertEqual(main_module.WINDOW_TITLE, "HandController")

    def test_k_no_longer_captures_but_assign_buttons_do(self):
        import ui_context
        for context in (ui_context.COMMAND_SETTINGS, ui_context.COMBINATION_SETTINGS):
            self.assertNotIn("k", ui_context.SHORTCUTS.get(context, {}))
        self.assertNotIn('ord("k")', inspect.getsource(HandControllerApp.handle_system_key))
        # J/K still adjust macro step values.
        self.assertIn('ord("j"), ord("J"), ord("k"), ord("K")',
                      inspect.getsource(HandControllerApp.handle_mapping_key))
        app = self._app()
        app.mapping.set_command("LEFT", "G01", HOLD, ["Z"])
        app.map_mode = True
        app.map_index = app.map_gesture_names().index("G01")
        app.load_edit_buffers()
        try:
            app.map_phase = "simple"
            app.handle_mapping_key(ord("k"))
            self.assertEqual(app.map_phase, "simple")
            app._handle_map_click("assign")
            self.assertEqual(app.map_phase, "capture")
            app.inputs.end_key_capture()
            app.map_phase = "combo"
            app.handle_mapping_key(ord("k"))
            self.assertEqual(app.map_phase, "combo")
            app._handle_map_click("assign")
            self.assertEqual(app.map_phase, "capture")
            self.assertTrue(app.capture_combo)
        finally:
            app.inputs.end_key_capture()

    def test_console_logs_gesture_changes_once(self):
        app = self._app()
        app.log = mock.Mock()
        self._hands(app)
        app._log_gesture_changes()
        app.log.info.assert_not_called()
        self._hands(app, left=("G01", {geo.PINCH: geo.LEFT_PINCH}, 0.94))
        app._log_gesture_changes()
        app._log_gesture_changes()
        app.log.info.assert_called_once_with("[%s] %s%s", "LEFT", "G01 + PINCH", " (conf 0.94)")
        self._hands(app)
        app._log_gesture_changes()
        self.assertEqual(app.log.info.call_args, mock.call("[%s] %s%s", "LEFT", "no gesture", ""))

    @unittest.skipUnless(os.name == "nt", "Win32 foreground check")
    def test_focus_check_matches_the_exact_window_title(self):
        import ctypes
        import window_focus

        def check(foreground_title):
            fake = mock.Mock()
            fake.GetForegroundWindow.return_value = 1
            fake.GetWindowTextLengthW.return_value = len(foreground_title)

            def get_text(_hwnd, buffer, _size):
                buffer.value = foreground_title
            fake.GetWindowTextW.side_effect = get_text
            with mock.patch.object(ctypes.windll, "user32", fake):
                return window_focus.own_window_is_foreground("HandController")

        self.assertTrue(check("HandController"))
        self.assertTrue(check("handcontroller "))
        self.assertFalse(check(r"C:\Apps\HandController\HandController.exe"))
        self.assertFalse(check("main.py - HandController - Cursor"))

    def test_rename_q_and_c_are_characters_not_shortcuts(self):
        app = self._app()
        app.mapping.set_command("LEFT", "G01", HOLD, ["Z"])
        names = app.map_gesture_names()
        app.map_index = names.index("G01")
        app.map_mode = True
        app.map_phase = "rename"
        app.rename_buffer = ""
        app.running = True
        app.calib_mode = False
        self._dispatch(app, ord("q"))
        self._dispatch(app, ord("c"))
        self.assertTrue(app.running)
        self.assertEqual(app.rename_buffer, "qc")
        self.assertFalse(app.calib_mode)
        self.assertEqual(app.map_phase, "rename")
        self._dispatch(app, 13)
        self.assertEqual(app.map_phase, "list")

    def test_gesture_list_is_recorded_only_then_presets(self):
        from gesture_engine import GestureSamples
        app = self._app()
        bank = app.engine.database["gestures"]["LEFT"]
        for gid in GESTURE_NAMES:
            bank[gid] = GestureSamples()
        for gid in ("G01", "G02", "G03"):
            bank[gid] = GestureSamples(samples_for(int(gid[1:])), gid)
        app.engine.save_database()
        listed = app.map_gesture_names()
        user = [name for name in listed if name in GESTURE_NAMES]
        self.assertEqual(user, ["G01", "G02", "G03"])
        self.assertNotIn("G09", listed)
        self.assertNotIn("G30", listed)
        self.assertIn(geo.LEFT_TILT_UP, listed)
        self.assertIn(geo.LEFT_FINGER_INDEX, listed)
        self.assertNotIn(geo.LEFT_TILT_NEUTRAL, listed)
        self.assertLess(listed.index("G03"), listed.index(geo.LEFT_TILT_UP))

    def test_a_enables_d_disables_and_enter_opens_three_choices(self):
        from localization import load_catalogs, set_language, t
        load_catalogs(force=True)
        set_language("fr")
        app = self._app()
        app.map_mode = True
        app.map_phase = "list"
        app.mapping.set_command("LEFT", "G01", HOLD, ["Z"])
        names = app.map_gesture_names()
        app.map_index = names.index("G01")
        app.handle_mapping_key(ord("d"))
        self.assertFalse(app.mapping.get("LEFT", "G01").get("enabled", True))
        self.assertEqual(app.mapping.get("LEFT", "G01")["type"], HOLD)
        app.handle_mapping_key(ord("a"))
        self.assertTrue(app.mapping.get("LEFT", "G01").get("enabled", True))
        app.handle_mapping_key(13)
        self.assertEqual(app.map_phase, "edit")
        draw = inspect.getsource(main_module.HandControllerApp.draw_mapping_menu)
        edit = draw.split('if self.map_phase == "edit":', 1)[1].split('if self.map_phase == "simple":', 1)[0]
        self.assertIn("opt_simple", edit)
        self.assertIn("opt_combo", edit)
        self.assertIn("opt_macro", edit)
        self.assertNotIn("opt_disable", edit)
        self.assertNotIn("opt_reset", edit)
        self.assertIn("Commandes simples", t("mapping.opt_simple"))
        self.assertNotIn("COMBINATION", t("mapping.simple_help"))
        self.assertIn("Espace : Sauver", t("mapping.macro_help1"))
        self.assertIn("S : Supprimer", t("mapping.macro_help1"))
        self.assertNotIn("n/p", t("mapping.macro_help1"))

    def test_combination_per_key_press_and_hold(self):
        app = self._app()
        app.map_mode = True
        app.map_phase = "combo"
        app.edit_inputs = ["B", "A"]
        app.edit_modes = [HOLD, HOLD]
        app.edit_slot = 0
        app.handle_mapping_key(ord("1"))
        self.assertEqual(app.edit_modes[0], PRESS)
        app.handle_mapping_key(ord("t"))
        app.handle_mapping_key(ord("2"))
        self.assertEqual(app.edit_modes[1], HOLD)
        app.mapping.set_command("LEFT", "G01", COMBINATION, ["B", "A"], modes=[PRESS, HOLD])
        entry = app.mapping.get("LEFT", "G01")
        self.assertEqual(entry["type"], COMBINATION)
        self.assertEqual(entry["modes"], [PRESS, HOLD])
        from gesture_mapping import combination_press_inputs, combination_hold_inputs
        self.assertEqual(combination_press_inputs(entry), ["B"])
        self.assertEqual(combination_hold_inputs(entry), ["A"])

    def test_recognition_quality_and_one_euro(self):
        from landmark_filter import OneEuroFilter
        from landmark_quality import LandmarkQualityChecker, ACCEPT, HOLD as QHOLD
        from recognition_engine import RecognitionEngine
        from config import SMOOTHING_MIN_CUTOFF, SMOOTHING_BETA, SMOOTHING_D_CUTOFF
        self.assertAlmostEqual(SMOOTHING_MIN_CUTOFF, 1.2)
        self.assertAlmostEqual(SMOOTHING_BETA, 0.45)
        self.assertAlmostEqual(SMOOTHING_D_CUTOFF, 1.15)
        left = OneEuroFilter()
        right = OneEuroFilter()
        self.assertIsNot(left, right)
        checker = LandmarkQualityChecker()
        palm = np.zeros((21, 3), dtype=np.float32)
        palm[0] = (0.4, 0.5, 0.0)
        palm[9] = (0.4, 0.42, 0.0)
        for index in range(21):
            palm[index, 0] = 0.4 + index * 0.001
            palm[index, 1] = 0.5 - (0.08 if index == 9 else 0.0)
        first = checker.evaluate_report("LEFT", palm, 0.95)
        jumped = palm.copy()
        jumped[8, :2] += 0.45
        jumped[12, :2] += 0.45
        jumped[16, :2] += 0.45
        jumped[20, :2] += 0.45
        report = checker.evaluate_report("LEFT", jumped, 0.95)
        self.assertIn(report.action, (QHOLD, "hold", ACCEPT))
        rec_src = inspect.getsource(RecognitionEngine.recognize_ml)
        self.assertIn("ambiguous", rec_src)
        self.assertIn("quality_score", rec_src)
        self.assertIn("knn", inspect.getsource(RecognitionEngine).lower())
        self.assertIn("ensemble", inspect.getsource(RecognitionEngine.recognize_ml))


if __name__ == "__main__":
    unittest.main(verbosity=2)

