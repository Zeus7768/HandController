"""Startup checks. Does not send input and does not modify user files."""
import importlib
import json
import os
import sys

import config
from localization import t


def _row(label, ok, detail, state=None):
    shown = state or ("OK" if ok else "ERROR")
    return f"{label:<16}: {shown:<6} {detail}".rstrip(), ok


def _import_version(module_name, label, version_attr="__version__"):
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        return _row(label, False, str(error))
    version = getattr(module, version_attr, "")
    detail = str(version) if version else "imported"
    return _row(label, True, detail)


def _json_file(label, path):
    if not os.path.isfile(path):
        return _row(label, True, t("diag.absent"))
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            json.load(handle)
    except json.JSONDecodeError:
        return _row(label, False, t("diag.invalid_json"))
    except OSError as error:
        return _row(label, False, str(error))
    return _row(label, True, os.path.basename(path))


def _camera_row(index):
    try:
        import cv2
        capture = cv2.VideoCapture(int(index))
        opened = bool(capture is not None and capture.isOpened())
        frame_ok = False
        if opened:
            frame_ok, _frame = capture.read()
        capture.release()
    except Exception as error:
        return _row("Camera", False, str(error))
    if opened and frame_ok:
        return _row("Camera", True, t("diag.camera_ok", index=index))
        return _row("Camera", False, t("diag.camera_fail", index=index))


def collect_diagnostic(check_camera=False, camera_index=0):
    """Return (ready, lines). A missing optional file is not a failure."""
    lines = [t("diag.title"), "-" * 40]
    checks = []
    checks.append(_row("Python", sys.version_info >= (3, 10), sys.version.split()[0]))
    checks.append(_import_version("cv2", "OpenCV"))
    checks.append(_import_version("mediapipe", "MediaPipe"))
    checks.append(_import_version("numpy", "NumPy"))
    try:
        importlib.import_module("pynput")
        checks.append(_row("pynput", True, "imported"))
    except Exception as error:
        checks.append(_row("pynput", False, str(error)))
    try:
        import sklearn
        checks.append(_row("sklearn", True, getattr(sklearn, "__version__", "imported")))
    except Exception as error:
        checks.append(_row("sklearn", True, str(error), state="SKIP"))
    model = config.resource_path(config.MODEL_FILE)
    checks.append(_row("Model", os.path.isfile(model), model if os.path.isfile(model) else t("diag.model_missing")))
    checks.append(_json_file("Gesture DB", config.user_data_path(config.GESTURES_FILE)))
    checks.append(_json_file("Mappings", config.user_data_path(config.MAPPING_FILE)))
    checks.append(_json_file("Settings", config.user_data_path(config.SETTINGS_FILE)))
    checks.append(_row("Input", True, t("diag.input")))
    try:
        importlib.import_module("landmark_replay")
        checks.append(_row("Replay", True, t("diag.replay")))
    except Exception as error:
        checks.append(_row("Replay", False, str(error)))
    checks.append(_row("Recording", True, t("diag.recording")))
    if check_camera:
        checks.append(_camera_row(camera_index))
    else:
        checks.append(_row("Camera", True, t("diag.camera_skip"), state="SKIP"))
    ready = all(ok for _line, ok in checks)
    for line, _ok in checks:
        lines.append(line)
    lines.append("-" * 40)
    lines.append(t("diag.ready") if ready else t("diag.not_ready"))
    return ready, lines


def format_report(check_camera=False, camera_index=0):
    ready, lines = collect_diagnostic(check_camera, camera_index)
    return ready, "\n".join(lines)
