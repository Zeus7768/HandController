"""Verify a PyInstaller onedir build and optionally pack the release ZIP.

Usage:
    python tools/verify_build.py              verify dist/HandController
    python tools/verify_build.py --zip        verify, then write release/HandController-v1.0.0-Windows.zip

The ZIP only contains end-user files. User data created by a test run
(settings.json, gestures*.json, mapping.json, logs/, recordings/) is never packed.
"""
from __future__ import annotations

import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist", "HandController")
EXE = os.path.join(DIST, "HandController.exe")
RELEASE_DIR = os.path.join(ROOT, "release")
ZIP_ROOT = "HandController-v1.0.0-Windows"
ZIP_NAME = ZIP_ROOT + ".zip"

REQUIRED_RESOURCES = (
    "hand_landmarker.task",
    os.path.join("localization", "fr.json"),
    os.path.join("localization", "en.json"),
    os.path.join("assets", "handcontroller_icon.ico"),
    os.path.join("assets", "handcontroller_icon.png"),
    os.path.join("assets", "handcontroller_icon_transparent.png"),
)
# (source in the repository, name inside the ZIP)
RELEASE_DOCS = (
    (os.path.join("docs", "README_WINDOWS.txt"), "README.txt"),
    ("LICENSE", "LICENSE.txt"),
    ("THIRD_PARTY_NOTICES.txt", "THIRD_PARTY_NOTICES.txt"),
)
LICENSES_DIR = "licenses"
USER_DATA = {"settings.json", "gestures.json", "gestures_backup.json", "mapping.json"}
USER_DIRS = {"logs", "recordings", "profiles", "cache", "temp"}
REQUIRED_ICO_SIZES = (16, 32, 48, 64, 128, 256)


def _resource_root(dist):
    internal = os.path.join(dist, "_internal")
    return internal if os.path.isdir(internal) else dist


def exe_has_admin_manifest(path):
    with open(path, "rb") as handle:
        data = handle.read()
    return b"requireAdministrator" in data


def exe_subsystem(path):
    """PE optional-header subsystem: 3 = console (CUI), 2 = windowed (GUI)."""
    import struct

    with open(path, "rb") as handle:
        data = handle.read(4096)
    if data[:2] != b"MZ":
        return None
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        return None
    return struct.unpack_from("<H", data, pe + 24 + 68)[0]


def ico_sizes(path):
    import struct

    with open(path, "rb") as handle:
        data = handle.read()
    _reserved, _kind, count = struct.unpack_from("<HHH", data, 0)
    sizes = []
    for index in range(count):
        width = data[6 + 16 * index]
        sizes.append(256 if width == 0 else width)
    return sorted(sizes)


def exe_icon_count(path):
    if os.name != "nt":
        return -1
    import ctypes
    shell32 = ctypes.windll.shell32
    shell32.ExtractIconExW.restype = ctypes.c_uint
    return int(shell32.ExtractIconExW(path, -1, None, None, 0))


def verify(dist=DIST):
    problems = []
    exe = os.path.join(dist, "HandController.exe")
    if not os.path.isfile(exe):
        return ["missing " + exe]
    root = _resource_root(dist)
    for rel in REQUIRED_RESOURCES:
        if not os.path.isfile(os.path.join(root, rel)):
            problems.append("missing resource " + rel)
    if not exe_has_admin_manifest(exe):
        problems.append("UAC manifest (requireAdministrator) not embedded")
    if exe_subsystem(exe) != 3:
        problems.append("EXE is not a console build (logs would not be visible)")
    ico = os.path.join(root, "assets", "handcontroller_icon.ico")
    if os.path.isfile(ico):
        missing = [size for size in REQUIRED_ICO_SIZES if size not in ico_sizes(ico)]
        if missing:
            problems.append("ICO lacks sizes %s" % missing)
    icons = exe_icon_count(exe)
    if icons == 0:
        problems.append("no icon embedded in the EXE")
    for banned in ("PySide6", "PyQt5", "PyQt6", "_sounddevice_data"):
        if os.path.isdir(os.path.join(root, banned)):
            problems.append(banned + " was bundled")
    return problems


def _release_entries(dist):
    for base, dirs, files in os.walk(dist):
        rel_base = os.path.relpath(base, dist)
        if rel_base == ".":
            dirs[:] = [d for d in dirs if d not in USER_DIRS]
        for name in files:
            if rel_base == "." and (name in USER_DATA or name.endswith((".log", ".bak", ".tmp"))):
                continue
            full = os.path.join(base, name)
            yield full, os.path.join(ZIP_ROOT, os.path.relpath(full, dist))
    for source, arcname in RELEASE_DOCS:
        path = os.path.join(ROOT, source)
        if os.path.isfile(path):
            yield path, os.path.join(ZIP_ROOT, arcname)
    licenses = os.path.join(ROOT, LICENSES_DIR)
    for base, _dirs, files in os.walk(licenses):
        for name in files:
            full = os.path.join(base, name)
            yield full, os.path.join(ZIP_ROOT, LICENSES_DIR, os.path.relpath(full, licenses))


def make_release_zip(dist=DIST, out_dir=RELEASE_DIR):
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, ZIP_NAME)
    if os.path.exists(target):
        os.remove(target)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for full, arc in _release_entries(dist):
            archive.write(full, arc)
    return target


def main(argv):
    dist = DIST
    if "--dist" in argv:
        index = argv.index("--dist")
        if index + 1 >= len(argv):
            print("[FAIL] --dist needs a folder")
            return 1
        dist = os.path.abspath(argv[index + 1])
    exe = os.path.join(dist, "HandController.exe")
    problems = verify(dist)
    if problems:
        for item in problems:
            print("[FAIL] " + item)
        return 1
    print("[OK] " + exe + " (%d bytes)" % os.path.getsize(exe))
    print("[OK] resources: " + ", ".join(REQUIRED_RESOURCES))
    print("[OK] UAC manifest: requireAdministrator")
    print("[OK] console subsystem (log terminal opens with the EXE)")
    print("[OK] icons in EXE: %d, ICO sizes: %s" % (
        exe_icon_count(exe), ico_sizes(os.path.join(_resource_root(dist), "assets", "handcontroller_icon.ico"))))
    if "--zip" in argv:
        target = make_release_zip(dist)
        print("[OK] release: " + target + " (%d bytes)" % os.path.getsize(target))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
