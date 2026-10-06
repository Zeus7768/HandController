"""Copy the licence files of every third-party package bundled in HandController.exe.

Run after changing dependency versions:
    python tools/collect_licenses.py

Texts come from the installed package metadata (dist-info), so they match the
versions that PyInstaller packs. Output: licenses/<distribution>/...
licenses/GPL-3.0.txt is kept as is (the LGPL-3.0 of pynput refers to it).
"""
from __future__ import annotations

import importlib.metadata as metadata
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "licenses")
KEEP = {"GPL-3.0.txt", "README.md"}

# Distributions found in the PyInstaller archive of HandController.exe.
BUNDLED = (
    "mediapipe", "opencv-python", "opencv-contrib-python", "numpy", "pynput",
    "scikit-learn", "scipy", "joblib", "threadpoolctl", "matplotlib", "contourpy",
    "kiwisolver", "cycler", "fonttools", "pyparsing", "python-dateutil", "six",
    "pillow", "packaging", "certifi", "absl-py", "flatbuffers", "cffi", "pycparser",
    "cloudpickle", "narwhals", "setuptools",
)
MARKERS = ("licen", "copying", "notice", "authors")


def licence_files(dist):
    for entry in dist.files or ():
        name = entry.name.lower()
        if any(marker in name for marker in MARKERS) and ".dist-info" in str(entry):
            yield entry


def newest(dist_name):
    """Several versions can be installed side by side; Python imports the newest."""
    from packaging.version import Version

    wanted = dist_name.lower().replace("_", "-")
    found = [dist for dist in metadata.distributions()
             if (dist.metadata.get("Name") or "").lower().replace("_", "-") == wanted]
    if not found:
        raise metadata.PackageNotFoundError(dist_name)
    return max(found, key=lambda dist: Version(dist.version))


def collect():
    os.makedirs(OUT, exist_ok=True)
    for name in os.listdir(OUT):
        path = os.path.join(OUT, name)
        if name not in KEEP:
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    rows = []
    for dist_name in BUNDLED:
        try:
            dist = newest(dist_name)
        except metadata.PackageNotFoundError:
            rows.append((dist_name, "-", "not installed", 0))
            continue
        target = os.path.join(OUT, dist_name)
        copied = 0
        for entry in licence_files(dist):
            relative = str(entry).split(".dist-info/", 1)[1]
            if relative.startswith("licenses/"):
                relative = relative[len("licenses/"):]
            dest = os.path.join(target, *relative.split("/"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(dist.locate_file(entry), dest)
            copied += 1
        expression = dist.metadata.get("License-Expression") or dist.metadata.get("License") or ""
        if len(expression) > 60 or "\n" in expression:
            expression = expression.splitlines()[0][:60]
        rows.append((dist_name, dist.version, expression or "see files", copied))
    python_licence = os.path.join(sys.base_prefix, "LICENSE.txt")
    if os.path.isfile(python_licence):
        os.makedirs(os.path.join(OUT, "python"), exist_ok=True)
        shutil.copyfile(python_licence, os.path.join(OUT, "python", "LICENSE.txt"))
        rows.append(("python", sys.version.split()[0], "PSF-2.0", 1))
    return rows


if __name__ == "__main__":
    for row in collect():
        print("%-24s %-14s %-40s files=%d" % row)
