# -*- mode: python ; coding: utf-8 -*-
"""Reproducible Windows onedir build for HandController 1.0.0.

Requires administrator at launch (UAC). Bundles the MediaPipe model and
FR/EN catalogs. User data (settings.json, gestures.json, mapping.json, recordings/, logs/)
is created next to the EXE via app_dir(), never inside _MEIPASS.

Build:
    pyinstaller --noconfirm HandController.spec
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [
    ("hand_landmarker.task", "."),
    ("localization/fr.json", "localization"),
    ("localization/en.json", "localization"),
    ("assets/handcontroller_icon.png", "assets"),
    ("assets/handcontroller_icon_transparent.png", "assets"),
    ("assets/handcontroller_icon.ico", "assets"),
]
binaries = []
hiddenimports = [
    "localization",
    "game_osd",
    "overlay_labels",
    "hud_renderer",
    "pynput.keyboard._win32",
    "pynput.mouse._win32",
    "sklearn",
    "sklearn.svm",
    "sklearn.ensemble",
]

for package in ("mediapipe", "cv2"):
    collected_datas, collected_binaries, collected_hidden = collect_all(package)
    datas += collected_datas
    binaries += collected_binaries
    hiddenimports += collected_hidden

# Optional SVM / Random Forest: compiled helpers such as sklearn._cyutility are
# not found by static analysis. Test packages are skipped to keep the build small.
try:
    hiddenimports += collect_submodules("sklearn", filter=lambda name: ".tests" not in name)
except Exception:
    pass

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # sounddevice comes with MediaPipe's audio tasks; HandController never uses audio,
    # and its PortAudio binaries (ASIO builds) would add licence terms for nothing.
    excludes=["PySide6", "PySide2", "PyQt5", "PyQt6", "sounddevice", "_sounddevice", "_sounddevice_data"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HandController",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    uac_admin=True,
    uac_uiaccess=False,
    manifest="admin.manifest",
    icon="assets/handcontroller_icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="HandController",
)
