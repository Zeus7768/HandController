@echo off
REM HandController 1.0.0 - reproducible Windows build (PyInstaller onedir).
REM Usage: build_exe.bat          build + verify
REM        build_exe.bat --zip    build + verify + release\HandController-v1.0.0-Windows.zip
setlocal
cd /d "%~dp0"

echo === HandController build ===

where python >nul 2>nul
if errorlevel 1 (
    echo [FAIL] Python not found in PATH.
    exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo [FAIL] Python 3.10 or newer is required.
    exit /b 1
)
for /f "delims=" %%v in ('python --version') do echo [OK] %%v

python -c "import cv2, mediapipe, numpy, pynput" 2>nul
if errorlevel 1 (
    echo [FAIL] Missing dependencies. Run: pip install -r requirements.txt
    exit /b 1
)
echo [OK] opencv, mediapipe, numpy, pynput

python -m PyInstaller --version >nul 2>nul
if errorlevel 1 (
    echo [FAIL] PyInstaller missing. Run: pip install pyinstaller
    exit /b 1
)
for /f "delims=" %%v in ('python -m PyInstaller --version') do echo [OK] PyInstaller %%v

for %%f in (HandController.spec admin.manifest hand_landmarker.task localization\fr.json localization\en.json assets\handcontroller_icon.ico assets\handcontroller_icon.png assets\handcontroller_icon_transparent.png) do (
    if not exist "%%f" (
        echo [FAIL] Missing %%f
        exit /b 1
    )
)
echo [OK] source resources present

REM The previous dist\ build stays untouched until the new one is built and verified.
echo --- cleaning old build\ ---
if exist build rmdir /s /q build

echo --- PyInstaller (staging: build\stage) ---
python -m PyInstaller --noconfirm --clean --distpath build\stage --workpath build\work HandController.spec
if errorlevel 1 (
    echo [FAIL] PyInstaller failed. The previous dist\ build was kept.
    exit /b 1
)

echo --- verification (staging) ---
python tools\verify_build.py --dist build\stage\HandController
if errorlevel 1 (
    echo [FAIL] Build verification failed. The previous dist\ build was kept.
    exit /b 1
)

echo --- replacing dist\HandController ---
if exist dist rmdir /s /q dist
if exist dist (
    echo [FAIL] dist\ is locked - close HandController.exe. New build kept in build\stage\HandController.
    exit /b 1
)
mkdir dist
move build\stage\HandController dist\HandController >nul
if errorlevel 1 (
    echo [FAIL] Could not move the new build into dist\.
    exit /b 1
)

echo --- verification (dist) ---
python tools\verify_build.py %1
if errorlevel 1 (
    echo [FAIL] Build verification failed.
    exit /b 1
)

echo --- cleaning build\ ---
rmdir /s /q build

echo.
echo === Done: dist\HandController\HandController.exe ===
endlocal
exit /b 0
