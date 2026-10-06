# HandController

**Version 1.0.0** — [Français](README_FR.md) · [Overview](README.md)

<img src="assets/handcontroller_icon.png" width="96" alt="HandController">

HandController is an **external webcam-based hand gesture controller** for Windows PCs and games.

It recognises your hand poses with a webcam, then simulates keyboard and mouse input. The goal: play or control any application with your hands, without special hardware and without touching the game.

```
Webcam → OpenCV → MediaPipe → Recognition Engine (k-NN / SVM / Random Forest + Geometry) → mapping → keyboard/mouse
```

## Download

Windows users can download the latest version from the [GitHub Releases page](../../releases/latest): take **`HandController-v1.0.0-Windows.zip`**, unzip it and run `HandController.exe`. Python is not needed. Source files are only needed if you want to modify or rebuild the program.

## Important: an external controller

HandController is an **external** controller. It:

- does **not** inject any DLL or code into a game
- does **not** read or modify game memory
- does **not** modify the game process
- does **not** modify game files
- installs no in-game hook and bypasses no protection or anti-cheat

Inputs are sent like a physical keyboard and mouse would, through the standard Windows input mechanism (`SendInput`, via the `pynput` library), to the active window.

## How it works

1. **Webcam**: OpenCV opens one camera (1280×720 by default).
2. **MediaPipe**: the Hand Landmarker model (`hand_landmarker.task`, included) finds 21 points on each hand, up to two hands.
3. **Recognition**: the landmarks are smoothed (One Euro filter), quality-checked, then compared with:
   - your **user gestures** (G01, G02, …), recorded during calibration and recognised by k-NN (SVM / Random Forest optional);
   - the **predefined gestures**, computed from the hand geometry (no recording needed).
4. **Mapping**: each recognised gesture triggers the action you assigned (key, mouse button, combination or macro).

Detected gestures are stabilised over several frames before they act, to avoid flicker.

### Left hand / right hand

HandController separates your **left hand** and your **right hand**. Each hand has its own gesture bank and its own mappings: LEFT:G01 and RIGHT:G01 are two different poses and two different actions. A pose recorded on the left hand is not recognised on the right hand.

### User gestures

Record as many poses as you want per hand (G01, G02, …) with **C** (calibration), give them a name if you like (custom names are never translated), then assign an action with **P**.

### Predefined gestures

| Gesture | Detection |
|---------|-----------|
| **PINCH** | thumb tip touches index tip |
| **TILT UP / DOWN / LEFT / RIGHT** | hand tilted in that direction |
| **FINGER UP** INDEX / MIDDLE / RING / PINKY / THUMB | a single raised finger |
| FIST / OPEN PALM | only if `fist_enabled` / `open_palm_enabled` are enabled in `settings.json` |

They are enabled and assigned in **Gesture settings** (**P**), per hand. The **E** screen lists them apart from user gestures (ON = sends commands). A disabled gesture is never shown or sent.

## Features

- Webcam hand tracking (MediaPipe), left and right hands independent
- User gestures (k-NN) and predefined gestures (PINCH, TILT, FINGER UP)
- Keyboard and mouse: PRESS, HOLD, COMBINATION, MACRO, WAIT
- One Euro smoothing and landmark quality gating
- **External screen overlay** (transparent, click-through OSD above the game)
- **Compact status display** in the camera window
- French / English
- Console with the live log when the EXE starts

## Requirements

- Windows 10 or 11 (64-bit)
- A webcam
- From source: Python **3.13** (version used for the tests; 3.10+ may work)

Dependencies (`requirements.txt`): `opencv-python`, `mediapipe`, `numpy`, `pynput`, `scikit-learn` (optional, only for SVM / Random Forest). The MediaPipe model `hand_landmarker.task` is included in the repository.

## Installation from source

```
git clone <repository-url> HandController
cd HandController
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Launch

```
python main.py
```

or `HandController.exe` from the ZIP. Check without opening the UI (no key is ever sent):

```
python main.py --diagnostic
```

`--diagnostic` checks Python, OpenCV, MediaPipe, NumPy, pynput, the model and the webcam, then exits.

## Webcam

The default camera is `camera_index: 0` in `settings.json`. Requested resolution: `camera_width` / `camera_height` (1280×720 by default). Only one camera and one stream are opened. Good front lighting helps detection a lot; you do not need to bring your hand close to the camera.

## Shortcuts (HandController window focused)

| Key | Action |
|-----|--------|
| G | GAME INPUT ON / OFF (OFF stops macros and releases every key) |
| S | Screen overlay ON / OFF |
| P | Gesture settings (mapping, predefined gestures) |
| E | Saved gestures |
| C | Calibration |
| I | Input panel |
| L | Language |
| F | Direction finger |
| H | Help |
| Q | Quit |

Shortcuts only apply while the HandController window is in front. Gestures keep acting on the active window. F8–F11 are not shortcuts: they can be assigned as game keys.

## Calibration and gesture recording

1. **C** or **Record Gesture**
2. Pick the hand (left / right) and the gesture (G01, G02, …)
3. Replace or append samples
4. Hold the pose during capture; vary position, distance and rotation slightly
5. **ESC** cancels a running capture

Geometric thresholds (pinch, tilt) are calibrated from **Gesture Settings** and written to `settings.json` (never to `gestures.json`).

Replay a landmark capture (never sends keys):

```
python main.py --replay recordings/replay-YYYYMMDD-HHMMSS.jsonl
python main.py --benchmark
```

## Mapping

```
Gesture id → mapping.json → keyboard/mouse
```

**P** opens gesture settings. Types: PRESS, HOLD, COMBINATION, MACRO. Keys: letters, SPACE, ENTER, SHIFT, CTRL, F1…F12, mouse buttons. The **Assign** button captures the next key you press.

## Macros

Gesture Settings → a gesture → type **Macro**.

| Key | Step |
|-----|------|
| A | PRESS |
| W | WAIT |
| H | HOLD |
| E | RELEASE |
| C | COMBINATION |
| B | REPEAT block |
| J / K | Adjust the selected value |
| T | Test (sends to the active window) |
| S | Save |

**G** (INPUT OFF) stops a running macro. Macros never block the webcam.

## Screen overlay (external OSD)

**S** shows or hides a small transparent Win32 window at the top-left of the screen, above the game (MSI Afterburner style):

```
LEFT
Present
PINCH

RIGHT
Present
G01 + TILT LEFT
```

- One line with what is really detected: the recorded gesture (G01…) then the active predefined gestures, joined by " + "
- Hand present with no gesture: "None"; hand absent: only "Absent"
- Transparent background, borderless, **click-through** (clicks go to the game)
- **Never** takes focus (no `SetForegroundWindow`; `SW_SHOWNOACTIVATE`, `WS_EX_NOACTIVATE`)
- No FPS, camera, CPU, RAM or debug
- Hiding the OSD does not stop recognition
- State saved in `settings.json` (`hud_overlay`)

Limitation: a game in **exclusive fullscreen** can hide any external window. Use **borderless windowed** mode to see the OSD.

## Status display (camera window)

A small semi-transparent block at the top-left of the camera window; its size follows the resolution:

```
LEFT HAND
Present
PINCH

RIGHT HAND
Present
G01   Conf: 0.92

INPUT: ON
Screen Overlay: OK
```

Third line: recorded gesture with its confidence (`G01   Conf: 0.92`), predefined gesture alone (`PINCH`), both (`G01 + PINCH   Conf: 0.92`) or `None   Conf: 0.00`. `INPUT` reflects the real input state. "Screen Overlay: OK" only appears when the OSD window is really shown.

## Languages

On first launch HandController asks **Français** / **English**. The choice is saved (`"language": "fr"` or `"en"`). **L** changes it later. Ids (G01), custom names, keys (SPACE, F1) and file names are never translated.

## Configuration

| File | Content |
|------|---------|
| `settings.json` | camera, language, thresholds, smoothing, `hud_overlay` |
| `mapping.json` | actions per hand and gesture |
| `gestures.json` | your recorded poses (k-NN) |

These files are created next to `main.py` (or the EXE) on first launch. They contain your personal data and are never published (excluded from Git and from the release ZIP). `python main.py --reset-config` restores default settings.

One Euro smoothing: `smoothing_min_cutoff`, `smoothing_beta`, `smoothing_d_cutoff`. Lower values = smoother but more latency.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Camera not found | Close other apps using it, check `camera_index` |
| No key reaches the game | Enable **G** (INPUT: ON); run as administrator if the game is elevated |
| A key stays pressed | Click the HandController window and press **G** (INPUT: OFF), or quit with **Q**: every key and mouse button is released |
| OSD not visible in game | Borderless windowed instead of exclusive fullscreen |
| Unstable gestures | Recalibrate with more varied samples, improve lighting |
| Shortcuts do nothing | Click the HandController window (shortcuts need focus) |

Log file: `logs/handcontroller.log` (also shown in the console of the EXE).

## Limitations

- External keyboard/mouse simulation only (`SendInput` via pynput)
- Some games ignore `SendInput`, especially in exclusive fullscreen or with anti-cheat
- The OSD can be hidden by exclusive fullscreen
- A learned pose must be calibrated for each hand
- Windows only
- Use with games at your own risk, and respect each game's rules

## Building the Windows EXE

```
pip install pyinstaller
build_exe.bat
```

The script checks Python and dependencies, builds with `HandController.spec` into `build\stage`, verifies the EXE (resources `hand_landmarker.task`, `localization/`, 16 to 256 px icon, UAC manifest, console EXE) and only replaces the old `dist\` once that check passes. `build/` is then removed. Output:

```
dist\HandController\HandController.exe
```

The EXE opens a console that shows the log (startup, MediaPipe, model, camera, recognised gestures, screen overlay, errors, shutdown). Quit with **Q** in the HandController window.

Release archive: `build_exe.bat --zip` writes `release\HandController-v1.0.0-Windows.zip` (the EXE and its resources, README, licenses; without `settings.json`, `gestures.json`, `mapping.json`, logs or recordings).

The icon is drawn by `tools/make_icon.py` (no third-party artwork).

## Administrator launch

The EXE embeds a `requireAdministrator` manifest (`admin.manifest`). Windows shows the UAC prompt once at launch; there is no relaunch loop. Administrator rights are needed to send keys to a game that itself runs as administrator (Windows UIPI isolation). From source, `python main.py` works without admin rights for non-elevated applications.

## Privacy and security

- The webcam is only used on your computer. No image, video, landmark or gesture is sent to the Internet; HandController has no server and no account.
- Never publish `gestures.json`, `recordings/` or `logs/`: they describe your hands and your setup.
- See [SECURITY.md](SECURITY.md).

## License

HandController is open source under the **MIT License**, see [LICENSE](LICENSE).

Third-party libraries and the MediaPipe model keep their own licenses (Apache-2.0, BSD, LGPL-3.0 for pynput, …), see [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) and the `licenses/` folder.

## Credits

- Hand landmarks: [MediaPipe](https://github.com/google-ai-edge/mediapipe) by Google (Apache-2.0)
- Video: [OpenCV](https://opencv.org) · Input: [pynput](https://github.com/moses-palmer/pynput)
- The repository started from the hand-tracking demo [python-handtrack](https://github.com/GedeAnanda/python-handtrack) by Gede Ananda; HandController has since been rewritten.
