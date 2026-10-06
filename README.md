# HandController

<img src="assets/handcontroller_icon.png" width="128" alt="HandController">

**Version 1.0.0** · MIT License

**HandController** is an external hand-gesture controller for PC games.

It uses a webcam, MediaPipe hand tracking, computer vision and machine-learning-based gesture recognition to turn hand poses into keyboard and mouse inputs.

The project is designed to control games **without modifying, injecting into, or patching the game process**.

```
Webcam → OpenCV → MediaPipe → Recognition Engine → Mapping → Keyboard / Mouse
```

- **[Français](README_FR.md)**: documentation complète
- **[English](README_EN.md)**: full documentation

---

## 📥 Download

Windows users can download the latest release from the [GitHub Releases page](../../releases/latest): **`HandController-v1.0.0-Windows.zip`**. Unzip it and run `HandController.exe` (no Python needed).

---

## 🎮 Tested Games

HandController has been tested on Windows with:

| Game | Tested |
|---|---|
| Elden Ring | ✅ |
| Blasphemous 2 | ✅ |
| Lies of P | ✅ |

These tests cover:

- hand tracking
- gesture recognition
- keyboard and mouse input
- independent left/right hand recognition
- predefined gestures
- custom gestures
- external OSD
- general stability

> These are games tested by the author, not a guarantee of universal compatibility.

---

## ✋ Features

### Hand tracking

- Webcam-based hand tracking (OpenCV, 1280×720 by default)
- MediaPipe Hand Landmarker (`hand_landmarker.task`, included): 21 points per hand
- Support for up to two hands
- Independent left/right hand processing (each hand has its own gestures, mappings, filter and stabilizer)
- Hand-loss handling (short hold of the last valid landmarks, no extrapolation)
- Landmark quality checking (ACCEPT / HOLD / REJECT)
- Landmark jump rejection
- One Euro smoothing
- Hand-pose and scale normalization (palm-local frame, unit palm length)
- Gesture confidence estimation
- Temporal gesture stabilization

### Gesture recognition

HandController combines several recognition techniques:

- **k-NN** (default classifier for custom gestures)
- **SVM** and **Random Forest** (optional, scikit-learn)
- **Ensemble** of k-NN + SVM + Random Forest (optional: hard, soft or weighted voting)
- **Geometry-based recognition** for the predefined gestures
- Confidence scoring (model scores, inter-model agreement, landmark quality)
- Temporal stabilization

The classifier is chosen with `ml_backend` in `settings.json`: `knn` (default), `svm`, `rf` or `ensemble`. SVM and Random Forest are trained only when your gesture bank changes, never per frame.

The recognition system is designed to remain robust when the hand moves, rotates or changes distance from the camera: features are computed in a palm-local frame, so position, rotation and distance matter much less than the pose itself.

---

## 🖐️ Predefined Gestures

HandController includes predefined geometric gestures that can be used directly in the mapping system, per hand:

| Gesture | Detection |
|---|---|
| **PINCH** | thumb tip touches index tip |
| **TILT UP / DOWN / LEFT / RIGHT** | hand tilted in that direction |
| **FINGER UP** (INDEX / MIDDLE / RING / PINKY / THUMB) | a single raised finger |
| FIST / OPEN PALM | only if `fist_enabled` / `open_palm_enabled` are enabled in `settings.json` |

They are enabled and assigned in **Gesture settings** (**P**). These predefined gestures are handled separately from the user's machine-learning gesture database (`gestures.json`), and their thresholds are calibrated in `settings.json`.

---

## 🤖 Custom Gestures

Users can record their own poses per hand (G01, G02, …) with **C** (calibration), optionally give them a name, then assign an action with **P**.

Custom gestures are stored locally in `gestures.json` and can be mapped to keyboard or mouse actions. LEFT:G01 and RIGHT:G01 are two different poses with two different actions.

Example:

```text
G01 → Z
G02 → A
G03 → Left Mouse Button
G04 → Shift
```

The exact gestures and mappings depend on the user's configuration.

---

## ⌨️ Keyboard & Mouse Input

HandController simulates normal computer input (`SendInput` via pynput): keyboard keys (letters, SPACE, ENTER, SHIFT, CTRL, arrows, F1…F12, …) and mouse buttons.

Mapping types:

| Type | Behaviour |
|---|---|
| **PRESS** | one tap when the gesture appears (with a cooldown) |
| **HOLD** | key held while the gesture is held |
| **COMBINATION** | several keys together (each held or tapped) |
| **MACRO** | a sequence of steps |

Macro steps: **PRESS**, **HOLD**, **RELEASE**, **WAIT** (up to 5 s), **COMBINATION** and **REPEAT** blocks.

The input system releases every active key and mouse button when input is turned off, when camera frames are lost, and on exit. A hand's held keys are released when that hand disappears for longer than a short grace period. Macros never block the webcam.

---

## 🖥️ External OSD

**S** shows or hides a small external on-screen display at the top-left of the screen, above the game:

```text
LEFT
Absent

RIGHT
Present
G01 + TILT LEFT
```

- One line with what is really detected: the recorded gesture, then the active predefined gestures, joined by " + "
- Hand present with no gesture: "None"; hand absent: only "Absent"
- Transparent, borderless, **click-through** window that **never** takes focus
- No FPS, camera, CPU or debug information
- Hiding the OSD does not stop recognition

The OSD is a separate Win32 window, independent from the game's own interface. It does not modify the game window or inject code into the game. A game in **exclusive fullscreen** can hide it: use **borderless windowed** mode.

The camera window also shows a compact status block (left hand, right hand, gesture and confidence, INPUT state, screen overlay).

---

## 🌍 Languages

HandController supports:

- 🇫🇷 Français
- 🇬🇧 English

The language is chosen on first launch and can be changed with **L**. It applies to the application's user-facing interface and messages. Internal identifiers such as gesture IDs (G01), custom names, key names and file names remain unchanged.

---

## 🎥 Requirements

### Hardware

- Windows PC
- Webcam
- Keyboard and/or mouse
- Recommended: stable webcam positioning and good front lighting

### Software

- Windows 10 or Windows 11 (64-bit)
- From source: Python **3.13** (version used for the tests; 3.10+ may work)

Dependencies (`requirements.txt`):

- `opencv-python`
- `mediapipe`
- `numpy`
- `pynput`
- `scikit-learn` (only needed for SVM / Random Forest / ensemble)

---

## 📦 Installation

### Windows executable

Download **`HandController-v1.0.0-Windows.zip`** from the [GitHub Releases page](../../releases/latest).

Extract the ZIP archive and launch:

```text
HandController-v1.0.0-Windows\HandController.exe
```

Keep the `_internal` folder next to `HandController.exe`: it contains the MediaPipe model, the translations and the icons. Windows asks for administrator rights (UAC) so HandController can send keys to games that also run as administrator. A console window shows the live log.

### Running from source

Clone the repository:

```
git clone https://github.com/Zeus7768/HandController.git
cd HandController
```

Install the dependencies (a virtual environment is recommended):

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then start HandController:

```
python main.py
```

Check the installation without opening the UI (no key is ever sent):

```
python main.py --diagnostic
```

### Shortcuts (HandController window focused)

| Key | Action |
|---|---|
| G | Game input ON / OFF (OFF stops macros and releases every key) |
| S | Screen overlay (OSD) ON / OFF |
| P | Gesture settings (mapping, predefined gestures) |
| E | Saved gestures |
| C | Calibration (record gestures) |
| I | Input panel |
| L | Language |
| F | Direction finger |
| H | Help |
| Q | Quit |

Shortcuts only apply while the HandController window is in front, so keys typed in a game never trigger them.

---

## ⚙️ Configuration

HandController stores its configuration in local files, created next to `main.py` (or the EXE) on first launch:

| File | Content |
|---|---|
| `settings.json` | camera, language, thresholds, smoothing, classifier, predefined gestures, OSD state |
| `mapping.json` | actions per hand and gesture |
| `gestures.json` | your recorded poses |

User gesture data is stored separately from the predefined geometric gesture system. These files contain personal data: they are excluded from Git and from the release ZIP. `python main.py --reset-config` restores default settings.

If you keep a local `gestures_backup.json`, HandController never reads or writes it. Do not modify it unless you intentionally want to replace your backup dataset.

---

## 🧠 Recognition Pipeline

```text
Webcam (OpenCV)
   ↓
MediaPipe Hand Landmarker
   ↓
Landmark Quality Check (ACCEPT / HOLD / REJECT, jump rejection)
   ↓
One Euro Filtering
   ↓
Hand Pose Normalization (palm-local frame)
   ↓
Feature Extraction (63 values)
   ↓
k-NN (default) / SVM / Random Forest / Ensemble     +     Geometry Engine (predefined gestures)
   ↓
Confidence Evaluation
   ↓
Gesture Stabilization
   ↓
Gesture Result
   ↓
Mapping
   ↓
Keyboard / Mouse Input
```

The left and right hands are processed independently. The geometry engine runs alongside the classifier; it never replaces the recognition of custom gestures.

---

## 🛡️ Safety

HandController is an external input controller.

It does not:

- inject code or DLLs into games;
- modify game processes;
- patch game executables;
- read or modify game memory;
- replace or modify game files;
- install in-game hooks or bypass protections.

It works by detecting hand movements externally and sending normal keyboard/mouse input to the operating system, like a physical keyboard and mouse would.

Use the software responsibly and respect the rules of the games and services you use it with.

---

## 🚨 Emergency Stop

HandController includes an emergency stop that immediately stops active inputs and macros:

- press **G** (INPUT OFF) in the HandController window: running macros stop and every key and mouse button is released;
- or open the **Input** panel (**I**) → Disable;
- or quit with **Q** (or the window's close button): inputs are released and the camera is closed.

Shortcuts need the HandController window in front: click it first. Always test the emergency stop before using HandController with a game.

---

## 📁 Project Structure

```text
HandController/
├── main.py                  # application, camera loop, UI
├── recognition_engine.py    # quality → classifier → confidence → stabilizer
├── gesture_engine.py        # gesture bank and k-NN
├── ml_engine.py             # SVM / Random Forest
├── ensemble_classifier.py   # k-NN + SVM + RF voting
├── geometry_engine.py       # predefined gestures (PINCH, TILT, FINGER UP)
├── landmark_filter.py       # One Euro filter
├── landmark_quality.py      # landmark quality and jump rejection
├── gesture_stabilizer.py    # temporal stabilization
├── gesture_mapping.py       # mapping.json, actions, macros
├── input_controller.py      # keyboard / mouse output
├── game_osd.py              # external OSD
├── localization/            # fr.json, en.json
├── assets/                  # icon (original artwork)
├── tools/                   # icon, licenses, build verification
├── licenses/                # third-party license texts
├── docs/
├── hand_landmarker.task     # MediaPipe model
├── run_tests.py             # automated tests
├── HandController.spec      # PyInstaller build
├── build_exe.bat
├── requirements.txt
└── README.md
```

`settings.json`, `mapping.json`, `gestures.json`, `recordings/` and `logs/` are created locally and are not part of the repository. The exact structure may evolve between releases.

---

## 🧪 Testing

The project contains automated tests covering important components of the application:

```
python run_tests.py
```

Before a release, the project should be tested for:

- gesture recognition
- left/right hand separation
- landmark filtering
- gesture stabilization
- keyboard input
- mouse input
- combinations
- macros
- OSD
- language switching
- configuration loading
- emergency stop
- application shutdown

Landmark recordings can be replayed without sending any key:

```
python main.py --replay recordings/replay-YYYYMMDD-HHMMSS.jsonl
python main.py --benchmark
```

### Real-world testing

HandController has been tested with:

- Elden Ring
- Blasphemous 2
- Lies of P

Compatibility may vary depending on the game, Windows configuration, webcam, lighting conditions and user-created gesture mappings.

---

## 🎮 Performance & Recognition

Recognition performance can depend on:

- webcam resolution
- lighting
- camera position
- distance between the hand and camera
- hand orientation
- background complexity
- CPU performance
- the selected classifier (`knn` is the lightest; `ensemble` runs three models)

For best results:

- Place the webcam in a stable position.
- Keep your hand clearly visible.
- Avoid very dark lighting.
- Avoid excessive motion blur.
- Keep enough distance from the camera for the entire hand to remain visible.
- Calibrate gestures in conditions similar to those used during gameplay.

---

## 🔧 Troubleshooting

### The hand is not detected

Check:

- webcam permissions;
- lighting;
- camera selection (`camera_index` in `settings.json`);
- whether the entire hand is visible;
- whether another application is using the webcam.

### Recognition is unstable

Try:

- improving lighting;
- moving slightly farther from the camera;
- keeping the hand inside the camera frame;
- recalibrating the gesture with more varied samples;
- reducing unnecessary background movement.

### Inputs are not reaching the game

Check:

- the selected mapping;
- whether input is enabled (**G**, INPUT: ON);
- whether the game is running with different Windows privileges (the EXE runs as administrator; from source, start the terminal as administrator for elevated games);
- whether the selected keyboard/mouse action works normally outside the game.

Some games ignore simulated input, especially in exclusive fullscreen or with anti-cheat.

### A key stays pressed

Click the HandController window and press **G** (INPUT: OFF), or quit with **Q**.

### The OSD does not appear

Check:

- whether the external OSD is enabled (**S**);
- whether the game is in exclusive fullscreen (use borderless windowed);
- whether another overlay is interfering.

Log file: `logs/handcontroller.log` (also shown in the console of the EXE).

---

## 🏗️ Windows EXE

```
pip install pyinstaller
build_exe.bat
```

Produces `dist\HandController\HandController.exe` (UAC administrator manifest, embedded icon, console window with the live log). `build_exe.bat --zip` also writes `release\HandController-v1.0.0-Windows.zip`. See the [full documentation](README_EN.md) for details.

---

## 📜 License

HandController is released under the [MIT License](LICENSE).

Third-party components keep their own licenses: see [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) and the `licenses/` folder.

---

## 🙏 Credits

HandController uses open-source libraries and technologies including:

- [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache-2.0)
- [OpenCV](https://opencv.org) (Apache-2.0)
- [NumPy](https://numpy.org) (BSD)
- [scikit-learn](https://scikit-learn.org) (BSD)
- [pynput](https://github.com/moses-palmer/pynput) (LGPL-3.0)
- certifi and other packages bundled in the Windows release (see [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt))

The MediaPipe Hand Landmarker model (`hand_landmarker.task`) is included unmodified in this repository and remains under its own terms (Apache-2.0).

The repository started from the hand-tracking demo [python-handtrack](https://github.com/GedeAnanda/python-handtrack) by Gede Ananda; HandController has since been rewritten.

Game names and trademarks belong to their respective owners.

---

## 🤝 Contributing

Contributions are welcome. You can contribute by:

- reporting bugs;
- suggesting improvements;
- improving recognition;
- adding tests;
- improving documentation;
- improving localization;
- proposing new features.

Before submitting a major change, please open an issue to discuss it. See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

---

## ⭐ Support the Project

If you find HandController useful:

- ⭐ Star the repository
- 🐛 Report bugs
- 💡 Suggest improvements
- 🔧 Contribute code
- 📖 Improve the documentation

---

## 📦 Release

Current release: **HandController v1.0.0**

The first public version focuses on the core hand-tracking and gesture-control system. Future versions may add additional recognition and control features.

---

## 📄 Open Source

HandController is open source and distributed under the MIT License. You are free to study, modify and redistribute the project according to the terms of the license.
