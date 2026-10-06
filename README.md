# HandController

**Version 1.0.0** · MIT License

<img src="assets/handcontroller_icon.png" width="96" alt="HandController">

HandController is an **external webcam-based hand gesture controller** for Windows PCs and games.

It recognises your hand poses with a webcam (MediaPipe), then simulates keyboard and mouse input. Your left and right hands are tracked separately, each with its own gestures: poses you record yourself (G01, G02, …) and predefined gestures (**PINCH**, **TILT**, **FINGER UP**).

```
Webcam → OpenCV → MediaPipe → Recognition Engine → mapping → keyboard/mouse
```

## Download

Windows users can download the latest release from the [GitHub Releases page](../../releases/latest): **`HandController-v1.0.0-Windows.zip`**. Unzip it and run `HandController.exe` (no Python needed).

## Documentation

- **[Français](README_FR.md)** — documentation complète
- **[English](README_EN.md)** — full documentation

## Safety: an external controller

HandController is **not** a game mod. It does not inject DLLs or code, does not read or modify game memory, does not modify the game process or game files, installs no in-game hook and bypasses no protection. It only sends normal keyboard and mouse events through Windows (`SendInput` via pynput), like a physical keyboard and mouse would.

## Quick start from source

```
pip install -r requirements.txt
python main.py
```

On first launch, choose **Français** / **English**. The camera opens in one OpenCV window named **HandController**, with a compact status block (left / right hand, gesture, confidence, INPUT, screen overlay).

- **C** records your gestures, **P** assigns keys, mouse buttons, combinations or macros
- **G** turns game input ON / OFF (OFF stops macros and releases every key)
- **S** shows or hides the external screen overlay (transparent, click-through, never takes focus)

Configuration lives in `settings.json`, `mapping.json` and `gestures.json`, created on first launch and never published.

## Windows EXE

```
build_exe.bat
```

Produces `dist\HandController\HandController.exe` (UAC administrator manifest, embedded icon, console window with the live log). `build_exe.bat --zip` also writes the release ZIP. See the full documentation for details.

## License

HandController is released under the [MIT License](LICENSE). Third-party components keep their own licenses: see [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) and `licenses/`.
