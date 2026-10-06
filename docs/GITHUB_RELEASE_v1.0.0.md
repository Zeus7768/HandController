# HandController v1.0.0

First public open-source release of HandController.

Features:
- webcam hand tracking
- left/right hand recognition
- custom gestures
- predefined gestures
- PINCH
- TILT
- FINGER UP
- keyboard/mouse control
- external game OSD
- French / English
- calibration
- macro support

Windows package included.

## Download

**`HandController-v1.0.0-Windows.zip`** (Windows 10/11, 64-bit). Unzip, then run `HandController\HandController.exe`. Python is not required.

- The EXE asks for administrator rights (UAC) so it can send keys to games that also run as administrator.
- A console window opens with the live log (camera, MediaPipe, recognised gestures, screen overlay, shutdown).
- `settings.json`, `mapping.json` and `gestures.json` are created next to the EXE on first launch. No personal data is included in the ZIP.

## Safety

HandController is an external controller. It does not inject DLLs or code, does not read or modify game memory, the game process or game files. It only sends normal keyboard and mouse events through Windows (`SendInput`).

## Known limitations

- Some games ignore `SendInput`, especially in exclusive fullscreen or with anti-cheat. Borderless windowed is the most reliable mode.
- A learned pose must be calibrated for each hand.
- Exclusive fullscreen games may hide the screen overlay.

## License

MIT. Third-party components keep their own licenses (see `THIRD_PARTY_NOTICES.txt` and `licenses/` in the ZIP).
