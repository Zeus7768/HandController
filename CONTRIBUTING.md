# Contributing

HandController is an external webcam controller. It sends normal keyboard and mouse input. It does not inject code into a game, edit game memory, or bypass anti-cheat.

## Setup

1. Use Python 3.13 if you can. That is the version the tests run on.
2. `pip install -r requirements.txt`
3. Place `hand_landmarker.task` next to `main.py`.
4. `python main.py --diagnostic`
5. `python run_tests.py`

## What to leave alone

- Do not modify `gestures_backup.json`.
- Do not commit personal `gestures.json`, `mapping.json`, `settings.json`, `recordings/`, or `logs/`.
- Do not add a second recognition model, two-hand gestures, or analog mouse in a maintenance change.
- INPUT OFF (**G** or the Input window) is the stop: it ends macros and releases every key. F8–F11 are assignable as game keys.
- After changing dependency versions, run `python tools/collect_licenses.py` and update `THIRD_PARTY_NOTICES.txt`.

## License

Contributions are accepted under the MIT License of the project (see `LICENSE`).
- UI strings go through `t("…")` and `localization/fr.json` / `localization/en.json`. Do not hard-code new user-facing text.

## Tests

`python run_tests.py` must pass before a change is proposed. A skipped test is reported as skipped, not deleted.
