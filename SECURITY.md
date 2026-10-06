# Security

HandController only simulates keyboard and mouse input on the machine where it runs. It does not connect to a server and it does not ask for an account.

HandController is an external controller: it does not inject DLLs or code, does not read or modify game memory, does not modify the game process or game files, and installs no in-game hook.

## Do not publish

- `recordings/` (landmark traces of your hands)
- `gestures.json` (your calibrated poses)
- `settings.json`, `mapping.json` (your personal configuration)
- `logs/handcontroller.log`
- `.env` files, tokens, or passwords

Those paths are listed in `.gitignore` and are never included in the release ZIP.

## If input sticks

Click the HandController window and press **G** (INPUT: OFF), or open **Input** (**I**), choose **Disable** and confirm. Both stop macros and release every key and mouse button this program is holding. Game input stays off until you enable it again. Quitting HandController (**Q** or the window close button) also releases everything.

## Reporting a problem

Describe what you did and what the log says. Do not attach a recording unless you have removed anything you consider private.
