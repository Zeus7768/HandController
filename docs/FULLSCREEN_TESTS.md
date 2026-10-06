# Manual fullscreen / display-mode tests

HandController sends **system-level** keyboard and mouse events (`SendInput`).
It does not inject into the game and does not force the game to the foreground.

Do these tests in Notepad first, then in a game. Inputs must stay **OFF** until
the target window is focused, then enable **Input**.

## TEST A — Windowed

1. Set the game to windowed mode
2. Click the game window
3. Enable Input
4. Trigger a HOLD gesture (for example W) and a PRESS gesture
5. Expected: the game receives the keys

## TEST B — Borderless windowed

1. Set the game to borderless / fullscreen windowed
2. Click the game
3. Enable Input
4. Expected: same as windowed in most engines. This is usually the most reliable mode.

## TEST C — Exclusive fullscreen

1. Set the game to exclusive fullscreen
2. Enable HandController input, then switch to the game
3. Trigger HOLD / PRESS / mouse
4. Expected: **only if** the game accepts synthetic `SendInput` events
5. If nothing happens: the game (or anti-cheat) is filtering simulated input. There is no supported bypass.

## TEST D — Preview visible

1. Leave the OpenCV camera window on screen (do not cover the whole game if you can avoid it)
2. Click the **game**, not the camera window
3. Enable input and gesture
4. Expected: the game is still controllable. The camera window must not need to be in front.

## TEST E — Preview behind the game

1. Put the game in front so it fully covers the OpenCV camera window
2. Trigger HOLD, PRESS, COMBINATION, MACRO
3. Expected: inputs continue; recognition does not need the camera window in front
4. Shortcuts (G, S, P, C, Q…) only work while the HandController window is focused

## TEST F — Two hands

1. LEFT: COMBINATION W+D (HOLD)
2. RIGHT: PRESS LEFT_MOUSE
3. Show both poses together
4. Hide the left hand
5. Expected: W and D release, mouse / right-hand mapping keeps working

## TEST G — Screen overlay (external OSD)

1. Focus the HandController window and press **S**
2. A transparent block appears at the top-left of the screen: LEFT / RIGHT, Present / Absent, detected gesture
3. Click the game through the overlay
4. Expected: clicks reach the game, the overlay never takes focus, the status block shows "Screen Overlay: OK"
5. Press **S** again: the overlay disappears, recognition continues

## TEST H — Exclusive fullscreen vs overlays

1. Set the game to exclusive fullscreen
2. Expected: the camera window and the screen overlay **may be hidden**. This is a Windows compositor limitation.
3. Do **not** inject into the game. Use borderless/windowed instead.
4. Recognition and `SendInput` continue even if no HandController window is visible.

## Also check

- **G** (INPUT OFF): macros stop, keys and mouse buttons release
- Closing HandController does not leave W / SHIFT / mouse down
- A running MACRO does not freeze the webcam
