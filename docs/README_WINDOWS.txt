HandController 1.0.0 — Windows
==============================

External webcam-based hand gesture controller for PC games.
Contrôleur externe de gestes de la main par webcam pour les jeux PC.

ENGLISH
-------
1. Run HandController.exe (Windows 10/11, 64-bit). Accept the administrator
   prompt (UAC): it lets HandController send keys to games that also run as
   administrator.
2. A console window shows the log. The camera window "HandController" opens.
3. Choose Français / English on first launch.
4. C = record your gestures, P = assign keys / predefined gestures (PINCH,
   TILT, FINGER UP), G = game input ON/OFF, S = screen overlay, Q = quit.
5. If a key stays pressed: click the HandController window and press G
   (INPUT OFF), or quit with Q. Every key and mouse button is released.

settings.json, mapping.json and gestures.json are created next to the EXE.
They hold your personal settings and poses: do not share them.

HandController is an external controller. It does not inject DLLs or code,
does not read or modify game memory, the game process or game files. It only
sends normal keyboard and mouse events through Windows (SendInput).

FRANÇAIS
--------
1. Lancez HandController.exe (Windows 10/11, 64 bits). Acceptez la demande
   administrateur (UAC) : elle permet d'envoyer des touches aux jeux lancés
   en administrateur.
2. Une console affiche le journal. La fenêtre caméra « HandController » s'ouvre.
3. Choisissez Français / English au premier lancement.
4. C = enregistrer vos gestes, P = attribuer des touches / gestes prédéfinis
   (PINCH, TILT, FINGER UP), G = entrées jeu ON/OFF, S = sur-impression
   d'écran, Q = quitter.
5. Si une touche reste enfoncée : cliquez sur la fenêtre HandController et
   appuyez sur G (INPUT OFF), ou quittez avec Q. Toutes les touches et
   boutons de souris sont relâchés.

settings.json, mapping.json et gestures.json sont créés à côté de l'EXE.
Ils contiennent vos réglages et vos poses : ne les partagez pas.

HandController est un contrôleur externe. Il n'injecte ni DLL ni code, ne lit
ni ne modifie la mémoire, le processus ou les fichiers du jeu. Il envoie
uniquement des événements clavier/souris normaux via Windows (SendInput).

Documentation, source code / code source: https://github.com/Zeus7768/HandController
License / Licence: MIT (LICENSE.txt). Third-party: THIRD_PARTY_NOTICES.txt, licenses/
