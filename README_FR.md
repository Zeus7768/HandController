# HandController

**Version 1.0.0** — [English](README_EN.md) · [Présentation](README.md)

<img src="assets/handcontroller_icon.png" width="96" alt="HandController">

HandController est un **contrôleur externe de gestes de la main par webcam** pour PC Windows et jeux vidéo.

Il reconnaît vos poses de main avec une webcam, puis simule le clavier et la souris. Objectif : jouer ou piloter n'importe quelle application avec les mains, sans matériel spécial et sans toucher au jeu.

```
Webcam → OpenCV → MediaPipe → Recognition Engine (k-NN / SVM / Random Forest + Geometry) → mapping → clavier/souris
```

## Téléchargement

Sous Windows, téléchargez la dernière version sur la [page Releases de GitHub](../../releases/latest) : prenez **`HandController-v1.0.0-Windows.zip`**, décompressez-le et lancez `HandController.exe`. Python n'est pas nécessaire. Les fichiers sources ne servent que pour modifier ou recompiler le programme.

## Important : un contrôleur externe

HandController est un contrôleur **externe**. Il :

- n'injecte **aucune** DLL ni aucun code dans un jeu
- ne lit ni ne modifie la **mémoire** du jeu
- ne modifie pas le **processus** du jeu
- ne modifie pas les **fichiers** du jeu
- n'installe aucun hook dans le jeu et ne contourne aucune protection ni anti-triche

Les entrées sont envoyées comme le ferait un vrai clavier ou une vraie souris, par le mécanisme d'entrée standard de Windows (`SendInput`, via la bibliothèque `pynput`), vers la fenêtre active.

## Fonctionnement

1. **Webcam** : OpenCV ouvre une caméra (1280×720 par défaut).
2. **MediaPipe** : le modèle Hand Landmarker (`hand_landmarker.task`, fourni) trouve 21 points sur chaque main, jusqu'à deux mains.
3. **Reconnaissance** : les points sont lissés (filtre One Euro), leur qualité est contrôlée, puis ils sont comparés :
   - à vos **gestes utilisateur** (G01, G02, …), enregistrés pendant la calibration et reconnus par k-NN (SVM / Random Forest optionnels) ;
   - aux **gestes prédéfinis**, calculés à partir de la géométrie de la main (aucun enregistrement nécessaire).
4. **Mapping** : chaque geste reconnu déclenche l'action que vous lui avez attribuée (touche, bouton de souris, combinaison ou macro).

Un geste détecté est stabilisé sur plusieurs images avant d'agir, pour éviter le scintillement.

### Main gauche / Main droite

HandController distingue votre **main gauche** et votre **main droite**. Chaque main a sa propre banque de gestes et ses propres mappings : GAUCHE:G01 et DROITE:G01 sont deux poses et deux actions différentes. Une pose enregistrée sur la main gauche n'est pas reconnue sur la main droite.

### Gestes utilisateur

Enregistrez autant de poses que vous voulez par main (G01, G02, …) avec **C** (calibration), donnez-leur un nom si vous le souhaitez (les noms personnalisés ne sont jamais traduits), puis attribuez une action avec **P**.

### Gestes prédéfinis

| Geste | Détection |
|-------|-----------|
| **PINCH** | le bout du pouce touche le bout de l'index |
| **TILT UP / DOWN / LEFT / RIGHT** | main inclinée dans cette direction |
| **FINGER UP** INDEX / MIDDLE / RING / PINKY / THUMB | un seul doigt levé |
| FIST / OPEN PALM | seulement si `fist_enabled` / `open_palm_enabled` sont activés dans `settings.json` |

Ils s'activent et s'attribuent dans les **Paramètres de gestes** (**P**), main par main. L'écran **E** les liste à part des gestes utilisateur (ON = envoie les commandes). Un geste désactivé n'est jamais affiché ni envoyé.

## Fonctions

- Suivi des mains par webcam (MediaPipe), main gauche et main droite indépendantes
- Gestes utilisateur (k-NN) et gestes prédéfinis (PINCH, TILT, FINGER UP)
- Clavier et souris : PRESS, HOLD, COMBINATION, MACRO, WAIT
- Lissage One Euro et contrôle de qualité des landmarks
- **Sur-impression d'écran externe** (OSD transparent, click-through, au-dessus du jeu)
- **Affichage d'état compact** dans la fenêtre caméra
- Français / anglais
- Console avec le journal en direct au lancement de l'EXE

## Prérequis

- Windows 10 ou 11 (64 bits)
- Une webcam
- Depuis les sources : Python **3.13** (version utilisée pour les tests ; 3.10+ peut fonctionner)

Dépendances (`requirements.txt`) : `opencv-python`, `mediapipe`, `numpy`, `pynput`, `scikit-learn` (optionnel, uniquement pour SVM / Random Forest). Le modèle MediaPipe `hand_landmarker.task` est fourni dans le dépôt.

## Installation depuis les sources

```
git clone <url-du-depot> HandController
cd HandController
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

```
python main.py
```

ou `HandController.exe` depuis le ZIP. Vérification sans ouvrir l'interface (aucune touche envoyée) :

```
python main.py --diagnostic
```

`--diagnostic` vérifie Python, OpenCV, MediaPipe, NumPy, pynput, le modèle et la webcam, puis quitte.

## Webcam

La caméra par défaut est `camera_index: 0` dans `settings.json`. Résolution demandée : `camera_width` / `camera_height` (1280×720 par défaut). Une seule caméra et un seul flux sont ouverts. Un bon éclairage de face améliore nettement la détection ; il n'est pas nécessaire de rapprocher la main de la caméra.

## Raccourcis (fenêtre HandController au premier plan)

| Touche | Action |
|--------|--------|
| G | GAME INPUT ON / OFF (OFF arrête les macros et relâche toutes les touches) |
| S | Sur-impression d'écran ON / OFF |
| P | Paramètres de gestes (mapping, gestes prédéfinis) |
| E | Gestes enregistrés |
| C | Calibration |
| I | Panneau Input |
| L | Langue |
| F | Doigt de direction |
| H | Aide |
| Q | Quitter |

Les raccourcis ne s'appliquent que lorsque la fenêtre HandController est devant. Les gestes, eux, continuent d'agir sur la fenêtre active. F8 à F11 ne sont pas des raccourcis : elles restent assignables comme touches de jeu.

## Calibration et enregistrement des gestes

1. **C** ou **Enregistrer un geste**
2. Choisir la main (gauche / droite) et le geste (G01, G02, …)
3. Remplacer ou ajouter des échantillons
4. Tenir la pose pendant la capture ; varier légèrement position, distance et rotation
5. **ESC** annule une capture en cours

Les seuils géométriques (pince, inclinaisons) se calibrent depuis **Paramètres de geste** et sont écrits dans `settings.json` (jamais dans `gestures.json`).

Replay d'une capture landmarks (n'envoie jamais de touches) :

```
python main.py --replay recordings/replay-AAAAMMJJ-HHMMSS.jsonl
python main.py --benchmark
```

## Mapping

```
ID de geste → mapping.json → clavier/souris
```

**P** ouvre les paramètres de gestes. Types : PRESS, HOLD, COMBINATION, MACRO. Touches : lettres, SPACE, ENTER, SHIFT, CTRL, F1…F12, boutons de souris. Le bouton **Assigner** capture la prochaine touche pressée.

## Macros

Paramètres de geste → un geste → type **Macro**.

| Touche | Étape |
|--------|-------|
| A | PRESS |
| W | WAIT |
| H | HOLD |
| E | RELEASE |
| C | COMBINATION |
| B | Bloc REPEAT |
| J / K | Ajuster la valeur sélectionnée |
| T | Tester (envoie vers la fenêtre active) |
| S | Sauver |

**G** (INPUT OFF) arrête une macro en cours. Les macros ne bloquent jamais la webcam.

## Sur-impression d'écran (OSD externe)

**S** affiche ou masque une petite fenêtre Win32 transparente en haut à gauche de l'écran, au-dessus du jeu (style MSI Afterburner) :

```
GAUCHE
Présente
PINCH

DROITE
Présente
G01 + TILT LEFT
```

- Une ligne par geste réellement détecté : geste enregistré (G01…) puis gestes prédéfinis actifs, séparés par « + »
- Main présente sans geste : « Aucun » ; main absente : seulement « Absente »
- Fond transparent, sans bordure, **click-through** (les clics passent au jeu)
- Ne prend **jamais** le focus (pas de `SetForegroundWindow` ; `SW_SHOWNOACTIVATE`, `WS_EX_NOACTIVATE`)
- Pas de FPS, caméra, CPU, RAM ni debug
- Masquer l'OSD ne coupe pas la reconnaissance
- État mémorisé dans `settings.json` (`hud_overlay`)

Limite : un jeu en **plein écran exclusif** peut masquer toute fenêtre externe. Utilisez le mode **fenêtré sans bordure** pour voir l'OSD.

## Affichage d'état (fenêtre caméra)

Un petit bloc semi-transparent en haut à gauche de la fenêtre caméra, dont la taille suit la résolution :

```
MAIN GAUCHE
Présente
PINCH

MAIN DROITE
Présente
G01   Conf: 0.92

INPUT: ON
Sur-impression d'écran: Ok
```

Troisième ligne : geste enregistré avec sa confiance (`G01   Conf: 0.92`), geste prédéfini seul (`PINCH`), les deux (`G01 + PINCH   Conf: 0.92`) ou `Aucun   Conf: 0.00`. `INPUT` reflète l'état réel des entrées. « Sur-impression d'écran: Ok » n'apparaît que lorsque la fenêtre OSD est réellement affichée. La fenêtre caméra affiche les textes sans accents (limite de la police OpenCV).

## Langues

Au premier lancement, HandController demande **Français** / **English**. Le choix est enregistré (`"language": "fr"` ou `"en"`). **L** change la langue ensuite. Les identifiants (G01), les noms personnalisés, les touches (SPACE, F1) et les noms de fichiers ne sont pas traduits.

## Configuration

| Fichier | Contenu |
|---------|---------|
| `settings.json` | caméra, langue, seuils, lissage, `hud_overlay` |
| `mapping.json` | actions par main et par geste |
| `gestures.json` | vos poses enregistrées (k-NN) |

Ces fichiers sont créés à côté de `main.py` (ou de l'EXE) au premier lancement. Ils contiennent vos données personnelles et ne sont jamais publiés (exclus de Git et du ZIP de release). `python main.py --reset-config` remet les réglages par défaut.

Lissage One Euro : `smoothing_min_cutoff`, `smoothing_beta`, `smoothing_d_cutoff`. Valeurs plus basses = plus lisse mais plus de latence.

## Dépannage

| Problème | Solution |
|----------|----------|
| Caméra introuvable | Fermer les autres logiciels qui l'utilisent, vérifier `camera_index` |
| Aucune touche envoyée au jeu | Activer **G** (INPUT: ON) ; lancer en administrateur si le jeu est élevé |
| Une touche reste enfoncée | Cliquer sur la fenêtre HandController et appuyer sur **G** (INPUT: OFF), ou quitter avec **Q** : toutes les touches et boutons de souris sont relâchés |
| OSD invisible en jeu | Mode fenêtré sans bordure au lieu du plein écran exclusif |
| Gestes instables | Recalibrer avec plus d'échantillons variés, améliorer l'éclairage |
| Raccourcis sans effet | Cliquer sur la fenêtre HandController (les raccourcis exigent le focus) |

Journal : `logs/handcontroller.log` (également affiché dans la console de l'EXE).

## Limites

- Simulation clavier/souris externe uniquement (`SendInput` via pynput)
- Certains jeux ignorent `SendInput`, surtout en plein écran exclusif ou avec anti-triche
- L'OSD peut être masqué par un plein écran exclusif
- Une pose apprise doit être calibrée pour chaque main
- Windows uniquement
- L'usage avec un jeu est sous votre responsabilité ; respectez les règles de chaque jeu

## Compilation de l'EXE Windows

```
pip install pyinstaller
build_exe.bat
```

Le script vérifie Python et les dépendances, compile avec `HandController.spec` dans `build\stage`, vérifie l'EXE (ressources `hand_landmarker.task`, `localization/`, icône 16 à 256 px, manifeste UAC, EXE console) et ne remplace l'ancien `dist\` qu'après cette vérification. `build/` est ensuite supprimé. Résultat :

```
dist\HandController\HandController.exe
```

L'EXE ouvre une console : elle affiche le journal (démarrage, MediaPipe, modèle, caméra, gestes reconnus, sur-impression, erreurs, arrêt). Quitter avec **Q** dans la fenêtre HandController.

Archive de release : `build_exe.bat --zip` produit `release\HandController-v1.0.0-Windows.zip` (l'EXE et ses ressources, README, licences ; sans `settings.json`, `gestures.json`, `mapping.json`, logs ni enregistrements).

L'icône est dessinée par `tools/make_icon.py` (aucune image tierce).

## Lancement administrateur

L'EXE contient un manifeste `requireAdministrator` (`admin.manifest`). Windows affiche la demande UAC une seule fois au lancement ; il n'y a aucune boucle de relance. Les droits administrateur sont nécessaires pour envoyer des touches à un jeu lui-même lancé en administrateur (isolation UIPI de Windows). En source, `python main.py` fonctionne sans droits administrateur pour les applications non élevées.

## Confidentialité et sécurité

- La webcam n'est utilisée que sur votre ordinateur. Aucune image, vidéo, landmark ou geste n'est envoyé sur Internet ; HandController n'a ni serveur ni compte.
- Ne publiez jamais `gestures.json`, `recordings/` ni `logs/` : ils décrivent vos mains et votre configuration.
- Voir [SECURITY.md](SECURITY.md).

## Licence

HandController est open source sous **licence MIT**, voir [LICENSE](LICENSE).

Les bibliothèques tierces et le modèle MediaPipe gardent leurs propres licences (Apache-2.0, BSD, LGPL-3.0 pour pynput, …), voir [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) et le dossier `licenses/`.

## Crédits

- Points de la main : [MediaPipe](https://github.com/google-ai-edge/mediapipe) de Google (Apache-2.0)
- Vidéo : [OpenCV](https://opencv.org) · Entrées : [pynput](https://github.com/moses-palmer/pynput)
- Le dépôt est parti de la démo de suivi des mains [python-handtrack](https://github.com/GedeAnanda/python-handtrack) de Gede Ananda ; HandController a depuis été entièrement réécrit.
