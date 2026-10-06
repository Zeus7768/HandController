# HandController V1 — last hand-tracking audit

This note records the last technical robustness pass before HandController V1
is frozen. Architecture is unchanged: MediaPipe Hands, OpenCV, NumPy, k-NN,
Geometry Engine, One Euro, GestureStabilizer. No PyTorch, TensorFlow, PySide6,
profiles, FLICK, or SWIPE.

`gestures_backup.json` is never opened by this work.

## 1. Audit of the five reference projects

Principles only. No code, models, or extra datasets were copied.

### 1. xinghaochen/awesome-hand-pose-estimation

| | |
| --- | --- |
| Technique | Survey of pose-estimation papers, datasets, and benchmark protocols. |
| Problem | Incomparable numbers (different splits, different metrics). |
| Interest | Keep our benchmark reproducible on **our** data. |
| Cost | None at runtime. |
| Complexity | Documentation / `classifier_benchmark.py` only. |
| Regression risk | Low if we do not add external datasets. |
| Integration | Yes, as methodology. |

**Kept:** leave-one-out on `gestures.json`, same 63-d palm-local features as live k-NN.

**Rejected:** the catalog of deep models, extra public datasets.

### 2. vladmandic/human (MIT)

| | |
| --- | --- |
| Technique | Separate detection / landmark / handedness scores; short temporal reuse when a body part drops for a few frames. |
| Problem | Opaque “one score”; flicker when a frame is briefly invalid. |
| Interest | Quality components + a tiny HOLD interpolator. |
| Cost | A few floats and a 21×3 copy per hand per HOLD frame. |
| Complexity | Low. |
| Regression risk | Medium if interpolation were allowed to fire mappings. |
| Integration | Yes, inside `LandmarkQualityChecker` / `LandmarkHold`. |

**Kept:** `detection_quality`, `finger_quality`, `handedness_quality`, `temporal_quality`, plus `quality_score` as a min-synthesis. Short reuse of the last ACCEPT landmarks.

**Rejected:** the Human library, body-wide interpolation, long-horizon prediction.

### 3. handtracking-io/yoha

| | |
| --- | --- |
| Technique | Split hand presence, orientation, landmarks, then pose. |
| Problem | Mixing tracking with recognition and with input mapping. |
| Interest | Confirm the existing pipeline. |
| Cost | None. |
| Complexity | None. |
| Regression risk | None if we only verify. |
| Integration | Already present. |

**Kept:** the separation (see section 7).

**Rejected:** TensorFlow.js, Yoha’s runtime, its gesture catalog.

### 4. CalciferZh/minimal-hand

| | |
| --- | --- |
| Technique | DetNet + IKNet + MANO; robustness to occlusion, scale, viewpoint, fast motion. |
| Problem | 2D landmarks break under those conditions. |
| Interest | Extra **tests**, not a 3D model. |
| Cost | Test-only. |
| Complexity | None in V1. |
| Regression risk | High if PyTorch/MANO were added. |
| Integration | Tests only. |

**Kept:** close / far / tilted / partial occlusion / fast-motion cases.

**Rejected:** PyTorch, MANO, DetNet, IKNet.

### 5. jaredrhod/barehands

| | |
| --- | --- |
| Technique | Distances relative to a rigid palm bone; “canyon” comparison of correct vs confused poses; sanity of landmark geometry. |
| Problem | Absolute pixel thresholds; guess-based tuning. |
| Interest | Relative metrics + diagnostic comparison. |
| Cost | Sanity check is a handful of norms per accepted hand. Diagnostics are offline. |
| Complexity | Low. |
| Regression risk | Medium if production PINCH/TILT thresholds were auto-written. |
| Integration | Yes, in Geometry Engine + `geometry_diagnostics.py`. |

**Kept:** wrist → middle MCP as `hand_scale`; curl / arch / spread / thumb span; conservative sanity; discriminator report.

**Rejected:** web UI, Three.js, server, OBS, JavaScript architecture, FLICK, SWIPE, automatic threshold writes.

## 2. Techniques retained vs rejected

| Retained | Where | Why |
| --- | --- | --- |
| Relative palm length | `geometry_engine.hand_scale` | Distance/scale invariance for geometry. |
| Finger curl / arch / spread / thumb span | `geometry_engine` | Shared helpers for the five Finger Up slots. |
| Geometry sanity | `geometry_sanity` → `LandmarkQualityChecker` | Low `finger_quality` / HOLD on exploded landmarks. |
| Quality components | `QualityReport`, `GestureResult` | Human-style scores without replacing MediaPipe values. |
| Short HOLD interpolation | `LandmarkHold` | Visual continuity only, ≤ grace frames, no extrapolation. |
| Discriminator tool | `geometry_diagnostics.py` | Canyon method; never writes `settings.json`. |
| LOO benchmark | `classifier_benchmark.py` | Reproducible on `gestures.json`. |
| Extra robustness tests | `run_tests.py` | Occlusion, distance, tilt, fast motion, LEFT/RIGHT isolation. |

| Rejected | Why |
| --- | --- |
| New tracker / MANO / DetNet / IKNet / TF.js | Out of V1 architecture. |
| Replace MediaPipe, OpenCV, NumPy, k-NN | Explicit freeze. |
| Replace One Euro or GestureStabilizer | Interpolation is additive and weaker than a filter. |
| New ML feature vector | Live path stays 63-d palm-local. Shape-10 remains benchmark-only. |
| Auto-tune PINCH/TILT/FINGER_UP | No production write from diagnostics. |
| FLICK / SWIPE / profiles / PySide6 / old hotkey Game OSD | Already removed; not reintroduced (the current screen overlay is a separate, display-only window toggled with S). |

PINCH/TILT/FINGER_UP **thresholds were not changed**. Existing tests already separate those poses. A “more logical” number without a measured gap would be a regression risk.

## 3. Pipeline (tracking vs gesture vs mapping)

```
MediaPipe Hands
  → landmarks + handedness score
  → LandmarkQualityChecker (ACCEPT / HOLD / REJECT + components)
  → One Euro (per hand)
  → LandmarkHold (display only, HOLD)
  → feature extraction (palm-local 63)     [ML]
  → k-NN / optional SVM / RF / ensemble    [ML]
  → Geometry Engine (image landmarks)      [geometry]
  → confidence
  → GestureStabilizer
  → GestureResult
  → mapping (PRESS / HOLD / COMBINATION / MACRO / WAIT)
  → InputController
```

- Tracking does not know keys.
- Recognition does not know keys.
- Mapping does not compute landmarks.
- LEFT and RIGHT never share filter, stabilizer, quality, interpolator, or ML state.
- An interpolated skeleton is **not** a new observation for mapping.

ML normalisation (palm orthonormal frame) and geometry normalisation (`hand_scale` on image landmarks) remain two systems.

## 4. Sanity bounds

Checked every quality frame. Conservative, from synthetic `PALM_POSE` (palm aspect ≈ 0.12) and typical MediaPipe image hands (aspect ≈ 0.5–1.5, finger extension ≈ 0.3–1.3):

- palm aspect outside `0.08 … 4.0`
- any finger extension > `4.0 × hand_scale`
- non-finite landmarks

A real foreshortened hand stays inside those limits. An exploded landmark set fails sanity: `finger_quality` drops and a previously good track HOLDs.

## 5. Temporal interpolation

`LandmarkHold` copies the last ACCEPT pose during HOLD, per hand, with a real timestamp argument (unused for prediction). It never extrapolates velocity. After the hold budget (aligned with `hand_loss_grace_frames`, default 2) it clears. Recognition and geometry stay frozen on HOLD/REJECT (`_quality_frozen`).

## 6. Diagnostic tool

```text
from geometry_diagnostics import metrics_from_landmarks, compare_metric_sets, format_discriminators
report = compare_metric_sets(correct_rows, impostor_rows)
print(format_discriminators(report))
```

Output lists non-overlapping metrics (min–max canyon) only. It does not patch `settings.json`.

## 7. Benchmark

Run: `python main.py --benchmark` (leave-one-out on `gestures.json` only).

No new live feature representation was added, so there is no production “old vs new vector” swap. `compare_shape_features` remains an optional k-NN-only study of the extra 10 distances. On this machine both were 100.0% LOO, so the live vector stays 63-d.

Measured on the same `gestures.json` after this audit (sklearn present):

| Hand | Model | Samples | Accuracy | Train | Predict |
| --- | --- | ---: | ---: | ---: | ---: |
| LEFT | k-NN | 210 | 100.0% | 12.6 ms | 0.090 ms |
| LEFT | SVM | 210 | 100.0% | 2237 ms | 0.718 ms |
| LEFT | Random Forest | 210 | 100.0% | 36784 ms | 7.434 ms |
| LEFT | Ensemble | 210 | 100.0% | 39021 ms | 0.052 ms |
| RIGHT | k-NN | 120 | 100.0% | 4.7 ms | 0.063 ms |
| RIGHT | SVM | 120 | 100.0% | 663 ms | 0.618 ms |
| RIGHT | Random Forest | 120 | 100.0% | 17545 ms | 7.109 ms |
| RIGHT | Ensemble | 120 | 100.0% | 18209 ms | 0.049 ms |

Ensemble disagreements: 0. Ambiguous: 0. Agreement on sampled votes: 100%.

100% LOO on this file is consistent with near-duplicate calibration samples. It is not open-world generalisation. k-NN remains the default live backend (lowest latency, no extra train cost).

KNN coords63: 100.0%. KNN coords+shape: 100.0%. Extra distances were **not** promoted to the live path.

## 8. Performance

New per-frame work (synthetic 21×3, this PC):

- `measure_hand` (already live): ~87 µs
- `geometry_sanity`: ~tens of µs after sharing one `hand_scale`
- `quality_report` (includes sanity): ~0.25 ms
- `LandmarkHold.observe`: ~2 µs
- `relative_metrics`: ~0.41 ms — **not** on the live path (diagnostics only)

Two hands at 30 FPS: quality+sanity stays well under 1 ms. MediaPipe remains the bottleneck. Interpolation and sanity were kept.

PINCH / TILT / Finger Up thresholds were not changed, so geometry detector cost is unchanged.

## 9. Limits

- MediaPipe still fails on heavy occlusion and extreme blur; we only HOLD briefly.
- Interpolation does not invent a hand after grace expires.
- Quality components are heuristics on MediaPipe landmarks, not a second detector.
- LOO accuracy on `gestures.json` can look perfect when samples are near-duplicates; that is a dataset property, not a claim of open-world generalisation.
- Geometry Finger Up still needs an image-space UP direction; curl/arch help extension, not a new gesture family.

## 10. Licences (attribution only)

Referenced as inspiration, not as vendored code:

- Human — MIT (vladmandic/human)
- awesome-hand-pose-estimation — documentation / survey listing (xinghaochen)
- Yoha — inspect upstream repository if redistributing *their* code (we did not)
- minimal-hand — research code; not included
- barehands — inspect upstream if redistributing *their* code (we did not)

HandController V1 ships none of those trees.
