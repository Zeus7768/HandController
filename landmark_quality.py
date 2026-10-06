"""Per-hand landmark validation. Independent LEFT / RIGHT state.

Gates keep the original MediaPipe values (handedness score, scale, jump).
quality_score is only a synthesis of those components, never a replacement.
"""
from dataclasses import dataclass

import numpy as np

from geometry_engine import LANDMARK_COUNT, MIDDLE_MCP, WRIST, geometry_sanity
from gesture_engine import SIDES, normalize_side

ACCEPT = "accept"
HOLD = "hold"
REJECT = "reject"

# Handedness below this is not a clean identity. MediaPipe typically reports >0.7.
_MIN_SCORE = 0.45
# Wrist → middle MCP in image units after the preview flip (coords ~0–1).
# Image units wrist→MCP. Farther hands stay valid; this is not a "come closer" gate.
_MIN_SCALE = 0.018
_MAX_SCALE = 0.85
# Allow a little overshoot; many fingertips sit near the frame edge.
_BOUNDS = (-0.18, 1.18)
_MAX_OUTSIDE = 4
# Wrist jump in image units in one frame: tracking glitch or swap.
_MAX_JUMP = 0.38
# Keep the previous decision only briefly. A real move then re-locks.
_MAX_HOLD = 3


def _clamp01(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


@dataclass
class QualityReport:
    """Per-frame quality. Components stay separate; quality_score is a summary."""

    action: str = REJECT
    quality_score: float = 0.0
    detection_quality: float = 0.0
    finger_quality: float = 0.0
    handedness_quality: float = 0.0
    temporal_quality: float = 0.0
    interpolated: bool = False
    sane: bool = False


class LandmarkHold:
    """Reuse the last ACCEPT landmarks during HOLD. No extrapolation.

    Interpolated points are for visual/temporal continuity only. They must
    not be treated as a new strong observation for mapping.
    """

    def __init__(self, max_frames=_MAX_HOLD):
        self.max_frames = max(1, int(max_frames))
        self.reset()

    def reset(self, side=None):
        if side is None:
            self._points = {name: None for name in SIDES}
            self._age = {name: 0 for name in SIDES}
            return
        name = normalize_side(side)
        if name is None:
            return
        self._points[name] = None
        self._age[name] = 0

    def observe(self, side, points, action, timestamp=None):
        """timestamp is recorded for API stability; HOLD never extrapolates in time."""
        name = normalize_side(side)
        if name is None:
            return None, False
        if action == ACCEPT and points is not None:
            array = np.asarray(points, dtype=np.float32)
            self._points[name] = array.copy()
            self._age[name] = 0
            return array, False
        if action == HOLD and self._points[name] is not None and self._age[name] < self.max_frames:
            self._age[name] += 1
            return self._points[name], True
        self._points[name] = None
        self._age[name] = 0
        return None, False


class LandmarkQualityChecker:
    """One previous wrist per hand. A long gap is handled by reset() from the app."""

    def __init__(self):
        self.reset()

    def reset(self, side=None):
        if side is None:
            self._wrist = {name: None for name in SIDES}
            self._xy = {name: None for name in SIDES}
            self._good = {name: False for name in SIDES}
            self._holds = {name: 0 for name in SIDES}
            self.last_report = {name: QualityReport() for name in SIDES}
            return
        name = normalize_side(side)
        if name is None:
            return
        self._wrist[name] = None
        self._xy[name] = None
        self._good[name] = False
        self._holds[name] = 0
        self.last_report[name] = QualityReport()

    def _fail(self, name, recoverable=False):
        if not self._good[name]:
            return REJECT
        self._holds[name] = self._holds.get(name, 0) + 1
        if self._holds[name] >= _MAX_HOLD:
            self._wrist[name] = None
            self._xy[name] = None
            self._good[name] = False
            self._holds[name] = 0
            return REJECT
        return HOLD

    def evaluate(self, side, points, score=1.0):
        return self.evaluate_report(side, points, score).action

    def evaluate_report(self, side, points, score=1.0):
        name = normalize_side(side)
        empty = QualityReport()
        if name is None:
            return empty
        array = np.asarray(points, dtype=np.float32) if points is not None else None
        if array is None or array.ndim != 2 or array.shape[0] < LANDMARK_COUNT or array.shape[1] < 2:
            action = self._fail(name, recoverable=False)
            report = QualityReport(action=action)
            self.last_report[name] = report
            return report
        xyz = array[:LANDMARK_COUNT, :3]
        if not np.isfinite(xyz).all():
            action = self._fail(name, recoverable=False)
            report = QualityReport(action=action)
            self.last_report[name] = report
            return report
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        wrist = xyz[WRIST, :2]
        scale = float(np.linalg.norm(xyz[MIDDLE_MCP, :2] - wrist))
        xy = xyz[:, :2]
        outside = int(np.count_nonzero((xy < _BOUNDS[0]) | (xy > _BOUNDS[1])))
        jump = 0.0
        prev = self._wrist[name]
        prev_xy = self._xy.get(name)
        if prev is not None:
            jump = float(np.linalg.norm(wrist - prev))
        landmark_jump = False
        if prev_xy is not None and prev_xy.shape == xy.shape:
            delta = np.linalg.norm(xy - prev_xy, axis=1)
            median = float(np.median(delta))
            # Individual points teleported while the hand as a whole barely moved.
            if median < 0.10 and int(np.count_nonzero(delta > median + 0.20)) >= 4:
                landmark_jump = True
        sane, _reasons = geometry_sanity(xyz)

        bad_size = scale < _MIN_SCALE or scale > _MAX_SCALE
        bad_score = score < _MIN_SCORE
        bad_bounds = outside > _MAX_OUTSIDE
        bad_jump = (prev is not None and jump > _MAX_JUMP) or landmark_jump
        # Extreme geometry: HOLD a previously good track; reject a first frame that cannot be a hand.
        # Mild foreshortening stays inside geometry_sanity bounds and is accepted.
        bad_geometry = not sane

        if bad_bounds or not np.isfinite(scale):
            action = HOLD if self._good[name] else REJECT
            recoverable = False
        elif bad_size or bad_score:
            action = HOLD if self._good[name] else REJECT
            recoverable = False
        elif bad_jump or bad_geometry:
            action = HOLD if self._good[name] else REJECT
            recoverable = True
        else:
            action = ACCEPT
            recoverable = True

        handedness_quality = _clamp01((score - _MIN_SCORE) / max(1.0 - _MIN_SCORE, 1e-6)) if score >= _MIN_SCORE else 0.0
        detection_quality = handedness_quality
        if bad_size:
            detection_quality *= 0.25
        finger_quality = 1.0 if sane and outside <= _MAX_OUTSIDE else (0.45 if sane else 0.15)
        if prev is None:
            temporal_quality = 1.0 if action == ACCEPT else 0.0
        elif jump <= _MAX_JUMP * 0.25:
            temporal_quality = 1.0
        elif jump <= _MAX_JUMP:
            temporal_quality = 0.7
        else:
            temporal_quality = 0.2
        quality_score = min(detection_quality, finger_quality, temporal_quality) if action == ACCEPT else 0.0
        if action != ACCEPT:
            quality_score = 0.35 * min(detection_quality, finger_quality, temporal_quality)

        if action == ACCEPT:
            self._wrist[name] = wrist.copy()
            self._xy[name] = xy.copy()
            self._good[name] = True
            self._holds[name] = 0
            report = QualityReport(
                action=ACCEPT,
                quality_score=quality_score,
                detection_quality=detection_quality,
                finger_quality=finger_quality,
                handedness_quality=handedness_quality,
                temporal_quality=temporal_quality,
                sane=sane,
            )
            self.last_report[name] = report
            return report

        self._holds[name] = self._holds.get(name, 0) + 1
        if self._holds[name] >= _MAX_HOLD:
            if recoverable and not (bad_size or bad_score or bad_bounds):
                self._wrist[name] = wrist.copy()
                self._xy[name] = xy.copy()
                self._good[name] = True
                self._holds[name] = 0
                action = ACCEPT
                quality_score = min(detection_quality, finger_quality, 0.55)
            else:
                self._wrist[name] = None
                self._xy[name] = None
                self._good[name] = False
                self._holds[name] = 0
                action = REJECT
                quality_score = 0.0
        report = QualityReport(
            action=action,
            quality_score=quality_score,
            detection_quality=detection_quality,
            finger_quality=finger_quality,
            handedness_quality=handedness_quality,
            temporal_quality=temporal_quality,
            sane=sane,
        )
        self.last_report[name] = report
        return report
