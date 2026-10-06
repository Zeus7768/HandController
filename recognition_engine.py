"""Unified recognition: quality, filter, ML ensemble, geometry stay separate.

MediaPipe stays in the camera loop. This module consumes one hand's landmarks.
"""
from dataclasses import dataclass, field

from confidence_engine import combine as combine_confidence
from ensemble_classifier import EnsembleClassifier, WEIGHTED
from gesture_engine import UNKNOWN, normalize_side
from gesture_stabilizer import GestureStabilizer
from landmark_quality import ACCEPT, HOLD, REJECT
from ml_engine import MLEngine
from ml_features import HandPoseNormalizer, ml_vector


@dataclass
class GestureResult:
    gesture_id: str = UNKNOWN
    gesture_name: str = ""
    hand: str = ""
    confidence: float = 0.0
    source: str = "none"
    alternatives: list = field(default_factory=list)
    geometry_gestures: dict = field(default_factory=dict)
    ml_gesture: str = UNKNOWN
    is_stable: bool = False
    quality: str = REJECT
    knn_gesture: str = UNKNOWN
    svm_gesture: str = UNKNOWN
    rf_gesture: str = UNKNOWN
    agreement: float = 0.0
    active_models: list = field(default_factory=list)
    ambiguous: bool = False
    quality_score: float = 0.0
    detection_quality: float = 0.0
    finger_quality: float = 0.0
    handedness_quality: float = 0.0
    temporal_quality: float = 0.0
    interpolated: bool = False


class RecognitionEngine:
    """One object per app. LEFT and RIGHT never share filter / ML / stabilizer state."""

    def __init__(self, gesture_engine, geometry_engine=None, settings=None):
        settings = settings or {}
        self.knn = gesture_engine
        self.geometry = geometry_engine
        self.normalizer = HandPoseNormalizer()
        self.ml = MLEngine(gesture_engine)
        self.ensemble = EnsembleClassifier(
            mode=settings.get("ensemble_mode", WEIGHTED),
            knn_weight=float(settings.get("knn_weight", 1.0)),
            svm_weight=float(settings.get("svm_weight", 1.0)),
            random_forest_weight=float(settings.get("random_forest_weight", 1.0)),
            min_confidence=float(settings.get("confidence_threshold", 0.55)),
            class_margin=float(settings.get("knn_class_margin", settings.get("class_margin", 0.12))),
        )
        self.stabilizer = GestureStabilizer(gesture_engine)
        self.backend = str(settings.get("ml_backend", "knn")).strip().lower()
        self.ml.train()

    def refresh_models(self):
        self.ml.mark_dirty()
        self.ml.train(force=True)

    def features(self, points):
        return self.normalizer.normalize(points)

    def recognize_ml(self, features, side, quality=ACCEPT, quality_report=None):
        """k-NN + optional SVM/RF. Geometry is not consulted here."""
        side = normalize_side(side)
        empty = GestureResult(hand=side or "", quality=quality)
        self._attach_quality(empty, quality_report)
        if side is None or features is None:
            return empty
        knn_id, knn_conf = self.knn.recognize(features, side)
        votes = {"knn": (knn_id, knn_conf)}
        extra = self.ml.predict(side, features)
        votes.update(extra)
        info = {
            "agreement": 1.0, "votes": {}, "top2": UNKNOWN,
            "active_models": ["knn"], "ambiguous": False,
        }
        backend = self.backend
        extra_models = [name for name in ("svm", "random_forest") if name in extra]
        if backend == "knn" or (backend == "ensemble" and not extra_models):
            chosen, conf, source = knn_id, knn_conf, "knn"
            info["active_models"] = ["knn"]
        elif backend == "svm" and "svm" in extra:
            chosen, conf, source = extra["svm"][0], extra["svm"][1], "svm"
            info["active_models"] = ["svm"]
        elif backend in ("rf", "random_forest") and "random_forest" in extra:
            chosen, conf, source = extra["random_forest"][0], extra["random_forest"][1], "random_forest"
            info["active_models"] = ["random_forest"]
        elif backend == "ensemble" and extra_models:
            chosen, conf, info = self.ensemble.decide(votes)
            source = "ensemble"
        else:
            chosen, conf, source = knn_id, knn_conf, "knn"
            info["active_models"] = ["knn"]
        if quality != ACCEPT:
            chosen, conf = UNKNOWN, 0.0
        elif quality_report is not None and float(getattr(quality_report, "quality_score", 0.0) or 0.0) < 0.38:
            chosen, conf = UNKNOWN, 0.0
        elif bool(info.get("ambiguous")):
            chosen, conf = UNKNOWN, 0.0
        confidence = combine_confidence(votes, chosen, quality=quality)
        if confidence <= 0.0:
            confidence = float(conf)
        result = GestureResult(
            gesture_id=chosen,
            gesture_name=self.knn.display_name(side, chosen) if chosen != UNKNOWN else "",
            hand=side,
            confidence=float(confidence),
            source=source,
            alternatives=[info.get("top2", UNKNOWN)] if info.get("top2") else [],
            ml_gesture=chosen,
            quality=quality,
            knn_gesture=knn_id,
            svm_gesture=extra.get("svm", (UNKNOWN, 0.0))[0],
            rf_gesture=extra.get("random_forest", (UNKNOWN, 0.0))[0],
            agreement=float(info.get("agreement", 0.0) or 0.0),
            active_models=list(info.get("active_models") or ["knn"]),
            ambiguous=bool(info.get("ambiguous", False)),
        )
        self._attach_quality(result, quality_report)
        return result

    @staticmethod
    def _attach_quality(result, quality_report):
        if quality_report is None:
            return
        result.quality_score = float(getattr(quality_report, "quality_score", 0.0) or 0.0)
        result.detection_quality = float(getattr(quality_report, "detection_quality", 0.0) or 0.0)
        result.finger_quality = float(getattr(quality_report, "finger_quality", 0.0) or 0.0)
        result.handedness_quality = float(getattr(quality_report, "handedness_quality", 0.0) or 0.0)
        result.temporal_quality = float(getattr(quality_report, "temporal_quality", 0.0) or 0.0)
        result.interpolated = bool(getattr(quality_report, "interpolated", False))

    def process_hand(self, landmarks, handedness, timestamp=None, score=1.0, quality=ACCEPT,
                     geometry_states=None, quality_report=None):
        """Landmarks already filtered. timestamp kept for API stability."""
        side = normalize_side(handedness)
        if side is None:
            empty = GestureResult(quality=quality)
            self._attach_quality(empty, quality_report)
            return empty
        feat = ml_vector(landmarks) if landmarks is not None else None
        result = self.recognize_ml(feat, side, quality=quality, quality_report=quality_report)
        result.geometry_gestures = dict(geometry_states or {})
        if quality in (HOLD, REJECT):
            result.gesture_id = UNKNOWN
            result.ml_gesture = UNKNOWN
            result.confidence = 0.0
            result.is_stable = False
            return result
        stable = self.stabilizer.update(side, result.gesture_id, result.confidence)
        result.gesture_id = stable
        result.is_stable = stable != UNKNOWN and stable == result.ml_gesture
        if stable != result.ml_gesture:
            result.confidence = 0.0
        return result
