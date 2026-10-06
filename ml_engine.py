"""Train and infer SVM / Random Forest. k-NN stays in GestureEngine.

Models are fitted only when gestures.json changes (dirty flag), never per frame.
sklearn is required for SVM/RF; if import fails those branches stay inactive.

SVM/RF need at least two distinct calibrated gesture IDs on that hand.
"""
import hashlib
import logging

import numpy as np

from gesture_engine import SIDES, UNKNOWN, samples_of
from ml_features import ml_vector_from_sample

log = logging.getLogger("handcontroller")
FEATURE_VERSION = "palm63-v1"
MIN_CLASSES = 2
SKIP_INSUFFICIENT = "insufficient_classes"


def sklearn_available():
    try:
        from sklearn.ensemble import RandomForestClassifier  # noqa: F401
        from sklearn.svm import LinearSVC  # noqa: F401
        return True
    except Exception:
        return False


def get_ml_backend_status(ml_engine=None, requested="knn"):
    """Capability vs what is actually trained. k-NN never needs sklearn."""
    requested = str(requested or "knn").strip().lower()
    sk = sklearn_available()
    trained_svm = False
    trained_rf = False
    reasons = {side: SKIP_INSUFFICIENT for side in SIDES}
    if ml_engine is not None:
        trained_svm = any(ml_engine._svm.get(side) is not None for side in SIDES)
        trained_rf = any(ml_engine._rf.get(side) is not None for side in SIDES)
        reasons = dict(getattr(ml_engine, "skip_reason", {}) or {})
    extra = trained_svm or trained_rf
    ensemble = bool(sk and extra and requested == "ensemble")
    return {
        "knn": True,
        "svm": bool(sk and trained_svm),
        "random_forest": bool(sk and trained_rf),
        "ensemble": ensemble,
        "sklearn_installed": sk,
        "skip_reason": reasons,
    }


def format_ml_startup(status, requested="knn"):
    requested = str(requested or "knn").strip().lower()
    if not status.get("sklearn_installed"):
        return "scikit-learn indisponible — SVM et Random Forest désactivés. k-NN uniquement."
    if requested == "ensemble" and status.get("ensemble"):
        return "Ensemble : KNN + SVM + Random Forest"
    if requested == "ensemble" and not status.get("ensemble"):
        extra = "Au moins 2 gestes calibrés distincts sont nécessaires pour entraîner SVM/RF."
        return "Ensemble demandé, un seul modèle actif : k-NN. " + extra
    if requested == "svm" and status.get("svm"):
        return "ML backend : SVM"
    if requested in ("rf", "random_forest") and status.get("random_forest"):
        return "ML backend : Random Forest"
    if requested == "svm" and not status.get("svm"):
        return "ML backend : KNN (SVM indisponible — classes insuffisantes ou sklearn)."
    if requested in ("rf", "random_forest") and not status.get("random_forest"):
        return "ML backend : KNN (Random Forest indisponible — classes insuffisantes ou sklearn)."
    return "ML backend : KNN"


def _dataset_signature(engine):
    hasher = hashlib.sha1()
    hasher.update(FEATURE_VERSION.encode("ascii"))
    if engine is None:
        return hasher.hexdigest()
    for side in SIDES:
        bank = engine.database.get("gestures", {}).get(side) or {}
        for name in sorted(bank):
            hasher.update(side.encode("ascii"))
            hasher.update(name.encode("ascii"))
            for sample in samples_of(bank.get(name)):
                feat = ml_vector_from_sample(sample)
                if feat is None:
                    continue
                hasher.update(feat.tobytes())
    return hasher.hexdigest()


def _softmax(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - np.max(arr)
    exp = np.exp(np.clip(arr, -30.0, 30.0))
    total = float(np.sum(exp)) or 1.0
    return exp / total


class MLEngine:
    """Per-hand Linear SVM and Random Forest on the same 63-d features as k-NN."""

    def __init__(self, gesture_engine, n_estimators=40, max_depth=12, svm_c=1.0):
        self.engine = gesture_engine
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.svm_c = float(svm_c)
        self.dirty = True
        self.signature = ""
        self.available = sklearn_available()
        self._svm = {side: None for side in SIDES}
        self._rf = {side: None for side in SIDES}
        self._labels = {side: [] for side in SIDES}
        self.skip_reason = {side: SKIP_INSUFFICIENT for side in SIDES}
        if not self.available:
            log.info("sklearn unavailable — SVM/RF disabled")

    def mark_dirty(self):
        self.dirty = True

    def train(self, force=False):
        signature = _dataset_signature(self.engine)
        if not force and not self.dirty and signature == self.signature:
            return False
        self.signature = signature
        self.dirty = False
        if not self.available or self.engine is None:
            return False
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import LinearSVC
        trained = False
        for side in SIDES:
            xs, ys = [], []
            for name, entry in self.engine.database["gestures"][side].items():
                for sample in samples_of(entry):
                    feat = ml_vector_from_sample(sample)
                    if feat is None:
                        continue
                    xs.append(feat)
                    ys.append(name)
            self._svm[side] = None
            self._rf[side] = None
            self._labels[side] = []
            classes = set(ys)
            if len(classes) < MIN_CLASSES:
                self.skip_reason[side] = SKIP_INSUFFICIENT
                continue
            self.skip_reason[side] = ""
            x = np.ascontiguousarray(np.stack(xs, axis=0))
            y = np.array(ys)
            try:
                svm = LinearSVC(C=self.svm_c, max_iter=4000, dual="auto")
            except TypeError:
                svm = LinearSVC(C=self.svm_c, max_iter=4000)
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=0,
                n_jobs=1,
            )
            try:
                svm.fit(x, y)
                rf.fit(x, y)
            except Exception as error:
                log.warning("ML train failed for %s: %s", side, error)
                self.skip_reason[side] = SKIP_INSUFFICIENT
                continue
            self._svm[side] = svm
            self._rf[side] = rf
            self._labels[side] = list(rf.classes_)
            trained = True
        return trained

    def _svm_guess(self, side, features):
        model = self._svm.get(side)
        if model is None:
            return UNKNOWN, 0.0
        query = np.asarray(features, dtype=np.float32).reshape(1, -1)
        try:
            pred = model.predict(query)[0]
            scores = model.decision_function(query)
        except Exception:
            return UNKNOWN, 0.0
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        classes = getattr(model, "classes_", None)
        if classes is None or scores.size == 0:
            return str(pred), 0.55
        if scores.size == 1:
            conf = 1.0 / (1.0 + np.exp(-float(scores[0])))
            if str(classes[1] if len(classes) > 1 else classes[0]) != str(pred):
                conf = 1.0 - conf
            return str(pred), float(max(0.0, min(1.0, conf)))
        probs = _softmax(scores)
        try:
            index = list(classes).index(pred)
            conf = float(probs[index])
        except (ValueError, IndexError):
            conf = float(np.max(probs))
        return str(pred), conf

    def _rf_guess(self, side, features):
        model = self._rf.get(side)
        if model is None:
            return UNKNOWN, 0.0
        query = np.asarray(features, dtype=np.float32).reshape(1, -1)
        try:
            pred = model.predict(query)[0]
            proba = model.predict_proba(query)[0]
        except Exception:
            return UNKNOWN, 0.0
        try:
            index = list(model.classes_).index(pred)
            conf = float(proba[index])
        except (ValueError, IndexError):
            conf = float(np.max(proba)) if proba is not None and len(proba) else 0.0
        return str(pred), conf

    def predict(self, side, features):
        """Return {'svm': (id, conf), 'random_forest': (id, conf)} — missing keys if untrained."""
        if self.dirty:
            self.train()
        result = {}
        if not self.available:
            return result
        svm = self._svm_guess(side, features)
        rf = self._rf_guess(side, features)
        if self._svm.get(side) is not None:
            result["svm"] = svm
        if self._rf.get(side) is not None:
            result["random_forest"] = rf
        return result
