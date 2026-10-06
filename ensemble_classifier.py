"""Combine k-NN, SVM and Random Forest votes. Never trains."""
from dataclasses import dataclass

from gesture_engine import UNKNOWN

HARD = "hard"
SOFT = "soft"
WEIGHTED = "weighted"
MODES = (HARD, SOFT, WEIGHTED)


@dataclass
class EnsemblePrediction:
    gesture: str = UNKNOWN
    confidence: float = 0.0
    active_models: list = None
    agreement: float = 0.0
    ambiguous: bool = True
    votes: dict = None
    top1: str = UNKNOWN
    top2: str = UNKNOWN
    mode: str = WEIGHTED

    def __post_init__(self):
        if self.active_models is None:
            self.active_models = []
        if self.votes is None:
            self.votes = {}


def _normalize_mode(value):
    name = str(value or WEIGHTED).strip().lower()
    if name in ("hard_voting", "majority"):
        return HARD
    if name in ("soft_voting", "soft"):
        return SOFT
    if name in ("weighted_voting", "weighted"):
        return WEIGHTED
    return WEIGHTED if name not in MODES else name


class EnsembleClassifier:
    def __init__(self, mode=WEIGHTED, knn_weight=1.0, svm_weight=1.0, random_forest_weight=1.0,
                 min_confidence=0.55, class_margin=0.12):
        self.mode = _normalize_mode(mode)
        self.weights = {
            "knn": max(0.0, float(knn_weight)),
            "svm": max(0.0, float(svm_weight)),
            "random_forest": max(0.0, float(random_forest_weight)),
        }
        self.min_confidence = float(min_confidence)
        self.class_margin = float(class_margin)

    def decide(self, votes):
        """votes: {model_name: (gesture_id, confidence)}. Returns (gesture, confidence, extras)."""
        usable = {}
        for name, payload in (votes or {}).items():
            if not payload:
                continue
            gesture, conf = payload[0], float(payload[1] if len(payload) > 1 else 0.0)
            if not gesture or gesture == UNKNOWN:
                continue
            usable[name] = (gesture, max(0.0, min(1.0, conf)))
        extras = {
            "votes": {name: {"gesture": g, "confidence": c} for name, (g, c) in usable.items()},
            "agreement": 0.0,
            "top1": UNKNOWN,
            "top2": UNKNOWN,
            "mode": self.mode,
            "active_models": list(usable.keys()),
            "ambiguous": True,
        }
        if not usable:
            extras["active_models"] = []
            return UNKNOWN, 0.0, extras

        scores = {}
        if self.mode == HARD:
            counts = {}
            for gesture, _conf in usable.values():
                counts[gesture] = counts.get(gesture, 0.0) + 1.0
            total = float(sum(counts.values())) or 1.0
            scores = {name: value / total for name, value in counts.items()}
        else:
            for model, (gesture, conf) in usable.items():
                weight = self.weights.get(model, 1.0)
                if self.mode == SOFT:
                    weight = 1.0
                scores[gesture] = scores.get(gesture, 0.0) + weight * conf
            total = float(sum(scores.values())) or 1.0
            scores = {name: value / total for name, value in scores.items()}

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best, best_score = ranked[0]
        extras["top1"] = best
        extras["top2"] = ranked[1][0] if len(ranked) > 1 else UNKNOWN
        extras["agreement"] = sum(1 for g, _c in usable.values() if g == best) / float(len(usable))
        extras["active_models"] = list(usable.keys())
        if best_score < self.min_confidence:
            extras["ambiguous"] = True
            return UNKNOWN, best_score, extras
        if len(ranked) > 1:
            gap = (best_score - ranked[1][1]) / max(best_score, 1e-6)
            if gap < self.class_margin:
                extras["ambiguous"] = True
                return UNKNOWN, best_score, extras
        if extras["agreement"] < 0.34 and len(usable) >= 2:
            extras["ambiguous"] = True
            return UNKNOWN, best_score, extras
        extras["ambiguous"] = False
        return best, float(best_score), extras
