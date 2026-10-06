"""Internal recognition confidence. Not a calibrated probability."""
from landmark_quality import ACCEPT

UNKNOWN = "INCONNU"


def agreement_ratio(votes, winner):
    """Fraction of non-unknown model votes that match winner."""
    if not winner or winner == UNKNOWN:
        return 0.0
    names = [name for name, _conf in votes.values() if name and name != UNKNOWN]
    if not names:
        return 0.0
    return sum(1 for name in names if name == winner) / float(len(names))


def combine(model_votes, winner, quality=ACCEPT, temporal=1.0):
    """Blend model scores, inter-model agreement and landmark quality.

    model_votes: dict name -> (gesture, score in 0..1)
    """
    if quality != ACCEPT or not winner or winner == UNKNOWN:
        return 0.0
    scores = [float(conf) for name, conf in model_votes.values() if name == winner]
    if not scores:
        scores = [float(conf) for name, conf in model_votes.values() if name and name != UNKNOWN]
    if not scores:
        return 0.0
    mean = sum(scores) / len(scores)
    agree = agreement_ratio(model_votes, winner)
    try:
        temporal = float(temporal)
    except (TypeError, ValueError):
        temporal = 1.0
    value = mean * (0.55 + 0.45 * agree) * max(0.0, min(1.0, temporal))
    return float(max(0.0, min(1.0, value)))
