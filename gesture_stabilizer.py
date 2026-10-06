"""Temporal decision filter. Independent of One Euro (landmark) filtering."""
from gesture_engine import GestureEngine


class GestureStabilizer:
    """LEFT/RIGHT independent. Delegates to GestureEngine so existing tests hold."""

    def __init__(self, engine):
        self.engine = engine

    def update(self, side, gesture, confidence=1.0):
        return self.engine.stabilize(side, gesture, confidence)

    def reset(self, side=None):
        if side is None:
            self.engine.reset_all()
            return
        self.engine.reset_hand(side)
