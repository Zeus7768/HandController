"""One Euro filter for MediaPipe landmarks. NumPy only.

One state per hand, 21 points x 3 axes. The first sample is returned as-is
so a hand that just appeared is not pulled toward an old position.
"""
import numpy as np

from geometry_engine import LANDMARK_COUNT


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Speed-adaptive low-pass on a vector. Independent of any other instance."""

    def __init__(self, min_cutoff=1.2, beta=0.45, d_cutoff=1.15):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        self._t = None
        self._x = None
        self._dx = None

    def filter(self, values, timestamp):
        sample = np.asarray(values, dtype=np.float32)
        shape = sample.shape
        flat = np.ascontiguousarray(sample).reshape(-1)
        if self._t is None or self._x is None or self._x.shape != flat.shape:
            self._t = float(timestamp)
            self._x = flat.copy()
            self._dx = np.zeros_like(flat)
            return sample
        dt = max(float(timestamp) - self._t, 1e-4)
        if dt > 0.25:
            self._t = float(timestamp)
            self._x = flat.copy()
            self._dx = np.zeros_like(flat)
            return sample
        derivative = (flat - self._x) / dt
        alpha_d = _alpha(self.d_cutoff, dt)
        self._dx = alpha_d * derivative + (1.0 - alpha_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        alpha = _alpha(cutoff, dt)
        self._x = alpha * flat + (1.0 - alpha) * self._x
        self._t = float(timestamp)
        return self._x.reshape(shape)


class Point:
    """Landmark with the .x .y .z attributes normalize_landmarks expects."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


def points_to_landmarks(points):
    array = np.asarray(points, dtype=np.float32).reshape(LANDMARK_COUNT, 3)
    return [Point(row[0], row[1], row[2]) for row in array]
