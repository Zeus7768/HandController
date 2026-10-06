"""Palm-local pose features for k-NN. Geometry Engine does not use this module.

Stored samples stay 63 values (21 x 3). Extra fingertip distances are computed
from the same local points for benchmarks; the live classifier keeps 63 unless
shape_weight > 0.
"""
import numpy as np

from geometry_engine import INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP, PINKY_TIP
from geometry_engine import RING_TIP, THUMB_TIP, WRIST

FEATURE_SIZE = 63

_INDEX_MCP = 5
_SYNTHETIC_ABSMAX = 3.0
SHAPE_SIZE = 10


def local_palm_points(points):
    """Wrist origin, orthonormal palm frame, unit palm length. Shape (21, 3)."""
    array = np.asarray(points, dtype=np.float32)
    xyz = np.zeros((21, 3), dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 10 or array.shape[1] < 3:
        return xyz
    count = min(21, int(array.shape[0]))
    xyz[:count] = array[:count, :3]
    if float(np.max(np.abs(xyz))) > _SYNTHETIC_ABSMAX:
        # Unit-test / sparse vectors already live in feature space.
        return xyz
    wrist = xyz[WRIST]
    rel = xyz - wrist
    palm = rel[MIDDLE_MCP]
    scale = float(np.linalg.norm(palm))
    if scale < 1e-6:
        scale = max(float(np.linalg.norm(rel)), 0.0001)
        return rel / scale
    y_axis = palm / scale
    across = rel[_INDEX_MCP] - rel[PINKY_MCP]
    x_axis = across - float(np.dot(across, y_axis)) * y_axis
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-5:
        helper = np.array((1.0, 0.0, 0.0), dtype=np.float32)
        if abs(float(np.dot(helper, y_axis))) > 0.9:
            helper = np.array((0.0, 0.0, 1.0), dtype=np.float32)
        x_axis = helper - float(np.dot(helper, y_axis)) * y_axis
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm < 1e-6:
            return rel / scale
    x_axis = x_axis / x_norm
    z_axis = np.cross(x_axis, y_axis)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm < 1e-6:
        return rel / scale
    z_axis = z_axis / z_norm
    x_axis = np.cross(y_axis, z_axis)
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-6:
        return rel / scale
    x_axis = x_axis / x_norm
    axes = np.stack((x_axis, y_axis, z_axis), axis=0)
    local = axes.dot(rel.T).T / scale
    if local[4, 0] < 0.0:
        local[:, 0] *= -1.0
        local[:, 2] *= -1.0
    return np.ascontiguousarray(local, dtype=np.float32)


def knn_features(points):
    """63 palm-local coordinates. Same contract as GestureEngine.normalize_points."""
    return local_palm_points(points).ravel()


def shape_features(local_points):
    """Fingertip distances in the palm frame. Complements the 63 coordinates."""
    pts = np.asarray(local_points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 21:
        return np.zeros(SHAPE_SIZE, dtype=np.float32)
    wrist = pts[WRIST]
    tips = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
    to_wrist = np.array([float(np.linalg.norm(pts[i] - wrist)) for i in tips], dtype=np.float32)
    adjacent = np.array([
        float(np.linalg.norm(pts[THUMB_TIP] - pts[INDEX_TIP])),
        float(np.linalg.norm(pts[INDEX_TIP] - pts[MIDDLE_TIP])),
        float(np.linalg.norm(pts[MIDDLE_TIP] - pts[RING_TIP])),
        float(np.linalg.norm(pts[RING_TIP] - pts[PINKY_TIP])),
        float(np.linalg.norm(pts[THUMB_TIP] - pts[PINKY_TIP])),
    ], dtype=np.float32)
    return np.concatenate((to_wrist, adjacent))


def features_from_sample(sample):
    """Rebuild palm-local 63-vector from a stored sample."""
    try:
        array = np.asarray(sample, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size != FEATURE_SIZE or not np.isfinite(array).all():
        return None
    return knn_features(array.reshape(21, 3))
