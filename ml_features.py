"""Shared ML features. Same vector for k-NN, SVM and Random Forest.

Live path: 63 palm-local coordinates (identical to GestureEngine / k-NN).
Shape extras stay available for benchmarks only.
"""
import numpy as np

from feature_extractor import FEATURE_SIZE, features_from_sample, knn_features
from feature_extractor import local_palm_points, shape_features

FEATURE_VERSION_BASE = "palm63-v1"
FEATURE_VERSION_EXTENDED = "palm63+shape10-v1"

__all__ = (
    "FEATURE_SIZE",
    "FEATURE_VERSION_BASE",
    "FEATURE_VERSION_EXTENDED",
    "HandPoseNormalizer",
    "features_from_sample",
    "knn_features",
    "local_palm_points",
    "ml_vector",
    "ml_vector_extended",
    "shape_features",
)


class HandPoseNormalizer:
    """Wrist origin, palm orthonormal frame, unit palm length. All recorded IDs."""

    def normalize(self, points):
        return knn_features(points)


def ml_vector(points):
    """63-d palm-local vector used by every live classifier."""
    return knn_features(points)


def ml_vector_from_sample(sample):
    return features_from_sample(sample)


def ml_vector_extended(points):
    """63 palm-local coords plus 10 shape distances. Benchmark only unless selected."""
    local = local_palm_points(points)
    return np.concatenate((local.ravel(), shape_features(local)))
