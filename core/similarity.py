from __future__ import annotations

from typing import Literal

import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-8) -> float:
    """Compute cosine similarity in [-1, 1]."""

    a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"Vector size mismatch: {a.size} vs {b.size}")

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < eps:
        return 0.0
    return float(np.dot(a, b) / denom)


def euclidean_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute Euclidean distance (L2)."""

    a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"Vector size mismatch: {a.size} vs {b.size}")
    return float(np.linalg.norm(a - b))


def similarity_score(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    metric: Literal["cosine", "euclidean"] = "cosine",
) -> float:
    """Compute a similarity score with a chosen metric.

    Notes:
        - Cosine similarity is returned as-is.
        - Euclidean distance is converted to a bounded similarity in (0, 1] via 1/(1+d).

    Args:
        vec_a: Feature vector.
        vec_b: Feature vector.
        metric: "cosine" or "euclidean".

    Returns:
        Similarity score.
    """

    if metric == "cosine":
        return cosine_similarity(vec_a, vec_b)

    if metric == "euclidean":
        d = euclidean_distance(vec_a, vec_b)
        return float(1.0 / (1.0 + d))

    raise ValueError(f"Unknown metric: {metric!r}")
