from __future__ import annotations

from typing import Literal

import numpy as np
from numba import njit


@njit(cache=True)
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-8) -> float:
    """Cosine similarity in [-1, 1]. Expects equal-length 1-D float32 arrays."""
    denom = (vec_a**2).sum() ** 0.5 * (vec_b**2).sum() ** 0.5
    if denom < eps:
        return 0.0
    return np.dot(vec_a, vec_b) / denom


def euclidean_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Euclidean (L2) distance between two vectors."""
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
    """Unified similarity score.

    - cosine:    returned as-is from [-1, 1].
    - euclidean: converted to (0, 1] via 1 / (1 + d).
    """
    if metric == "cosine":
        a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
        b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
        if a.size != b.size:
            raise ValueError(f"Vector size mismatch: {a.size} vs {b.size}")
        return float(cosine_similarity(a, b))
    if metric == "euclidean":
        return 1.0 / (1.0 + euclidean_distance(vec_a, vec_b))
    raise ValueError(f"Unknown metric: {metric!r}")
