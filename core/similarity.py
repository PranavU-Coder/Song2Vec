from __future__ import annotations

from typing import Literal

import numpy as np
from numba import njit


@njit(cache=True)
def _cosine_similarity_kernel(
    vec_a: np.ndarray, vec_b: np.ndarray, eps: float
) -> float:
    """Compiled cosine kernel. Expects equal-length 1-D float32 arrays — no validation."""
    denom = (vec_a**2).sum() ** 0.5 * (vec_b**2).sum() ** 0.5
    if denom < eps:
        return 0.0
    return np.dot(vec_a, vec_b) / denom


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-8) -> float:
    """Cosine similarity in [-1, 1].

    Accepts any array-like input; normalises to 1-D float32 and validates
    size before dispatching to the compiled kernel.

    Args:
        vec_a, vec_b: Feature vectors (any shape; flattened internally).
        eps:          Denominator floor to avoid division by zero.

    Returns:
        Cosine similarity in [-1, 1], or 0.0 if either vector is near-zero.

    Raises:
        ValueError: if the flattened vectors have different lengths.
    """
    a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"Vector size mismatch: {a.size} vs {b.size}")
    return float(_cosine_similarity_kernel(a, b, eps))


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

    - cosine:    returned as-is in [-1, 1].
    - euclidean: converted to (0, 1] via 1 / (1 + d).
    """
    if metric == "cosine":
        return cosine_similarity(vec_a, vec_b)
    if metric == "euclidean":
        return 1.0 / (1.0 + euclidean_distance(vec_a, vec_b))
    raise ValueError(f"Unknown metric: {metric!r}")
