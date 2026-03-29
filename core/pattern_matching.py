"""Pattern matching for bass sequences using temporal alignment.

Methods:
    1. Extract bass spectrogram (20-250 Hz over time)
    2. Compute cross-correlation to find aligned segments
    3. Calculate frame-by-frame similarity scores
    4. Return matched regions and overall similarity
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from numba import njit
from scipy import signal

from .dtw import fast_dtw


class PatternMatch(NamedTuple):
    """Result of pattern matching between two bass sequences."""

    overall_similarity: float
    matched_segments: list[dict]
    correlation: np.ndarray
    lags: np.ndarray
    frame_similarity: np.ndarray


def compute_bass_spectrogram_features(S_bass: np.ndarray) -> np.ndarray:
    """Reduce bass spectrogram to a 1-D per-frame log-energy envelope.

    Args:
        S_bass: shape (n_freq_bass, n_frames)

    Returns:
        shape (n_frames,), float32 log-energy per frame.
    """
    if S_bass.size == 0:
        return np.array([], dtype=np.float32)
    energy = np.sum(S_bass**2, axis=0)
    return np.log(np.maximum(energy, 1e-10)).astype(np.float32)


def cross_correlate_patterns(
    pattern_a: np.ndarray, pattern_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised cross-correlation between two 1-D bass energy sequences."""
    a = np.asarray(pattern_a, dtype=np.float32)
    b = np.asarray(pattern_b, dtype=np.float32)

    if a.size == 0 or b.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    a_std, b_std = np.std(a), np.std(b)
    if a_std < 1e-8 or b_std < 1e-8:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    a_norm = (a - np.mean(a)) / a_std
    b_norm = (b - np.mean(b)) / b_std

    correlation = signal.correlate(a_norm, b_norm, mode="full", method="auto")
    lags = signal.correlation_lags(len(a_norm), len(b_norm), mode="full")
    correlation /= max(len(a_norm), len(b_norm))

    return correlation.astype(np.float32), lags.astype(np.int32)


def dtw_distance(
    a: np.ndarray, b: np.ndarray, window: int = 1
) -> tuple[float, np.ndarray]:
    """DTW distance via fast_dtw (symmetric Sakoe-Chiba P=1, normalised by I+J).

    Args:
        a, b:   1-D sequences.
        window: FastDTW radius. Default 1 gives ~8.6% error at O(N) cost.
                Use 2 for ~5% error. The old window=50 Sakoe-Chiba band
                semantics are not equivalent; radius controls search width
                around the projected path, not a global diagonal band.

    Returns:
        (normalised_distance, cost_matrix (I×J))
    """
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    if a.size == 0 or b.size == 0:
        return np.inf, np.array([], dtype=np.float32)
    dist, cost, _ = fast_dtw(a, b, radius=window)
    return dist, cost


@njit(cache=True)
def frame_wise_similarity(
    energy_a: np.ndarray, energy_b: np.ndarray, window_size: int = 3
) -> np.ndarray:
    """Per-frame best cosine similarity via sliding-window context vectors.

    For each frame i in energy_a, finds the most similar frame j in energy_b
    by comparing their ±window_size context neighbourhoods with cosine
    similarity.  Returns a 1-D array (not an N×N matrix) because only the
    per-row maximum is needed downstream.

    The norm and dot product are always computed over the same aligned slice
    so that edge frames (where one window is shorter) are compared fairly.

    Complexity: O(N_a × N_b × w) — compiled by numba.

    Args:
        energy_a, energy_b: 1-D log-energy envelopes, float32.
        window_size:         Half-width of the context window.

    Returns:
        best_sim: shape (len(energy_a),), float32.
    """
    n_a = len(energy_a)
    n_b = len(energy_b)
    best_sim = np.zeros(n_a, dtype=np.float32)

    for i in range(n_a):
        sa = max(0, i - window_size)
        ea = min(n_a, i + window_size + 1)
        # half-widths actually available on each side for frame i
        left_i = i - sa
        right_i = ea - i - 1

        best = -1.0
        for j in range(n_b):
            # Align the context window around j to the same shape as around i
            sa_j = max(0, j - left_i)
            ea_j = min(n_b, j + right_i + 1)
            # Further clip so both windows are the same length
            l = min(ea - sa, ea_j - sa_j)
            wa = energy_a[sa : sa + l]
            wb = energy_b[sa_j : sa_j + l]

            norm_a = 0.0
            norm_b = 0.0
            dot = 0.0
            for k in range(l):
                norm_a += wa[k] * wa[k]
                norm_b += wb[k] * wb[k]
                dot += wa[k] * wb[k]

            norm_a = norm_a**0.5
            norm_b = norm_b**0.5
            if norm_a < 1e-8 or norm_b < 1e-8:
                continue

            sim = dot / (norm_a * norm_b)
            if sim > best:
                best = sim

        if best >= 0.0:
            best_sim[i] = best

    return best_sim


def detect_pattern_matches(
    similarities: np.ndarray,
    threshold: float = 0.6,
    min_segment_length: int = 5,
) -> list[dict]:
    """Detect contiguous regions above `threshold` in a 1-D similarity array.

    start_frame and end_frame are 0-indexed into `similarities`;
    end_frame is exclusive (half-open interval [start, end)).
    """
    if similarities.size == 0:
        return []

    above = similarities >= threshold
    if not np.any(above):
        return []

    segments = []
    in_seg = False
    start = 0
    n = len(above)

    for i in range(n + 1):
        hit = above[i] if i < n else False
        if hit and not in_seg:
            start = i
            in_seg = True
        elif not hit and in_seg:
            length = i - start
            if length >= min_segment_length:
                segments.append(
                    {
                        "start_frame": int(start),
                        "end_frame": int(i),
                        "length_frames": int(length),
                        "mean_similarity": float(np.mean(similarities[start:i])),
                    }
                )
            in_seg = False

    return segments


def match_bass_patterns(
    S_bass_a: np.ndarray,
    S_bass_b: np.ndarray,
    use_dtw: bool = True,
    dtw_radius: int = 1,
    # Deprecated — kept for one release so existing callers don't break.
    # sr and hop_length are no longer used internally; pass them and a
    # DeprecationWarning is raised but the call succeeds.
    sr: int | None = None,
    hop_length: int | None = None,
) -> PatternMatch:
    """Compare bass patterns between two spectrograms.

    Args:
        S_bass_a, S_bass_b: Bass spectrograms, shape (n_freq, n_frames).
        use_dtw:            Use fast_dtw for overall alignment score;
                            otherwise use cross-correlation peak.
        dtw_radius:         FastDTW search radius forwarded to fast_dtw.
        sr:                 Deprecated. No longer used. Will be removed.
        hop_length:         Deprecated. No longer used. Will be removed.

    Returns:
        PatternMatch with scores, matched segments, correlation, and
        per-frame similarity.
    """
    if sr is not None or hop_length is not None:
        warnings.warn(
            "The 'sr' and 'hop_length' arguments to match_bass_patterns() are "
            "deprecated and will be removed in the next release. "
            "They are no longer used internally.",
            DeprecationWarning,
            stacklevel=2,
        )
    energy_a = compute_bass_spectrogram_features(S_bass_a)
    energy_b = compute_bass_spectrogram_features(S_bass_b)

    if energy_a.size == 0 or energy_b.size == 0:
        return PatternMatch(
            overall_similarity=0.0,
            matched_segments=[],
            correlation=np.array([], dtype=np.float32),
            lags=np.array([], dtype=np.int32),
            frame_similarity=np.array([], dtype=np.float32),
        )

    frame_sim = frame_wise_similarity(energy_a, energy_b, window_size=3)

    if use_dtw:
        dist, _ = dtw_distance(energy_a, energy_b, window=dtw_radius)
        # 1/(1+dist) maps [0, ∞) → (0, 1] monotonically.
        # max(0, 1-dist) would collapse to 0 for any dist > 1, which is
        # common since FastDTW returns unnormalised log-energy distances.
        overall_sim = 1.0 / (1.0 + dist) if not np.isinf(dist) else 0.0
    else:
        correlation, lags = cross_correlate_patterns(energy_a, energy_b)
        overall_sim = float(np.max(correlation)) if correlation.size > 0 else 0.0

    matched = detect_pattern_matches(frame_sim, threshold=0.5, min_segment_length=3)
    correlation, lags = cross_correlate_patterns(energy_a, energy_b)

    return PatternMatch(
        overall_similarity=float(overall_sim),
        matched_segments=matched,
        correlation=correlation,
        lags=lags,
        frame_similarity=frame_sim,
    )
