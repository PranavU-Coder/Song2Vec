"""Pattern matching for bass sequences using temporal alignment.

This module detects similar bass patterns between two songs by analyzing
how bass energy evolves over time, similar to Shazam fingerprinting.

Methods:
    1. Extract bass spectrogram (20-250 Hz over time)
    2. Compute cross-correlation to find aligned segments
    3. Calculate frame-by-frame similarity scores
    4. Return matched regions and overall similarity
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import signal


class PatternMatch(NamedTuple):
    """Result of pattern matching between two bass sequences."""

    overall_similarity: float
    matched_segments: list[dict]
    correlation: np.ndarray
    lags: np.ndarray
    frame_similarity: np.ndarray


def compute_bass_spectrogram_features(S_bass: np.ndarray) -> np.ndarray:
    """Reduce bass spectrogram to a 1D per-frame energy envelope.

    Args:
        S_bass: Bass spectrogram shape (n_freq_bass, n_frames)

    Returns:
        energy_t: shape (n_frames,) - per-frame bass energy
    """
    if S_bass.size == 0:
        return np.array([], dtype=np.float32)

    # Sum energy across frequency bins, log scale
    energy = np.sum(S_bass**2, axis=0)
    energy = np.maximum(energy, 1e-10)  # Avoid log(0)
    return np.log(energy).astype(np.float32)


def cross_correlate_patterns(pattern_a: np.ndarray, pattern_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized cross-correlation between two patterns.

    Args:
        pattern_a, pattern_b: 1D bass energy sequences

    Returns:
        correlation: cross-correlation values
        lags: lag positions
    """
    # Normalize patterns
    a = np.asarray(pattern_a, dtype=np.float32)
    b = np.asarray(pattern_b, dtype=np.float32)

    if a.size == 0 or b.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    # Standardize
    a_std = np.std(a)
    b_std = np.std(b)

    if a_std < 1e-8 or b_std < 1e-8:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int32)

    a_norm = (a - np.mean(a)) / a_std
    b_norm = (b - np.mean(b)) / b_std

    # Cross-correlate
    correlation = signal.correlate(a_norm, b_norm, mode="full", method="auto")
    lags = signal.correlation_lags(len(a_norm), len(b_norm), mode="full")

    # Normalize by length
    correlation = correlation / max(len(a_norm), len(b_norm))

    return correlation.astype(np.float32), lags.astype(np.int32)


def dtw_distance(a: np.ndarray, b: np.ndarray, window: int = 50) -> tuple[float, np.ndarray]:
    """Compute Dynamic Time Warping distance with Sakoe-Chiba band constraint.

    Args:
        a, b: 1D sequences
        window: Sakoe-Chiba band constraint (frames)

    Returns:
        distance: DTW distance (normalized)
        cost_matrix: accumulated cost matrix for visualization
    """
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()

    if a.size == 0 or b.size == 0:
        return float("inf"), np.array([], dtype=np.float32)

    n, m = len(a), len(b)

    # Initialize cost matrix (very large values where we don't search)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0

    # Fill with Sakoe-Chiba band constraint
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if abs(i - j) <= window:
                d = abs(a[i - 1] - b[j - 1])
                cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Normalize by path length
    distance = cost[n, m] / (n + m)
    return float(distance), cost[1:, 1:].astype(np.float32)


def frame_wise_similarity(pattern_a: np.ndarray, pattern_b: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Compute per-frame similarity using sliding windows.

    Args:
        pattern_a, pattern_b: 1D bass energy sequences
        window_size: Frames to average for smoothing

    Returns:
        similarities: shape (max(len_a, len_b),) with similarity at each frame
    """
    a = np.asarray(pattern_a, dtype=np.float32).flatten()
    b = np.asarray(pattern_b, dtype=np.float32).flatten()

    if a.size == 0 or b.size == 0:
        return np.array([], dtype=np.float32)

    # Pad shorter sequence with edge values
    max_len = max(len(a), len(b))
    a_padded = np.pad(a, (0, max_len - len(a)), mode="edge")
    b_padded = np.pad(b, (0, max_len - len(b)), mode="edge")

    # Compute per-frame differences
    frame_diffs = np.abs(a_padded - b_padded)

    # Normalize to [0, 1]
    if np.max(frame_diffs) > 0:
        frame_diffs = frame_diffs / np.max(frame_diffs)

    # Convert to similarity (invert distance)
    similarities = 1.0 - frame_diffs

    # Smooth with window
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        similarities = np.convolve(similarities, kernel, mode="same")

    return similarities.astype(np.float32)


def detect_pattern_matches(
    similarities: np.ndarray,
    threshold: float = 0.6,
    min_segment_length: int = 5,
) -> list[dict]:
    """Detect contiguous regions of high similarity.

    Args:
        similarities: Per-frame similarity scores in [0, 1]
        threshold: Similarity threshold to consider as match
        min_segment_length: Minimum frames for a match

    Returns:
        List of matched segments with start, end, and mean similarity
    """
    if similarities.size == 0:
        return []

    # Find regions above threshold
    above_threshold = similarities >= threshold
    if not np.any(above_threshold):
        return []

    # Find contiguous segments
    segments = []
    in_segment = False
    start_idx = 0

    for i, is_match in enumerate(np.concatenate(([False], above_threshold, [False]))):
        if is_match and not in_segment:
            start_idx = i
            in_segment = True
        elif not is_match and in_segment:
            end_idx = i
            length = end_idx - start_idx
            if length >= min_segment_length:
                mean_sim = float(np.mean(similarities[start_idx:end_idx]))
                segments.append({
                    "start_frame": int(start_idx),
                    "end_frame": int(end_idx),
                    "length_frames": int(length),
                    "mean_similarity": mean_sim,
                })
            in_segment = False

    return segments


def match_bass_patterns(
    S_bass_a: np.ndarray,
    S_bass_b: np.ndarray,
    sr: int,
    hop_length: int,
    use_dtw: bool = True,
) -> PatternMatch:
    """Compare bass patterns between two spectrograms.

    Args:
        S_bass_a, S_bass_b: Bass spectrograms shape (n_freq, n_frames)
        sr: Sample rate
        hop_length: STFT hop length in samples
        use_dtw: If True, use DTW for alignment; else use cross-correlation

    Returns:
        PatternMatch with similarity scores and matched regions
    """
    # Extract 1D bass energy envelopes
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

    # Compute frame-by-frame similarity
    frame_sim = frame_wise_similarity(energy_a, energy_b, window_size=3)

    # Compute overall alignment
    if use_dtw:
        dtw_dist, _ = dtw_distance(energy_a, energy_b)
        # Convert DTW distance to similarity (lower distance = higher similarity)
        overall_sim = max(0.0, 1.0 - dtw_dist)
    else:
        correlation, lags = cross_correlate_patterns(energy_a, energy_b)
        if correlation.size > 0:
            overall_sim = float(np.max(correlation))
        else:
            overall_sim = 0.0

    # Detect high-similarity segments
    threshold = 0.5
    matched = detect_pattern_matches(frame_sim, threshold=threshold, min_segment_length=3)

    correlation, lags = cross_correlate_patterns(energy_a, energy_b)

    return PatternMatch(
        overall_similarity=float(overall_sim),
        matched_segments=matched,
        correlation=correlation,
        lags=lags,
        frame_similarity=frame_sim,
    )