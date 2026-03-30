"""Pattern matching for bass sequences using temporal alignment.

Methods:
    1. Extract bass spectrogram (20-250 Hz over time)
    2. Compute per-frame energy envelope and onset strength
    3. Calculate frame-by-frame similarity on both energy and rhythm
    4. Use DTW on z-normalized envelopes to avoid loudness bias
    5. Return matched regions with mapping to both songs' timelines
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from numba import njit
from scipy import signal

from .dtw import fast_dtw


@dataclass
class PatternMatch:
    """Result of pattern matching between two bass sequences."""

    overall_similarity: float
    matched_segments: list[dict]
    correlation: np.ndarray
    lags: np.ndarray
    frame_similarity: np.ndarray
    energy_a: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    energy_b: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    onset_a: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    onset_b: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    threshold: float = 0.5


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


def compute_onset_envelope(S_bass: np.ndarray) -> np.ndarray:
    """Half-wave rectified spectral flux in the bass band.

    Positive jumps in frame-to-frame log-energy capture rhythmic onsets /
    beat attacks. Negative changes (energy decays) are zeroed out so only
    the attack transients remain — this isolates the rhythm skeleton from
    the overall energy contour.
    """
    if S_bass.size == 0:
        return np.array([], dtype=np.float32)
    energy = np.sum(S_bass**2, axis=0)
    log_energy = np.log(np.maximum(energy, 1e-10))
    flux = np.diff(log_energy, prepend=log_energy[0])
    return np.maximum(flux, 0.0).astype(np.float32)


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
        (normalised_distance, cost_matrix (I x J))
    """
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    if a.size == 0 or b.size == 0:
        return np.inf, np.array([], dtype=np.float32)
    dist, cost, _ = fast_dtw(a, b, radius=window)
    return dist, cost


@njit(cache=True)
def frame_wise_similarity(
    energy_a: np.ndarray,
    energy_b: np.ndarray,
    window_size: int = 8,
    search_radius: int = 8,
) -> np.ndarray:
    """Per-frame cosine similarity using local tempo-aligned search.

    For each frame i in energy_a, this maps to a tempo-aligned center frame in
    energy_b and searches only in a local +/-search_radius neighborhood.
    This avoids inflated scores caused by matching each frame against any
    arbitrary point in the other song.

    The norm and dot product are always computed over the same aligned slice
    so that edge frames (where one window is shorter) are compared fairly.

    Complexity: O(N_a x search_radius x w) -- compiled by numba.

    Args:
        energy_a, energy_b: 1-D log-energy envelopes, float32.
        window_size:         Half-width of the context window (~186ms at 43fps
                             captures beat-level rhythmic context).
        search_radius:       Half-width of search around aligned center in B.

    Returns:
        best_sim: shape (len(energy_a),), float32.
    """
    n_a = len(energy_a)
    n_b = len(energy_b)
    best_sim = np.zeros(n_a, dtype=np.float32)

    denom_a = n_a - 1 if n_a > 1 else 1
    denom_b = n_b - 1 if n_b > 1 else 1

    for i in range(n_a):
        sa = max(0, i - window_size)
        ea = min(n_a, i + window_size + 1)
        left_i = i - sa
        right_i = ea - i - 1

        j_center = int(round(i * denom_b / denom_a))
        j_start = max(0, j_center - search_radius)
        j_end = min(n_b - 1, j_center + search_radius)

        best = -1.0
        for j in range(j_start, j_end + 1):
            sa_j = max(0, j - left_i)
            ea_j = min(n_b, j + right_i + 1)
            l = min(ea - sa, ea_j - sa_j)
            wa = energy_a[sa : sa + l]
            wb = energy_b[sa_j : sa_j + l]

            mean_a = 0.0
            mean_b = 0.0
            for k in range(l):
                mean_a += wa[k]
                mean_b += wb[k]
            mean_a /= l
            mean_b /= l

            norm_a = 0.0
            norm_b = 0.0
            dot = 0.0
            for k in range(l):
                da = wa[k] - mean_a
                db = wb[k] - mean_b
                norm_a += da * da
                norm_b += db * db
                dot += da * db

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


def _map_segment_to_b(seg: dict, n_a: int, n_b: int) -> tuple[int, int]:
    """Map a segment from Song A's frame indices to corresponding Song B frames."""
    denom = n_a - 1 if n_a > 1 else 1
    ratio = (n_b - 1) / denom if denom > 0 else 1.0
    start_b = int(round(seg["start_frame"] * ratio))
    end_b = int(round(seg["end_frame"] * ratio))
    return max(0, start_b), min(n_b, max(end_b, start_b + 1))


def match_bass_patterns(
    S_bass_a: np.ndarray,
    S_bass_b: np.ndarray,
    use_dtw: bool = True,
    dtw_radius: int = 1,
    sr: int | None = None,
    hop_length: int | None = None,
) -> PatternMatch:
    """Compare bass patterns between two spectrograms.

    The scoring combines four independent measures to reduce inflated
    similarity between unrelated songs:

    1. **DTW on z-normalized energy** — elastic alignment of energy
       *shape*, not absolute level.  Z-normalization removes loudness
       bias that made different songs look similar.
    2. **Energy frame similarity** — local windowed correlation of energy
       envelopes with beat-sized context (window_size=8, ~370 ms).
    3. **Onset/rhythm frame similarity** — same windowed correlation but
       on half-wave-rectified spectral flux, isolating beat attacks from
       the energy contour.
    4. **Cross-correlation peak** — global lag-based agreement.

    A power-curve (^1.5) compresses the blended score so that
    moderate raw similarities map to lower displayed percentages,
    giving more separation between genuinely similar and unrelated pairs.

    Args:
        S_bass_a, S_bass_b: Bass spectrograms, shape (n_freq, n_frames).
        use_dtw:            Use fast_dtw for overall alignment score.
        dtw_radius:         FastDTW search radius forwarded to fast_dtw.
        sr:                 Deprecated. No longer used.
        hop_length:         Deprecated. No longer used.

    Returns:
        PatternMatch with scores, matched segments (mapped to both songs),
        correlation, per-frame similarity, energy/onset envelopes, and
        the adaptive threshold used for segment detection.
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
    onset_a = compute_onset_envelope(S_bass_a)
    onset_b = compute_onset_envelope(S_bass_b)

    empty = PatternMatch(
        overall_similarity=0.0,
        matched_segments=[],
        correlation=np.array([], dtype=np.float32),
        lags=np.array([], dtype=np.int32),
        frame_similarity=np.array([], dtype=np.float32),
        energy_a=energy_a,
        energy_b=energy_b,
        onset_a=onset_a,
        onset_b=onset_b,
    )

    if energy_a.size == 0 or energy_b.size == 0:
        return empty

    n_a, n_b = len(energy_a), len(energy_b)

    # Cap search radius to ~350ms (15 frames at ~43 fps).  The old formula
    # grew to hundreds of frames for long songs, letting every frame find a
    # spurious match somewhere in a 10-second window.
    search_radius = min(15, max(6, int(0.01 * max(n_a, n_b))))
    window_size = 8

    energy_frame_sim = frame_wise_similarity(
        energy_a,
        energy_b,
        window_size=window_size,
        search_radius=search_radius,
    )

    onset_frame_sim = frame_wise_similarity(
        onset_a,
        onset_b,
        window_size=window_size,
        search_radius=search_radius,
    )

    # Weight rhythm higher — beat alignment is the strongest discriminator.
    frame_sim = 0.35 * energy_frame_sim + 0.65 * onset_frame_sim
    energy_mean = float(np.mean(energy_frame_sim)) if energy_frame_sim.size > 0 else 0.0
    onset_mean = float(np.mean(onset_frame_sim)) if onset_frame_sim.size > 0 else 0.0

    correlation, lags = cross_correlate_patterns(energy_a, energy_b)
    corr_peak = float(np.max(correlation)) if correlation.size > 0 else 0.0
    corr_component = float(np.clip((corr_peak + 1.0) / 2.0, 0.0, 1.0))

    if use_dtw:
        a_std = float(np.std(energy_a))
        b_std = float(np.std(energy_b))
        energy_a_z = (energy_a - np.mean(energy_a)) / max(a_std, 1e-8)
        energy_b_z = (energy_b - np.mean(energy_b)) / max(b_std, 1e-8)
        dist, _ = dtw_distance(energy_a_z, energy_b_z, window=dtw_radius)
        dtw_component = 1.0 / (1.0 + dist) if not np.isinf(dist) else 0.0

        raw_sim = (
            0.25 * dtw_component
            + 0.25 * energy_mean
            + 0.35 * onset_mean
            + 0.15 * corr_component
        )
    else:
        raw_sim = 0.50 * corr_component + 0.25 * energy_mean + 0.25 * onset_mean

    overall_sim = float(raw_sim**1.5)

    # --- Segment detection with strict thresholds ---
    # 85th percentile with a high floor (0.55) so only genuinely strong
    # alignment passes; min_segment_length=40 (~1 s, roughly 2 beats at
    # 120 BPM) rejects tiny spurious blips.
    adaptive_threshold = (
        float(np.clip(np.percentile(frame_sim, 85), 0.55, 0.85))
        if frame_sim.size > 0
        else 0.6
    )
    matched = detect_pattern_matches(
        frame_sim,
        threshold=adaptive_threshold,
        min_segment_length=40,
    )

    for seg in matched:
        start_b, end_b = _map_segment_to_b(seg, n_a, n_b)
        seg["start_frame_b"] = start_b
        seg["end_frame_b"] = end_b

    # Drop weak segments and reject entire-song false positives.
    matched = [s for s in matched if s["mean_similarity"] >= 0.55]

    if matched and frame_sim.size > 0:
        full_cover = [
            seg
            for seg in matched
            if seg["length_frames"] >= int(0.95 * frame_sim.size)
            and seg["mean_similarity"] >= 0.98
        ]
        if full_cover and (overall_sim < 0.98 or corr_component < 0.98):
            matched = [seg for seg in matched if seg not in full_cover]

    # If the overall similarity is very low the songs are clearly different;
    # any surviving segments are noise.
    if overall_sim < 0.12:
        matched = []

    return PatternMatch(
        overall_similarity=float(np.clip(overall_sim, 0.0, 1.0)),
        matched_segments=matched,
        correlation=correlation,
        lags=lags,
        frame_similarity=frame_sim,
        energy_a=energy_a,
        energy_b=energy_b,
        onset_a=onset_a,
        onset_b=onset_b,
        threshold=adaptive_threshold,
    )
