"""Structural similarity between songs via Sakoe-Chiba DTW & FastDTW.

DTW uses the symmetric Sakoe-Chiba form (P=1, normalised by I+J).
FastDTW achieves O(N) time/space by recursively coarsening, projecting,
and refining the warp path; at radius=1 error is ~8.6% vs optimal.
`lb_keogh` provides an O(N) lower bound for batch pruning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numba import njit


@dataclass
class StructuralSegment:
    """A detected structural section in a single song."""

    start_frame: int
    end_frame: int
    mean_energy: float
    label: str = ""


@dataclass
class StructuralMatch:
    """Result of comparing the structural layouts of two songs."""

    dtw_distance: float  # symmetric DTW distance
    dtw_similarity: float  # 1 / (1 + dtw_distance) in (0, 1]
    ssm_cosine_similarity: float  # cosine between flattened SSMs
    overall_similarity: float  # weighted combination
    cost_matrix: np.ndarray  # accumulated DP cost (I x J)
    warping_path: List[Tuple[int, int]]
    segments_a: List[StructuralSegment]
    segments_b: List[StructuralSegment]
    section_matches: List[dict]


def build_self_similarity_matrix(
    energy: np.ndarray,
    section_frames: int = 16,
    metric: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    """SSM from a bass log-energy envelope. Returns (ssm [0,1], boundaries)."""
    energy = np.asarray(energy, dtype=np.float32).flatten()
    if energy.size == 0:
        return np.zeros((0, 0), dtype=np.float32), np.array([], dtype=np.int32)

    n_sections = max(1, len(energy) // section_frames)
    trunc = n_sections * section_frames
    padded = np.pad(energy[:trunc], (0, trunc - len(energy[:trunc])), mode="edge")
    section_feats = padded.reshape(n_sections, section_frames)
    boundaries = np.array(
        list(range(0, trunc, section_frames)) + [len(energy)], dtype=np.int32
    )

    if metric == "cosine":
        normed = section_feats / np.maximum(
            np.linalg.norm(section_feats, axis=1, keepdims=True), 1e-8
        )
        ssm = (np.clip(normed @ normed.T, -1.0, 1.0).astype(np.float32) + 1.0) / 2.0
    elif metric == "euclidean":
        diff = section_feats[:, None, :] - section_feats[None, :, :]
        ssm = (1.0 / (1.0 + np.linalg.norm(diff, axis=-1))).astype(np.float32)
    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    return ssm, boundaries


@njit(cache=True)
def _sakoe_chiba_dtw(a: np.ndarray, b: np.ndarray, window: int) -> Tuple:
    """Exact symmetric Sakoe-Chiba DTW, P=1, normalised by I+J.

    Base case for fast_dtw when sequences are short (≤ radius+2).
    """
    I, J = len(a), len(b)
    INF = np.inf
    g = np.full((I + 2, J + 2), INF, dtype=np.float64)
    g[1, 1] = 2.0 * abs(a[0] - b[0])

    for i in range(1, I + 1):
        for j in range(max(1, i - window), min(J, i + window) + 1):
            ai, bj = i - 1, j - 1
            dij = abs(a[ai] - b[bj])
            c1 = g[i - 1, j - 1] + 2.0 * dij
            c2 = g[i - 1, j - 2] + 2.0 * abs(a[ai] - b[j - 2]) + dij if j >= 2 else INF
            c3 = g[i - 2, j - 1] + 2.0 * abs(a[i - 2] - b[bj]) + dij if i >= 2 else INF
            g[i, j] = min(c1, min(c2, c3))

    raw = g[I, J]
    dist = raw / (I + J) if not np.isinf(raw) else np.inf
    return dist, g, _backtrack_path(g, I, J, window)


@njit(cache=True)
def _backtrack_path(g: np.ndarray, I: int, J: int, window: int) -> np.ndarray:
    """Greedy backtrack through g (1-indexed). Returns 0-indexed (K,2) path."""
    path_arr = np.zeros((I + J, 2), dtype=np.int32)
    idx = 0
    i, j = I, J

    while i > 1 or j > 1:
        path_arr[idx, 0] = i - 1
        path_arr[idx, 1] = j - 1
        idx += 1
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            best_cost = np.inf
            next_i, next_j = i - 1, j - 1
            for ni, nj in ((i - 1, j - 1), (i - 1, j - 2), (i - 2, j - 1)):
                if (
                    ni >= 1
                    and nj >= 1
                    and abs(ni - nj) <= window
                    and g[ni, nj] < best_cost
                ):
                    best_cost, next_i, next_j = g[ni, nj], ni, nj
            i, j = next_i, next_j

    path_arr[idx, 0] = 0
    path_arr[idx, 1] = 0
    return path_arr[: idx + 1][::-1]


@njit(cache=True)
def _coarsen(seq: np.ndarray) -> np.ndarray:
    """Halve by averaging adjacent pairs; last element repeated for odd length."""
    n = len(seq)
    half = (n + 1) // 2
    out = np.empty(half, dtype=np.float32)
    for i in range(half):
        j = i * 2
        out[i] = (seq[j] + seq[j + 1]) / 2.0 if j + 1 < n else seq[j]
    return out


@njit(cache=True)
def _build_search_window(
    path: np.ndarray, len_a: int, len_b: int, radius: int
) -> np.ndarray:
    """Project a low-res warp path to full resolution and expand by radius.

    Returns window[i] = [j_lo, j_hi] (inclusive, 0-indexed).
    """
    window = np.full((len_a, 2), -1, dtype=np.int32)

    for k in range(len(path)):
        pi = int(path[k, 0])
        pj = int(path[k, 1])
        for di in range(2):
            i_full = 2 * pi + di
            if i_full >= len_a:
                continue
            j_lo = max(0, 2 * pj - radius)
            j_hi = min(len_b - 1, 2 * pj + 1 + radius)
            if window[i_full, 0] == -1:
                window[i_full, 0] = j_lo
                window[i_full, 1] = j_hi
            else:
                if j_lo < window[i_full, 0]:
                    window[i_full, 0] = j_lo
                if j_hi > window[i_full, 1]:
                    window[i_full, 1] = j_hi

    # fill any uncovered rows via linear interpolation
    denom = len_a - 1 if len_a > 1 else 1
    for i in range(len_a):
        if window[i, 0] == -1:
            j = int(round(i * (len_b - 1) / denom))
            window[i, 0] = max(0, j - radius)
            window[i, 1] = min(len_b - 1, j + radius)

    return window


@njit(cache=True)
def _constrained_dtw_symmetric(
    a: np.ndarray, b: np.ndarray, window: np.ndarray
) -> np.ndarray:
    """Symmetric P=1 DTW restricted to a search window. Returns 1-indexed cost matrix g."""
    I, J = len(a), len(b)
    INF = np.inf
    g = np.full((I + 2, J + 2), INF, dtype=np.float64)
    g[1, 1] = 2.0 * abs(a[0] - b[0])

    for i in range(1, I + 1):
        j_lo = max(1, int(window[i - 1, 0]) + 1)
        j_hi = min(J, int(window[i - 1, 1]) + 1)
        for j in range(j_lo, j_hi + 1):
            ai, bj = i - 1, j - 1
            dij = abs(a[ai] - b[bj])
            c1 = g[i - 1, j - 1] + 2.0 * dij
            c2 = g[i - 1, j - 2] + 2.0 * abs(a[ai] - b[j - 2]) + dij if j >= 2 else INF
            c3 = g[i - 2, j - 1] + 2.0 * abs(a[i - 2] - b[bj]) + dij if i >= 2 else INF
            g[i, j] = min(c1, min(c2, c3))

    return g


def fast_dtw(
    seq_a: np.ndarray,
    seq_b: np.ndarray,
    radius: int = 1,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """FastDTW: O(N) approximate DTW via iterative multilevel coarsen→project→refine.

    Replaces the previous recursive implementation. All inner kernels
    (_coarsen, _build_search_window, _constrained_dtw_symmetric) are
    @njit-compiled, so only the Python-level loop over resolution levels
    remains interpreted — typically 4–8 levels for song-length sequences.

    radius=1: ~8.6% error; radius=2: ~5%; radius=10: ~1.5% (Salvador & Chan).
    Falls back to exact _sakoe_chiba_dtw when sequences are short (≤ radius+2).

    Returns: (normalised_distance, cost_matrix (I×J), path_array (K,2)).
    """
    a = np.asarray(seq_a, dtype=np.float32).flatten()
    b = np.asarray(seq_b, dtype=np.float32).flatten()

    if a.size == 0 or b.size == 0:
        return (
            np.inf,
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 2), dtype=np.int32),
        )

    min_size = radius + 2

    # build the coarsened resolution stack iteratively
    resolutions: List[Tuple[np.ndarray, np.ndarray]] = []
    ca, cb = a, b
    while len(ca) > min_size and len(cb) > min_size:
        resolutions.append((ca, cb))
        ca = _coarsen(ca)
        cb = _coarsen(cb)

    # base case -> exact DTW on the coarsest level
    base_dist, base_g, path = _sakoe_chiba_dtw(ca, cb, window=max(len(ca), len(cb)))

    # If we never entered the loop (sequences were already short), return now
    if not resolutions:
        I, J = len(ca), len(cb)
        return (
            float(base_dist),
            base_g[1 : I + 1, 1 : J + 1].astype(np.float32),
            path,
        )

    # refine upward through each resolution level
    for ra, rb in reversed(resolutions):
        I, J = len(ra), len(rb)
        win = _build_search_window(path, I, J, radius)
        g = _constrained_dtw_symmetric(ra, rb, win)
        path = _backtrack_path(g, I, J, max(I, J))

    raw = float(g[I, J])
    dist = raw / (I + J) if not np.isinf(raw) else np.inf
    return dist, g[1 : I + 1, 1 : J + 1].astype(np.float32), path


@njit(cache=True)
def get_keogh_envelope(query: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-point min/max envelope of query within ±window. Precompute once per batch."""
    n = len(query)
    U = np.empty(n, dtype=np.float64)
    L = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = max(0, i - window)
        e = min(n, i + window + 1)
        U[i] = np.max(query[s:e])
        L[i] = np.min(query[s:e])
    return U, L


@njit(cache=True)
def lb_keogh(candidate: np.ndarray, U: np.ndarray, L: np.ndarray) -> float:
    """O(N) DTW lower bound. If ≥ current best, skip fast_dtw entirely."""
    lb = 0.0
    for i in range(len(candidate)):
        if candidate[i] > U[i]:
            lb += (candidate[i] - U[i]) ** 2
        elif candidate[i] < L[i]:
            lb += (L[i] - candidate[i]) ** 2
    return lb**0.5


def _ssm_diag(ssm: np.ndarray) -> np.ndarray:
    """Extract the main diagonal as a float32 1-D array."""
    return np.diag(ssm).astype(np.float32)


def _assign_repeat_labels(
    ssm: np.ndarray,
    boundaries: List[int],
    sim_threshold: float = 0.65,
) -> List[str]:
    """Letter-label segments; same letter = mean cross-SSM ≥ sim_threshold."""
    n_segs = len(boundaries) - 1
    labels = ["?"] * n_segs
    next_label = ord("A")
    for i in range(n_segs):
        if labels[i] != "?":
            continue
        labels[i] = chr(next_label)
        si, ei = boundaries[i], boundaries[i + 1]
        for j in range(i + 1, n_segs):
            if labels[j] != "?":
                continue
            cross = ssm[si:ei, boundaries[j] : boundaries[j + 1]]
            if cross.size > 0 and float(np.mean(cross)) >= sim_threshold:
                labels[j] = chr(next_label)
        next_label += 1
    return labels


def detect_structural_segments(
    ssm: np.ndarray,
    boundaries: np.ndarray,
    novelty_threshold: float = 0.15,
    min_segment_sections: int = 2,
) -> List[StructuralSegment]:
    """Segment detection via checkerboard-kernel novelty curve."""
    n = ssm.shape[0]
    if n < 2:
        s = int(boundaries[0]) if len(boundaries) > 0 else 0
        e = int(boundaries[-1]) if len(boundaries) > 1 else n
        return [
            StructuralSegment(
                start_frame=s, end_frame=e, mean_energy=float(np.mean(ssm))
            )
        ]

    k = max(1, min(4, n // 4))
    kernel = np.ones((2 * k, 2 * k), dtype=np.float32)
    kernel[:k, k:] = kernel[k:, :k] = -1.0

    novelty = np.zeros(n, dtype=np.float32)
    for t in range(k, n - k):
        patch = ssm[t - k : t + k, t - k : t + k]
        if patch.shape == kernel.shape:
            novelty[t] = float(np.sum(patch * kernel))

    nov_range = novelty.max() - novelty.min()
    if nov_range > 1e-8:
        novelty = (novelty - novelty.min()) / nov_range

    struct_bounds = [0]
    for t in range(1, n - 1):
        if (
            novelty[t] > novelty_threshold
            and novelty[t] >= novelty[t - 1]
            and novelty[t] >= novelty[t + 1]
            and t - struct_bounds[-1] >= min_segment_sections
        ):
            struct_bounds.append(t)
    struct_bounds.append(n)

    labels = _assign_repeat_labels(ssm, struct_bounds)
    return [
        StructuralSegment(
            start_frame=int(boundaries[s]),
            end_frame=int(boundaries[min(e, len(boundaries) - 1)]),
            mean_energy=float(np.mean(ssm[s:e, s:e])),
            label=labels[idx],
        )
        for idx, (s, e) in enumerate(zip(struct_bounds[:-1], struct_bounds[1:]))
    ]


def match_structural_sections(
    segments_a: List[StructuralSegment],
    segments_b: List[StructuralSegment],
    ssm_a: np.ndarray,
    ssm_b: np.ndarray,
    section_frames: int,
) -> List[dict]:
    """Match each segment in A to its most cosine-similar segment in B."""

    def _profile(ssm: np.ndarray, seg: StructuralSegment) -> np.ndarray:
        i0 = min(seg.start_frame // section_frames, ssm.shape[0] - 1)
        i1 = min(max(i0 + 1, seg.end_frame // section_frames), ssm.shape[0])
        return np.mean(ssm[i0:i1, :], axis=0).astype(np.float32)

    def _cos(pa: np.ndarray, pb: np.ndarray) -> float:
        n = min(len(pa), len(pb))
        if n == 0:
            return 0.0
        pa, pb = pa[:n], pb[:n]
        na, nb = float(np.linalg.norm(pa)), float(np.linalg.norm(pb))
        return float(np.dot(pa, pb) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 0.0

    matches = []
    for seg_a in segments_a:
        prof_a = _profile(ssm_a, seg_a)
        best_sim, best_b = -1.0, None
        for seg_b in segments_b:
            sim = _cos(prof_a, _profile(ssm_b, seg_b))
            if sim > best_sim:
                best_sim, best_b = sim, seg_b
        matches.append(
            {
                "segment_a": {
                    "start_frame": seg_a.start_frame,
                    "end_frame": seg_a.end_frame,
                    "label": seg_a.label,
                },
                "segment_b": {
                    "start_frame": best_b.start_frame if best_b else -1,
                    "end_frame": best_b.end_frame if best_b else -1,
                    "label": best_b.label if best_b else "?",
                },
                "similarity": float(best_sim),
                "label_match": best_b is not None and seg_a.label == best_b.label,
            }
        )
    return matches


def compare_song_structures(
    S_bass_a: np.ndarray,
    S_bass_b: np.ndarray,
    section_frames: int = 16,
    fast_dtw_radius: int = 1,
    ssm_metric: str = "cosine",
    novelty_threshold: float = 0.15,
    dtw_weight: float = 0.5,
    ssm_weight: float = 0.5,
) -> StructuralMatch:
    """Compare large-scale structural similarity of two songs.

    Builds SSMs, runs FastDTW on their diagonals, detects and matches
    segments, returns a StructuralMatch with DTW + cosine scores blended.
    """
    from .pattern_matching import compute_bass_spectrogram_features

    _empty = StructuralMatch(
        np.inf, 0.0, 0.0, 0.0, np.zeros((0, 0), dtype=np.float32), [], [], [], []
    )

    energy_a = compute_bass_spectrogram_features(S_bass_a)
    energy_b = compute_bass_spectrogram_features(S_bass_b)
    if energy_a.size == 0 or energy_b.size == 0:
        return _empty

    ssm_a, bounds_a = build_self_similarity_matrix(energy_a, section_frames, ssm_metric)
    ssm_b, bounds_b = build_self_similarity_matrix(energy_b, section_frames, ssm_metric)
    if ssm_a.size == 0 or ssm_b.size == 0:
        return _empty

    # SSM cosine similarity (upper triangle, trimmed to same size)
    n_min = min(ssm_a.shape[0], ssm_b.shape[0])
    idx = np.triu_indices(n_min, k=0)
    flat_a = ssm_a[:n_min, :n_min][idx].astype(np.float32)
    flat_b = ssm_b[:n_min, :n_min][idx].astype(np.float32)
    na, nb = float(np.linalg.norm(flat_a)), float(np.linalg.norm(flat_b))
    ssm_cos_sim = float(
        np.clip(
            np.dot(flat_a, flat_b) / (na * nb) if na > 1e-8 and nb > 1e-8 else 0.0,
            0.0,
            1.0,
        )
    )

    diag_a = _ssm_diag(ssm_a)
    diag_b = _ssm_diag(ssm_b)
    dtw_dist, cost_mat, path_arr = fast_dtw(diag_a, diag_b, radius=fast_dtw_radius)
    dtw_sim = float(1.0 / (1.0 + dtw_dist)) if not np.isinf(dtw_dist) else 0.0
    warp_path = [
        (int(path_arr[k, 0]), int(path_arr[k, 1])) for k in range(len(path_arr))
    ]

    segments_a = detect_structural_segments(ssm_a, bounds_a, novelty_threshold)
    segments_b = detect_structural_segments(ssm_b, bounds_b, novelty_threshold)
    section_matches = match_structural_sections(
        segments_a, segments_b, ssm_a, ssm_b, section_frames
    )

    overall = (dtw_weight * dtw_sim + ssm_weight * ssm_cos_sim) / (
        dtw_weight + ssm_weight
    )
    return StructuralMatch(
        dtw_dist,
        dtw_sim,
        ssm_cos_sim,
        float(overall),
        cost_mat,
        warp_path,
        segments_a,
        segments_b,
        section_matches,
    )


def batch_compare_structures(
    query_S_bass: np.ndarray,
    candidate_S_bass_list: List[np.ndarray],
    section_frames: int = 16,
    fast_dtw_radius: int = 1,
    ssm_metric: str = "cosine",
) -> List[float]:
    """Rank candidates against a query using LB-Keogh pruning + FastDTW.

    Keogh envelope is computed once; candidates whose lower bound ≥ current
    best skip the full fast_dtw call entirely.
    Returns DTW distances in candidate_S_bass_list order.
    """
    from .pattern_matching import compute_bass_spectrogram_features

    q_energy = compute_bass_spectrogram_features(query_S_bass)
    if q_energy.size == 0:
        return [np.inf] * len(candidate_S_bass_list)

    q_ssm, _ = build_self_similarity_matrix(q_energy, section_frames, ssm_metric)
    q_diag = _ssm_diag(q_ssm).astype(np.float64)
    U, L = get_keogh_envelope(q_diag, fast_dtw_radius)

    distances: List[float] = []
    best_dist = np.inf

    for cand in candidate_S_bass_list:
        c_energy = compute_bass_spectrogram_features(cand)
        if c_energy.size == 0:
            distances.append(np.inf)
            continue
        c_ssm, _ = build_self_similarity_matrix(c_energy, section_frames, ssm_metric)
        c_diag = _ssm_diag(c_ssm).astype(np.float64)
        n = min(len(q_diag), len(c_diag))
        if lb_keogh(c_diag[:n], U[:n], L[:n]) >= best_dist:
            distances.append(np.inf)
            continue
        dist, _, _ = fast_dtw(
            q_diag.astype(np.float32), c_diag.astype(np.float32), fast_dtw_radius
        )
        distances.append(dist)
        if dist < best_dist:
            best_dist = dist

    return distances
