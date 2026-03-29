"""Bass feature extraction.

Method overview:
    1) Compute STFT magnitude spectrogram
    2) Keep only bins in the bass band (default 20–250 Hz)
    3) Aggregate per-frame bass energy statistics
    4) Add simple temporal modulation descriptors (how bass energy changes over time)
    5) Optionally add sub-band descriptors (e.g., 20–60, 60–120, 120–250)

The output is a fixed-length feature vector suitable for classical similarity
(cosine) and later ML training.

Notes / references (high-level):
    - STFT-based short-time energy representations are standard in MIR.
    - Sub-band energy features resemble band-limited spectral descriptors used
        widely in audio classification/tagging baselines.
    - Temporal modulation features are inspired by rhythm/energy-envelope features
        used in beat tracking and danceability estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np


@dataclass(frozen=True)
class BassFeatureConfig:
    """Configuration for bass feature extraction."""

    bass_min_hz: float = 20.0
    bass_max_hz: float = 250.0
    # Default sub-bands chosen to split sub-bass / bass / upper-bass.
    # If you set this to an empty list, sub-band features are disabled.
    bass_subbands_hz: Tuple[Tuple[float, float], ...] = (
        (20.0, 60.0),
        (60.0, 120.0),
        (120.0, 250.0),
    )
    n_fft: int = 4096
    hop_length: int = 512
    window: str = "hann"
    center: bool = True


def _safe_percentiles(x: np.ndarray, qs: Sequence[float]) -> List[float]:
    if x.size == 0:
        return [0.0 for _ in qs]
    return [float(v) for v in np.percentile(x, qs)]


def _temporal_centroid(weights: np.ndarray, eps: float = 1e-10) -> float:
    """Energy-weighted centroid over frames, normalized to [0, 1]."""

    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    s = float(np.sum(w))
    if w.size == 0 or s < eps:
        return 0.0
    idx = np.arange(w.size, dtype=np.float32)
    centroid = float(np.sum(idx * w) / s)
    return float(centroid / max(w.size - 1, 1))


def compute_stft_magnitude(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    window: str = "hann",
    center: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute STFT magnitude spectrogram.

    Returns:
        S_mag: shape (n_freq, n_frames)
        freqs_hz: shape (n_freq,)
        times_s: shape (n_frames,)
    """

    y = np.asarray(y, dtype=np.float32)
    D = librosa.stft(
        y=y, n_fft=n_fft, hop_length=hop_length, window=window, center=center
    )
    S_mag = np.abs(D).astype(np.float32)
    freqs_hz = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32)
    times_s = librosa.frames_to_time(
        np.arange(S_mag.shape[1]), sr=sr, hop_length=hop_length
    )
    return S_mag, freqs_hz, times_s.astype(np.float32)


def isolate_frequency_band(
    S_mag: np.ndarray,
    freqs_hz: np.ndarray,
    fmin: float,
    fmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice spectrogram rows to keep only frequency bins within [fmin, fmax]."""

    freqs_hz = np.asarray(freqs_hz)
    idx = np.where((freqs_hz >= fmin) & (freqs_hz <= fmax))[0]
    if idx.size == 0:
        # Return empty band with correct time axis.
        return S_mag[:0, :], freqs_hz[:0]

    return S_mag[idx, :], freqs_hz[idx]


def bass_energy(S_bass: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Compute per-frame bass-band energy.

    We use squared magnitude summed across frequency bins.

    Returns:
        energy_t: shape (n_frames,)
    """

    if S_bass.size == 0:
        return np.zeros((S_bass.shape[1],), dtype=np.float32)
    energy_t = np.sum(S_bass**2, axis=0)
    return (energy_t + eps).astype(np.float32)


def bass_feature_vector(
    y: np.ndarray,
    sr: int,
    config: Optional[BassFeatureConfig] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Extract a fixed-length bass feature vector.

     The returned vector concatenates:

     1) Full-band (bass) energy stats over time (log energy):
         - mean, std
         - percentiles: 10, 50, 90
         - temporal centroid of energy (normalized)

     2) Temporal dynamics on the bass energy envelope:
         - mean(abs(diff(log_energy)))  (average frame-to-frame change)
         - std(diff(log_energy))        (variability of changes)

     3) Optional sub-band log-energy stats (default 3 sub-bands):
         For each sub-band we compute mean and std of log-energy.

     Vector length:
        - base = 8
        - + 2 * len(config.bass_subbands_hz)

    Args:
        y: Waveform (mono recommended).
        sr: Sample rate.
        config: BassFeatureConfig.

    Returns:
        (feature_vector, debug)

    feature_vector: shape (8 + 2 * n_subbands,)
        debug: intermediate arrays useful for plotting/inspection.
    """

    if config is None:
        config = BassFeatureConfig()

    S_mag, freqs_hz, times_s = compute_stft_magnitude(
        y=y,
        sr=sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        window=config.window,
        center=config.center,
    )

    S_bass, bass_freqs = isolate_frequency_band(
        S_mag=S_mag,
        freqs_hz=freqs_hz,
        fmin=config.bass_min_hz,
        fmax=config.bass_max_hz,
    )

    e_t = bass_energy(S_bass)
    log_e = np.log(e_t).astype(np.float32)

    # Full-band stats.
    mean = float(np.mean(log_e)) if log_e.size else 0.0
    std = float(np.std(log_e)) if log_e.size else 0.0
    p10, p50, p90 = _safe_percentiles(log_e, [10, 50, 90])
    centroid_norm = _temporal_centroid(e_t)

    # Simple temporal modulation descriptors.
    if log_e.size >= 2:
        dlog = np.diff(log_e)
        dlog_abs_mean = float(np.mean(np.abs(dlog)))
        dlog_std = float(np.std(dlog))
    else:
        dlog_abs_mean = 0.0
        dlog_std = 0.0

    feats: List[float] = [
        mean,
        std,
        p10,
        p50,
        p90,
        centroid_norm,
        dlog_abs_mean,
        dlog_std,
    ]

    # Sub-band features (mean/std per sub-band).
    subband_debug: List[np.ndarray] = []
    for fmin, fmax in config.bass_subbands_hz:
        S_sb, sb_freqs = isolate_frequency_band(
            S_mag=S_mag, freqs_hz=freqs_hz, fmin=fmin, fmax=fmax
        )
        e_sb = bass_energy(S_sb)
        log_e_sb = np.log(e_sb).astype(np.float32)
        feats.append(float(np.mean(log_e_sb)) if log_e_sb.size else 0.0)
        feats.append(float(np.std(log_e_sb)) if log_e_sb.size else 0.0)
        subband_debug.append(log_e_sb)

    feat = np.asarray(feats, dtype=np.float32)

    debug: Dict[str, np.ndarray] = {
        "S_mag": S_mag,
        "freqs_hz": freqs_hz,
        "times_s": times_s,
        "S_bass": S_bass,
        "bass_freqs_hz": bass_freqs,
        "bass_energy_t": e_t,
        "bass_log_energy_t": log_e,
        "bass_subband_log_energy_t": np.asarray(subband_debug, dtype=object),
    }

    return feat, debug
