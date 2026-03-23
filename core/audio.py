from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import numpy as np


PathLike = Union[str, Path]


@dataclass(frozen=True)
class AudioData:
    y: np.ndarray
    sr: int
    path: Optional[Path] = None


def load_audio(
    path: PathLike,
    sr: Optional[int] = 22050,
    mono: bool = True,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> AudioData:
    """Load an audio file using librosa.

    Args:
        path: Path to an audio file supported by librosa/audioread.
        sr: Target sampling rate. Use None to preserve the original sample rate.
        mono: If True, mix down to mono.
        offset: Start reading after this time (in seconds).
        duration: Only load up to this much audio (in seconds).

    Returns:
        AudioData with waveform `y` (float32) and sample rate `sr`.
    """

    p = Path(path)
    y, loaded_sr = librosa.load(
        p.as_posix(), sr=sr, mono=mono, offset=offset, duration=duration
    )
    # librosa returns float32 in [-1, 1] typically, but don't assume perfect bounds.
    y = np.asarray(y, dtype=np.float32)
    return AudioData(y=y, sr=int(loaded_sr), path=p)


def resample_audio(y: np.ndarray, orig_sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    """Resample audio to a new sample rate.

    Args:
        y: Waveform.
        orig_sr: Original sample rate.
        target_sr: Desired sample rate.

    Returns:
        (y_resampled, target_sr)
    """

    if orig_sr == target_sr:
        return np.asarray(y, dtype=np.float32), int(orig_sr)

    y_rs = librosa.resample(y=np.asarray(y, dtype=np.float32), orig_sr=orig_sr, target_sr=target_sr)
    return np.asarray(y_rs, dtype=np.float32), int(target_sr)


def normalize_waveform(y: np.ndarray, method: str = "peak", eps: float = 1e-8) -> np.ndarray:
    """Normalize a waveform.

    Supported methods:
      - "peak": scale by max(|y|) so peak amplitude becomes 1.0
      - "rms": scale by RMS so RMS becomes 1.0

    Args:
        y: Waveform.
        method: Normalization method.
        eps: Numerical stability constant.

    Returns:
        Normalized waveform (float32).

    Raises:
        ValueError: if method is unknown.
    """

    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return y

    if method == "peak":
        denom = float(np.max(np.abs(y)))
    elif method == "rms":
        denom = float(np.sqrt(np.mean(y**2)))
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")

    if denom < eps:
        return y

    return (y / denom).astype(np.float32)