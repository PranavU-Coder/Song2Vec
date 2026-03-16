"""Spectrogram plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


PathLike = Union[str, Path]


def plot_spectrogram_with_bass_band(
    S_mag: np.ndarray,
    sr: int,
    hop_length: int,
    n_fft: int,
    bass_band_hz: Tuple[float, float] = (20.0, 250.0),
    title: str = "Spectrogram (dB)",
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> None:
    """Plot a magnitude spectrogram in dB and highlight the bass frequency band."""

    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=ax,
    )
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    fmin, fmax = bass_band_hz
    ax.axhspan(fmin, fmax, color="yellow", alpha=0.15, label=f"Bass {fmin:.0f}-{fmax:.0f} Hz")
    ax.legend(loc="upper right")

    fig.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p.as_posix(), dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)