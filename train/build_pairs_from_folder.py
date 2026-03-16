"""Build a draft pair dataset from a folder of audio files (bass-only).

Goal:
  - Speed up creation of `data/pairs.csv` for training.

What it does:
  1) Scans an input folder (default: data/raw/) for audio files
  2) Extracts bass features for each track
  3) Suggests likely *positive* pairs using nearest neighbors in feature space
  4) Generates *negative* pairs by sampling tracks that are far apart
  5) Writes a draft CSV you can manually review/edit

Important:
  - This script does NOT know the true labels. It only proposes a starting point.
  - You must review and correct labels before training.

Usage:
  python train/build_pairs_from_folder.py --input-dir data/raw --output-csv data/pairs.csv

Notes:
  - MP3 decoding may require ffmpeg installed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity


# Allow running directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.bass_features import BassFeatureConfig, bass_feature_vector
from utils.audio_loader import load_audio, normalize_waveform


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass(frozen=True)
class PairBuildConfig:
    input_dir: str = "data/raw"
    output_csv: str = "data/pairs.csv"
    sample_rate: int = 22050
    normalize: str = "rms"  # none|peak|rms

    # Pair generation
    top_k_neighbors: int = 3
    positives_per_track: int = 2
    negatives_per_track: int = 2

    # Similarity thresholds
    min_pos_cosine: float = 0.90
    max_neg_cosine: float = 0.60

    # Feature extraction
    bass_config: BassFeatureConfig = BassFeatureConfig()


def find_audio_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return files


def extract_features(paths: Sequence[Path], cfg: PairBuildConfig) -> np.ndarray:
    feats: List[np.ndarray] = []
    for p in paths:
        audio = load_audio(p.as_posix(), sr=cfg.sample_rate, mono=True)
        y = audio.y
        if cfg.normalize != "none":
            y = normalize_waveform(y, method=cfg.normalize)
        feat, _ = bass_feature_vector(y=y, sr=audio.sr, config=cfg.bass_config)
        feats.append(feat)
    return np.stack(feats, axis=0).astype(np.float32)


def build_pairs(paths: Sequence[Path], feats: np.ndarray, cfg: PairBuildConfig) -> pd.DataFrame:
    """Return a draft labeled pairs dataframe."""

    # Cosine similarity matrix.
    sim = sk_cosine_similarity(feats)
    n = sim.shape[0]

    rows = []
    used = set()  # (i,j)

    def add_pair(i: int, j: int, label: int, reason: str, score: float) -> None:
        a, b = (i, j) if i < j else (j, i)
        if a == b:
            return
        key = (a, b)
        if key in used:
            return
        used.add(key)
        rows.append(
            {
                "path_a": paths[a].as_posix(),
                "path_b": paths[b].as_posix(),
                "label": int(label),
                "auto_reason": reason,
                "auto_cosine": float(score),
            }
        )

    rng = np.random.default_rng(42)

    for i in range(n):
        # Candidate neighbors (excluding self) sorted by similarity desc.
        order = np.argsort(-sim[i])
        order = [j for j in order if j != i]

        # Positives: choose top neighbors above threshold.
        pos_added = 0
        for j in order[: cfg.top_k_neighbors]:
            if sim[i, j] >= cfg.min_pos_cosine and pos_added < cfg.positives_per_track:
                add_pair(i, j, 1, "nearest_neighbor", sim[i, j])
                pos_added += 1

        # Negatives: sample from far tracks below threshold.
        far = np.where(sim[i] <= cfg.max_neg_cosine)[0]
        far = np.asarray([j for j in far if j != i], dtype=int)
        if far.size:
            picks = rng.choice(
                far,
                size=min(cfg.negatives_per_track, far.size),
                replace=False,
            )
            for j in picks:
                add_pair(i, int(j), 0, "far_sample", sim[i, int(j)])

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sort for readability: positives first, then highest similarity.
    df = df.sort_values(by=["label", "auto_cosine"], ascending=[False, False]).reset_index(drop=True)
    return df


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auto-build a draft bass-only pair dataset.")
    p.add_argument("--input-dir", type=str, default="data/raw")
    p.add_argument("--output-csv", type=str, default="data/pairs.csv")
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--normalize", choices=["none", "peak", "rms"], default="rms")

    p.add_argument("--top-k-neighbors", type=int, default=3)
    p.add_argument("--positives-per-track", type=int, default=2)
    p.add_argument("--negatives-per-track", type=int, default=2)
    p.add_argument("--min-pos-cosine", type=float, default=0.90)
    p.add_argument("--max-neg-cosine", type=float, default=0.60)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = PairBuildConfig(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        sample_rate=args.sr,
        normalize=args.normalize,
        top_k_neighbors=args.top_k_neighbors,
        positives_per_track=args.positives_per_track,
        negatives_per_track=args.negatives_per_track,
        min_pos_cosine=args.min_pos_cosine,
        max_neg_cosine=args.max_neg_cosine,
    )

    input_dir = Path(cfg.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    paths = find_audio_files(input_dir)
    if len(paths) < 3:
        raise ValueError("Need at least 3 audio files to generate pairs.")

    feats = extract_features(paths, cfg)
    df = build_pairs(paths, feats, cfg)

    out_path = Path(cfg.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Found {len(paths)} audio files in {input_dir}")
    print(f"Wrote {len(df)} draft pairs to {out_path}")
    print("\nNext:")
    print("  1) Open data/pairs.csv and fix labels (auto labels are only guesses)")
    print("  2) Remove columns auto_reason/auto_cosine if you want")
    print("  3) Train: python train/train_similarity_model.py")


if __name__ == "__main__":
    main()