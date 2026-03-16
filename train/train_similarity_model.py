"""Train a minimal similarity model (bass-only, working today).

This script turns the current "bass feature vector" baseline into something you can
**train now** (without PyTorch) using a simple supervised model.

Why not a Siamese net yet?
    - Siamese / metric learning is the long-term direction, but it needs more data,
        more plumbing, and PyTorch.
    - A classical model gives you an end-to-end training loop immediately and helps
        validate your dataset + labels.

What this script does:
    1) Read a pair dataset CSV (see `train/pairs_schema.md`)
    2) Extract bass features for each unique audio path (with caching)
    3) Build pairwise features from two songs' bass vectors
    4) Train a classifier (logistic regression) that predicts if two songs are similar
    5) Save the trained pipeline to disk (joblib)

The trained model outputs `p(similar)` which can act as a learned similarity score.

Later upgrade path:
    - Replace the classifier with a Siamese network and contrastive/triplet loss.
    - Keep the same dataset format and feature caching.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Allow running this file directly: `python train/train_similarity_model.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.bass_features import BassFeatureConfig, bass_feature_vector
from utils.audio_loader import load_audio, normalize_waveform


@dataclass
class TrainingConfig:
    """Configuration for bass-only training."""

    # Data
    pairs_csv: str = "data/pairs.csv"
    sample_rate: int = 22050
    normalize: str = "rms"  # "none" | "peak" | "rms"

    # Feature extraction
    bass_config: BassFeatureConfig = BassFeatureConfig()

    # Model/training
    test_size: float = 0.2
    random_state: int = 42
    max_iter: int = 2000

    # Outputs
    out_dir: str = "models"
    model_name: str = "bass_similarity_logreg.joblib"

    seed: int = 42
    device: Optional[str] = None  # reserved for PyTorch later


def _pairwise_features(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    """Build a symmetric pairwise feature vector from two song vectors.

    Common trick in similarity learning with fixed vectors:
      - absolute difference captures distance-like info
      - elementwise product captures alignment/correlation-like info
    """

    a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
    b = np.asarray(vec_b, dtype=np.float32).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"Vector size mismatch: {a.size} vs {b.size}")
    return np.concatenate([np.abs(a - b), a * b], axis=0).astype(np.float32)


def load_pairs_csv(pairs_csv: str) -> pd.DataFrame:
    """Load and validate pair CSV."""

    path = Path(pairs_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"Pairs CSV not found: {pairs_csv}. Create it using train/pairs_schema.md"
        )

    df = pd.read_csv(path)
    required = {"path_a", "path_b", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in pairs CSV: {sorted(missing)}")

    df = df.copy()
    df["label"] = df["label"].astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("label must be 0/1")

    return df


def compute_bass_feature_cache(
    paths: Iterable[str],
    sample_rate: int,
    normalize: str,
    bass_cfg: BassFeatureConfig,
) -> Dict[str, np.ndarray]:
    """Compute bass features for unique paths (in-memory cache)."""

    cache: Dict[str, np.ndarray] = {}
    for p in sorted(set(paths)):
        audio = load_audio(p, sr=sample_rate, mono=True)
        y = audio.y
        if normalize != "none":
            y = normalize_waveform(y, method=normalize)
        feat, _ = bass_feature_vector(y=y, sr=audio.sr, config=bass_cfg)
        cache[p] = feat
    return cache


def build_pair_dataset(df: pd.DataFrame, feat_cache: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Turn (path_a, path_b, label) rows into (X, y)."""

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for row in df.itertuples(index=False):
        a = feat_cache[str(row.path_a)]
        b = feat_cache[str(row.path_b)]
        X_list.append(_pairwise_features(a, b))
        y_list.append(int(row.label))

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y


def train_bass_similarity_model(config: TrainingConfig) -> Pipeline:
    """Train a simple classifier that predicts if two songs are similar."""

    df = load_pairs_csv(config.pairs_csv)
    all_paths = list(df["path_a"].astype(str).values) + list(df["path_b"].astype(str).values)

    feat_cache = compute_bass_feature_cache(
        paths=all_paths,
        sample_rate=config.sample_rate,
        normalize=config.normalize,
        bass_cfg=config.bass_config,
    )

    X, y = build_pair_dataset(df, feat_cache)

    # Stratified splitting requires at least 2 samples per class.
    unique, counts = np.unique(y, return_counts=True)
    can_stratify = (unique.size > 1) and bool(np.all(counts >= 2))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y if can_stratify else None,
    )

    if np.unique(y_train).size < 2:
        # With tiny datasets it's easy for the train split to contain only one class.
        # In that case, fall back to training on all data.
        print(
            "Warning: train split contains a single class. "
            "Train/test split disabled; training on all pairs. "
            "Add more labeled pairs for meaningful evaluation."
        )
        X_train, y_train = X, y
        X_test, y_test = X, y

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=config.max_iter,
                    class_weight="balanced",
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nEvaluation (holdout):")
    print(classification_report(y_test, y_pred, digits=3))

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / config.model_name
    dump(
        {
            "model": model,
            "bass_feature_config": config.bass_config,
            "pairwise_feature": "concat(abs(a-b), a*b)",
            "sample_rate": config.sample_rate,
            "normalize": config.normalize,
        },
        out_path,
    )
    print(f"Saved model to: {out_path}")
    return model


def generate_embeddings_placeholder() -> None:
    """Reserved for the future Siamese embedding generator.

    Today we train on engineered pairwise features.
    """

    print(
        "Not used in the bass-only classical baseline. "
        "When switching to PyTorch, implement embedding generation here."
    )


def train_siamese_network_placeholder() -> None:
    """Reserved for future PyTorch Siamese training."""

    print(
        "Not used in the bass-only classical baseline. "
        "When switching to PyTorch, implement Siamese training here."
    )


def main() -> None:
    """Train and save a bass-only similarity model from labeled pairs."""

    config = TrainingConfig()
    train_bass_similarity_model(config)


if __name__ == "__main__":
    main()