from __future__ import annotations

import argparse
from pathlib import Path

from features.bass_features import BassFeatureConfig, bass_feature_vector
from similarity.similarity_engine import similarity_score
from utils.audio_loader import load_audio, normalize_waveform


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare musical similarity between two songs.")
    parser.add_argument("song_a", type=str, help="Path to first audio file")
    parser.add_argument("song_b", type=str, help="Path to second audio file")
    parser.add_argument("--sr", type=int, default=22050, help="Target sample rate for loading")
    parser.add_argument(
        "--metric",
        type=str,
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Similarity metric",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        choices=["none", "peak", "rms"],
        default="peak",
        help="Waveform normalization to apply after loading",
    )
    parser.add_argument(
        "--dump-features",
        action="store_true",
        help="Print bass feature vectors (useful for debugging/experiments)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    a = load_audio(args.song_a, sr=args.sr, mono=True)
    b = load_audio(args.song_b, sr=args.sr, mono=True)

    if args.normalize != "none":
        y_a = normalize_waveform(a.y, method=args.normalize)
        y_b = normalize_waveform(b.y, method=args.normalize)
    else:
        y_a, y_b = a.y, b.y

    config = BassFeatureConfig()
    feat_a, _ = bass_feature_vector(y=y_a, sr=a.sr, config=config)
    feat_b, _ = bass_feature_vector(y=y_b, sr=b.sr, config=config)

    bass_sim = similarity_score(feat_a, feat_b, metric=args.metric)

    print("Song Comparison Report")
    print("----------------------")
    print(f"Song A: {Path(args.song_a).name}")
    print(f"Song B: {Path(args.song_b).name}")
    print(f"Feature length: {feat_a.size}")
    print(f"Bass similarity ({args.metric}): {bass_sim:.2f}")

    if args.dump_features:
        print("\nBass features A:")
        print(feat_a)
        print("\nBass features B:")
        print(feat_b)


if __name__ == "__main__":
    main()