from .audio import AudioData, load_audio, normalize_waveform, resample_audio
from .features import BassFeatureConfig, bass_feature_vector, compute_stft_magnitude, isolate_frequency_band
from .pattern_matching import (
    PatternMatch,
    cross_correlate_patterns,
    detect_pattern_matches,
    dtw_distance,
    frame_wise_similarity,
    match_bass_patterns,
)
from .similarity import cosine_similarity, euclidean_distance, similarity_score

__all__ = [
    # Audio
    "AudioData",
    "load_audio",
    "normalize_waveform",
    "resample_audio",
    # Features
    "BassFeatureConfig",
    "bass_feature_vector",
    "compute_stft_magnitude",
    "isolate_frequency_band",
    # Pattern Matching
    "PatternMatch",
    "match_bass_patterns",
    "cross_correlate_patterns",
    "frame_wise_similarity",
    "dtw_distance",
    "detect_pattern_matches",
    # Similarity
    "cosine_similarity",
    "euclidean_distance",
    "similarity_score",
]
