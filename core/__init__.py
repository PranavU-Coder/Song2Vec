from .audio import AudioData, load_audio, normalize_waveform, resample_audio
from .features import (
    BassFeatureConfig,
    bass_feature_vector,
    compute_stft_magnitude,
    isolate_frequency_band,
)
from .pattern_matching import (
    PatternMatch,
    compute_bass_spectrogram_features,
    cross_correlate_patterns,
    detect_pattern_matches,
    dtw_distance,
    frame_wise_similarity,
    match_bass_patterns,
)
from .similarity import cosine_similarity, euclidean_distance, similarity_score
from .dtw import (
    StructuralMatch,
    StructuralSegment,
    batch_compare_structures,
    compare_song_structures,
    fast_dtw,
    get_keogh_envelope,
    lb_keogh,
)

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
    # Pattern matching
    "PatternMatch",
    "compute_bass_spectrogram_features",
    "cross_correlate_patterns",
    "detect_pattern_matches",
    "dtw_distance",
    "frame_wise_similarity",
    "match_bass_patterns",
    # Similarity
    "cosine_similarity",
    "euclidean_distance",
    "similarity_score",
    # Structural DTW
    "StructuralMatch",
    "StructuralSegment",
    "batch_compare_structures",
    "compare_song_structures",
    "fast_dtw",
    "get_keogh_envelope",
    "lb_keogh",
]
