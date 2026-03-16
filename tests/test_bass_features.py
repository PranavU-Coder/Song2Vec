import numpy as np

from features.bass_features import BassFeatureConfig, bass_feature_vector
from similarity.similarity_engine import cosine_similarity, similarity_score


def test_bass_feature_vector_is_fixed_length() -> None:
    sr = 22050
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 80 * t)).astype(np.float32)

    config = BassFeatureConfig()
    feat, debug = bass_feature_vector(y=y, sr=sr, config=config)

    expected_len = 8 + 2 * len(config.bass_subbands_hz)
    assert feat.shape == (expected_len,)
    assert "bass_log_energy_t" in debug


def test_similarity_scores_are_reasonable() -> None:
    a = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    assert cosine_similarity(a, b) == 1.0
    assert abs(cosine_similarity(a, c)) < 1e-6

    # Euclidean similarity mapping should be 1 for identical vectors.
    assert similarity_score(a, b, metric="euclidean") == 1.0