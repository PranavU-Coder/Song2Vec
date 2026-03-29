import numpy as np

from core.pattern_matching import match_bass_patterns


def _make_bass_spec(
    seed: int,
    n_freq: int = 16,
    n_frames: int = 180,
    cycles: float = 8.0,
    phase: float = 0.0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 0.05, size=(n_freq, n_frames)).astype(np.float32)

    t = np.linspace(0, cycles * np.pi, n_frames, dtype=np.float32) + phase
    pulse = (np.sin(t) + 1.2).astype(np.float32)
    shape = np.linspace(0.8, 1.2, n_freq, dtype=np.float32).reshape(n_freq, 1)
    return np.maximum(base + shape * pulse, 1e-6).astype(np.float32)


def test_identical_song_patterns_score_high():
    a = _make_bass_spec(seed=7)
    result = match_bass_patterns(a, a, use_dtw=True)

    assert 0.0 <= result.overall_similarity <= 1.0
    assert result.overall_similarity > 0.85
    assert result.frame_similarity.size == a.shape[1]


def test_related_patterns_score_higher_than_unrelated():
    a = _make_bass_spec(seed=11)
    b_related = (a * 0.98 + 0.02).astype(np.float32)
    b_unrelated = _make_bass_spec(seed=97, cycles=28.0, phase=np.pi / 3)
    b_unrelated[:, ::4] *= 0.05

    sim_related = match_bass_patterns(a, b_related, use_dtw=True).overall_similarity
    sim_unrelated = match_bass_patterns(a, b_unrelated, use_dtw=True).overall_similarity

    assert sim_related > sim_unrelated


def test_unrelated_songs_do_not_report_full_length_perfect_match():
    a = _make_bass_spec(seed=31, cycles=7.0, phase=0.0)
    b = _make_bass_spec(seed=91, cycles=29.0, phase=np.pi / 2)
    b[::2, :] *= 0.15

    result = match_bass_patterns(a, b, use_dtw=True)
    n = result.frame_similarity.size

    suspicious = [
        seg
        for seg in result.matched_segments
        if n > 0
        and seg["length_frames"] >= int(0.95 * n)
        and seg["mean_similarity"] >= 0.98
    ]

    assert len(suspicious) == 0


def test_unrelated_songs_do_not_saturate_frame_similarity():
    a = _make_bass_spec(seed=123, cycles=6.0, phase=0.0)
    b = _make_bass_spec(seed=987, cycles=31.0, phase=np.pi / 2)
    b[:, 1::3] *= 0.2

    result = match_bass_patterns(a, b, use_dtw=True)
    frame_sim = result.frame_similarity

    assert frame_sim.size > 0
    assert float(np.mean(frame_sim)) < 0.95
