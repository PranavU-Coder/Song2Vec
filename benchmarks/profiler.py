import sys
import os
import time
import numpy as np

# point to the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.pattern_matching import dtw_distance, frame_wise_similarity


def run_benchmark(iterations: int = 100, frames: int = 2000):
    np.random.seed(42)
    a = np.random.rand(frames).astype(np.float32)
    b = np.random.rand(frames).astype(np.float32)

    dtw_window = 50
    frame_window = 5

    print("Warming up JIT")
    # warming up both functions to clear any compilation/lazy-loading overhead
    dtw_distance(a[:10], b[:10], window=dtw_window)
    frame_wise_similarity(a[:10], b[:10], window_size=frame_window)

    # benchmark DTW
    dtw_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        dtw_distance(a, b, window=dtw_window)
        dtw_times.append(time.perf_counter() - start)

    avg_dtw = sum(dtw_times) / iterations
    min_dtw = min(dtw_times)
    print(f"DTW: {avg_dtw:.5f}s (avg) | {min_dtw:.5f}s (min)")

    # benchmark Frame Similarity
    frame_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        frame_wise_similarity(a, b, window_size=frame_window)
        frame_times.append(time.perf_counter() - start)

    avg_frame = sum(frame_times) / iterations
    min_frame = min(frame_times)
    print(f"Frame Sim: {avg_frame:.5f}s (avg)  |  {min_frame:.5f}s (min)")


if __name__ == "__main__":
    run_benchmark(iterations=100)
