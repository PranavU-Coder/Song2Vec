import sys
import os
import time
import numpy as np

# point to the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.pattern_matching import dtw_distance, frame_wise_similarity


def run_benchmark():
    a = np.random.rand(2000).astype(np.float32)
    b = np.random.rand(2000).astype(np.float32)

    print("Warming up JIT")
    dtw_distance(a[:10], b[:10], window=50)

    # 1. Benchmark DTW
    start = time.perf_counter()
    dtw_distance(a, b, window=50)
    dtw_time = time.perf_counter() - start
    print(f"DTW: {dtw_time:.5f} seconds")

    # 2. Benchmark Frame Similarity
    start = time.perf_counter()
    frame_wise_similarity(a, b, window_size=5)
    frame_time = time.perf_counter() - start
    print(f"Frame Sim Execution: {frame_time:.5f} seconds")


if __name__ == "__main__":
    run_benchmark()
