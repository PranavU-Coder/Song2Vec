from __future__ import annotations

import gc
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from flask import jsonify, request
from werkzeug.utils import secure_filename

from core import (
    BassFeatureConfig,
    bass_feature_vector,
    compute_stft_magnitude,
    isolate_frequency_band,
    load_audio,
    match_bass_patterns,
    normalize_waveform,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"mp3", "wav", "flac", "ogg", "m4a"}
MAX_FILE_SIZE_MB = 100  # Maximum file size in MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def serialize_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: serialize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_numpy(item) for item in obj]
    return obj


def downsample_spectrogram(S: np.ndarray, target_frames: int = 500) -> np.ndarray:
    """Downsample spectrogram to reduce data size for visualization.

    Args:
        S: Spectrogram of shape (n_freq, n_frames)
        target_frames: Target number of frames (default 500 for performance)

    Returns:
        Downsampled spectrogram
    """
    n_freq, n_frames = S.shape
    if n_frames <= target_frames:
        return S

    # Calculate step size to achieve target frame count
    step = int(np.ceil(n_frames / target_frames))
    return S[:, ::step]


def downsample_array(arr: np.ndarray, target_size: int = 500) -> np.ndarray:
    """Downsample 1D array using max pooling."""
    if len(arr) <= target_size:
        return arr

    step = int(np.ceil(len(arr) / target_size))
    return arr[::step]


def process_song(filepath: str, sr: int = 22050, duration: int = None) -> dict[str, Any] | None:
    """Load audio and extract bass features and spectrogram."""
    try:
        # Validate file exists and is readable
        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            return {"success": False, "error": "Audio file not found"}

        # Check file size
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            return {"success": False, "error": f"File too large (max {MAX_FILE_SIZE_MB}MB)"}

        logger.info(f"Loading audio from {Path(filepath).name}")
        audio = load_audio(filepath, sr=sr, mono=True, duration=duration)
        logger.info(f"Audio loaded: {audio.y.shape[0]} samples")

        y = normalize_waveform(audio.y, method="peak")

        config = BassFeatureConfig()

        logger.info("Computing spectrogram...")
        # Get the full spectrogram and metadata for visualization
        S_mag, freqs_hz, times_s = compute_stft_magnitude(
            y=y,
            sr=audio.sr,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            window=config.window,
            center=config.center,
        )

        logger.info(f"Spectrogram shape: {S_mag.shape}")

        # Isolate bass band
        S_bass, bass_freqs = isolate_frequency_band(
            S_mag=S_mag,
            freqs_hz=freqs_hz,
            fmin=config.bass_min_hz,
            fmax=config.bass_max_hz,
        )

        # Get feature vector
        feat_vec, _ = bass_feature_vector(y=y, sr=audio.sr, config=config)

        return {
            "success": True,
            "filename": Path(filepath).name,
            "duration_seconds": float(len(y) / audio.sr),
            "S_mag": S_mag,
            "freqs_hz": freqs_hz,
            "times_s": times_s,
            "S_bass": S_bass,
            "bass_freqs": bass_freqs,
            "feature_vector": feat_vec,
            "sr": audio.sr,
            "hop_length": config.hop_length,
        }
    except MemoryError:
        logger.error("Out of memory while processing audio")
        return {"success": False, "error": "Audio file too large - out of memory"}
    except Exception as e:
        logger.error(f"Error processing song: {str(e)}")
        return {"success": False, "error": f"Error processing audio: {str(e)}"}


def register_routes(app) -> None:
    """Register all API routes to Flask app."""

    @app.route("/api/compare", methods=["POST"])
    def compare_songs() -> dict[str, Any]:
        """Compare two uploaded songs using bass pattern matching."""
        path_a = None
        path_b = None

        try:
            # Check for file uploads
            if "song_a" not in request.files or "song_b" not in request.files:
                return jsonify({"error": "Missing song files"}), 400

            file_a = request.files["song_a"]
            file_b = request.files["song_b"]

            if not file_a or not file_b or file_a.filename == "" or file_b.filename == "":
                return jsonify({"error": "Invalid file uploads"}), 400

            if not (allowed_file(file_a.filename) and allowed_file(file_b.filename)):
                return (
                    jsonify({"error": "Unsupported file format. Use MP3, WAV, FLAC, OGG, or M4A"}),
                    400,
                )

            # Check file sizes before processing
            file_a.seek(0, os.SEEK_END)
            file_b.seek(0, os.SEEK_END)
            size_a_mb = file_a.tell() / (1024 * 1024)
            size_b_mb = file_b.tell() / (1024 * 1024)
            file_a.seek(0)
            file_b.seek(0)

            if size_a_mb > MAX_FILE_SIZE_MB or size_b_mb > MAX_FILE_SIZE_MB:
                return jsonify({"error": f"Files too large (max {MAX_FILE_SIZE_MB}MB)"}), 400

            # Save temporarily with unique names
            filename_a = secure_filename(file_a.filename)
            filename_b = secure_filename(file_b.filename)
            unique_a = str(uuid.uuid4())[:8]
            unique_b = str(uuid.uuid4())[:8]

            path_a = os.path.join(tempfile.gettempdir(), f"song2vec_{unique_a}_{filename_a}")
            path_b = os.path.join(tempfile.gettempdir(), f"song2vec_{unique_b}_{filename_b}")

            logger.info(f"Saving uploaded files to temp...")
            file_a.save(path_a)
            file_b.save(path_b)

            # Process both songs
            logger.info("Processing song A...")
            result_a = process_song(path_a, sr=22050)
            if not result_a["success"]:
                return jsonify({"error": f"Failed to process Song 1: {result_a['error']}"}), 400

            logger.info("Processing song B...")
            result_b = process_song(path_b, sr=22050)
            if not result_b["success"]:
                return jsonify({"error": f"Failed to process Song 2: {result_b['error']}"}), 400

            # Match bass patterns
            logger.info("Matching bass patterns...")
            try:
                pattern_match = match_bass_patterns(
                    S_bass_a=result_a["S_bass"],
                    S_bass_b=result_b["S_bass"],
                    sr=result_a["sr"],
                    hop_length=result_a["hop_length"],
                    use_dtw=True,
                )
            except Exception as e:
                logger.error(f"Pattern matching failed: {str(e)}")
                return jsonify({"error": f"Failed to match patterns: {str(e)}"}), 500

            # Prepare response with visualization data
            logger.info("Preparing response...")

            # Downsample spectrograms for browser performance
            S_bass_a_ds = downsample_spectrogram(result_a["S_bass"], target_frames=500)
            S_bass_b_ds = downsample_spectrogram(result_b["S_bass"], target_frames=500)
            times_a_ds = downsample_array(result_a["times_s"], target_size=500)
            times_b_ds = downsample_array(result_b["times_s"], target_size=500)
            frame_sim_ds = downsample_array(pattern_match.frame_similarity, target_size=500)

            response = {
                "song_a": {
                    "filename": result_a["filename"],
                    "duration_seconds": result_a["duration_seconds"],
                    "freqs_hz": serialize_numpy(result_a["bass_freqs"]),
                    "times_s": serialize_numpy(times_a_ds),
                    "S_bass_db": serialize_numpy(
                        20 * np.log10(np.maximum(S_bass_a_ds, 1e-10))
                    ),
                    "bass_freqs": serialize_numpy(result_a["bass_freqs"]),
                },
                "song_b": {
                    "filename": result_b["filename"],
                    "duration_seconds": result_b["duration_seconds"],
                    "freqs_hz": serialize_numpy(result_b["bass_freqs"]),
                    "times_s": serialize_numpy(times_b_ds),
                    "S_bass_db": serialize_numpy(
                        20 * np.log10(np.maximum(S_bass_b_ds, 1e-10))
                    ),
                    "bass_freqs": serialize_numpy(result_b["bass_freqs"]),
                },
                "similarity": {
                    "overall_similarity": pattern_match.overall_similarity,
                    "matched_segments": pattern_match.matched_segments,
                    "frame_similarity": serialize_numpy(frame_sim_ds),
                },
            }

            logger.info("Comparison complete, cleaning up...")
            return jsonify(response)

        except MemoryError:
            logger.error("Out of memory")
            return jsonify({"error": "Insufficient memory - try smaller files"}), 500
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return jsonify({"error": f"Server error: {str(e)[:100]}"}), 500
        finally:
            # Cleanup temporary files
            try:
                if path_a and os.path.exists(path_a):
                    os.remove(path_a)
                    logger.info("Cleaned up temp file A")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file A: {e}")

            try:
                if path_b and os.path.exists(path_b):
                    os.remove(path_b)
                    logger.info("Cleaned up temp file B")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file B: {e}")

            # Force garbage collection
            gc.collect()