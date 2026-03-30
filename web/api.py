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
    compute_stft_magnitude,
    isolate_frequency_band,
    load_audio,
    match_bass_patterns,
    normalize_waveform,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"mp3", "wav", "flac", "ogg", "m4a"}
MAX_FILE_SIZE_MB = 100


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


def downsample_array(arr: np.ndarray, target_size: int = 500) -> np.ndarray:
    """Downsample a 1D array by stride sampling."""
    if len(arr) <= target_size:
        return arr
    step = int(np.ceil(len(arr) / target_size))
    return arr[::step]


def process_song(
    filepath: str, sr: int = 22050, duration: float | None = None
) -> dict[str, Any] | None:
    """Load audio and extract bass features and spectrogram."""
    try:
        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            return {"success": False, "error": "Audio file not found"}

        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            return {
                "success": False,
                "error": f"File too large (max {MAX_FILE_SIZE_MB}MB)",
            }

        logger.info(f"Loading audio from {Path(filepath).name}")
        audio = load_audio(filepath, sr=sr, mono=True, duration=duration)
        logger.info(f"Audio loaded: {audio.y.shape[0]} samples")

        y = normalize_waveform(audio.y, method="peak")

        config = BassFeatureConfig()

        logger.info("Computing spectrogram...")
        S_mag, freqs_hz, times_s = compute_stft_magnitude(
            y=y,
            sr=audio.sr,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            window=config.window,
            center=config.center,
        )

        logger.info(f"Spectrogram shape: {S_mag.shape}")

        S_bass, bass_freqs = isolate_frequency_band(
            S_mag=S_mag,
            freqs_hz=freqs_hz,
            fmin=config.bass_min_hz,
            fmax=config.bass_max_hz,
        )

        return {
            "success": True,
            "filename": Path(filepath).name,
            "duration_seconds": float(len(y) / audio.sr),
            "times_s": times_s,
            "S_bass": S_bass,
            "bass_freqs": bass_freqs,
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
            if "song_a" not in request.files or "song_b" not in request.files:
                return jsonify({"error": "Missing song files"}), 400

            file_a = request.files["song_a"]
            file_b = request.files["song_b"]

            if (
                not file_a
                or not file_b
                or file_a.filename == ""
                or file_b.filename == ""
            ):
                return jsonify({"error": "Invalid file uploads"}), 400

            if not (allowed_file(file_a.filename) and allowed_file(file_b.filename)):
                return (
                    jsonify(
                        {
                            "error": "Unsupported file format. Use MP3, WAV, FLAC, OGG, or M4A"
                        }
                    ),
                    400,
                )

            file_a.seek(0, os.SEEK_END)
            file_b.seek(0, os.SEEK_END)
            size_a_mb = file_a.tell() / (1024 * 1024)
            size_b_mb = file_b.tell() / (1024 * 1024)
            file_a.seek(0)
            file_b.seek(0)

            if size_a_mb > MAX_FILE_SIZE_MB or size_b_mb > MAX_FILE_SIZE_MB:
                return jsonify(
                    {"error": f"Files too large (max {MAX_FILE_SIZE_MB}MB)"}
                ), 400

            filename_a = secure_filename(file_a.filename)
            filename_b = secure_filename(file_b.filename)
            unique_a = str(uuid.uuid4())[:8]
            unique_b = str(uuid.uuid4())[:8]

            path_a = os.path.join(
                tempfile.gettempdir(), f"song2vec_{unique_a}_{filename_a}"
            )
            path_b = os.path.join(
                tempfile.gettempdir(), f"song2vec_{unique_b}_{filename_b}"
            )

            logger.info("Saving uploaded files to temp...")
            file_a.save(path_a)
            file_b.save(path_b)

            logger.info("Processing song A...")
            result_a = process_song(path_a, sr=22050)
            if not result_a["success"]:
                return jsonify(
                    {"error": f"Failed to process Song 1: {result_a['error']}"}
                ), 400

            logger.info("Processing song B...")
            result_b = process_song(path_b, sr=22050)
            if not result_b["success"]:
                return jsonify(
                    {"error": f"Failed to process Song 2: {result_b['error']}"}
                ), 400

            logger.info("Matching bass patterns...")
            try:
                pattern_match = match_bass_patterns(
                    S_bass_a=result_a["S_bass"],
                    S_bass_b=result_b["S_bass"],
                    use_dtw=True,
                )
            except Exception as e:
                logger.error(f"Pattern matching failed: {str(e)}")
                return jsonify({"error": f"Failed to match patterns: {str(e)}"}), 500

            logger.info("Preparing response...")

            # Bass energy in dB for readable line plots
            energy_a = np.sum(result_a["S_bass"] ** 2, axis=0)
            energy_a_db = 10 * np.log10(np.maximum(energy_a, 1e-10))
            energy_b = np.sum(result_b["S_bass"] ** 2, axis=0)
            energy_b_db = 10 * np.log10(np.maximum(energy_b, 1e-10))

            target_frames = 500
            times_a_ds = downsample_array(
                result_a["times_s"], target_size=target_frames
            )
            times_b_ds = downsample_array(
                result_b["times_s"], target_size=target_frames
            )
            energy_a_ds = downsample_array(energy_a_db, target_size=target_frames)
            energy_b_ds = downsample_array(energy_b_db, target_size=target_frames)
            frame_sim_ds = downsample_array(
                pattern_match.frame_similarity, target_size=target_frames
            )

            frame_duration_a = result_a["hop_length"] / result_a["sr"]
            frame_duration_b = result_b["hop_length"] / result_b["sr"]

            matched_segments = [
                {
                    **seg,
                    "start_time_a": float(seg["start_frame"] * frame_duration_a),
                    "end_time_a": float(seg["end_frame"] * frame_duration_a),
                    "start_time_b": float(
                        seg.get("start_frame_b", 0) * frame_duration_b
                    ),
                    "end_time_b": float(seg.get("end_frame_b", 0) * frame_duration_b),
                }
                for seg in pattern_match.matched_segments
            ]

            response = {
                "song_a": {
                    "filename": result_a["filename"],
                    "duration_seconds": result_a["duration_seconds"],
                    "times_s": serialize_numpy(times_a_ds),
                    "bass_energy_db": serialize_numpy(energy_a_ds),
                },
                "song_b": {
                    "filename": result_b["filename"],
                    "duration_seconds": result_b["duration_seconds"],
                    "times_s": serialize_numpy(times_b_ds),
                    "bass_energy_db": serialize_numpy(energy_b_ds),
                },
                "similarity": {
                    "overall_similarity": pattern_match.overall_similarity,
                    "matched_segments": matched_segments,
                    "frame_similarity": serialize_numpy(frame_sim_ds),
                    "threshold": pattern_match.threshold,
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

            gc.collect()
