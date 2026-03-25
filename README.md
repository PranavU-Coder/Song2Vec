# Song2Vec — Bass Pattern Similarity Detection

![graphs](/data/images/Song2Vec.png)

>> Join the Discord Server for discussion: https://discord.gg/AqMSZ3b3xM

**Bass pattern recognition and similarity detection** 

Compare musical similarity between two songs by analyzing how their bass patterns evolve over time.

## What This Project Does

**Song2Vec** is a **bass pattern recognition system** that detects if the bass sequences of two songs are similar, like Shazam fingerprinting.

### Simple Explanation
When you listen to music, you notice bass patterns—the low-frequency rhythm bumps. Song2Vec:
1. Extracts the bass from two songs
2. Compares if their bass patterns move in similar ways over time
3. Tells you how similar they are (as a percentage)
4. Shows you where the matches are using interactive graphs

## How It Works

### The Audio Processing Pipeline

**Step 1: Load & Normalize**
- Load audio file with librosa at 22050 Hz
- Normalize amplitude (peak or RMS normalization)

**Step 2: Extract Bass Spectrogram**
- Apply Short-Time Fourier Transform (STFT) with n_fft=4096, hop_length=512
- Isolate bass frequency band (20–250 Hz)
- Create spectrogram showing "how loud each frequency at each moment in time"
- Result: 43 frequency bins × N time frames

**Step 3: Compute Energy Envelope**
- Sum bass energy across all frequencies for each time frame
- Convert to log scale for perceptual alignment
- Get single curve: "bass energy level at each moment"

**Step 4: Temporal Pattern Matching** (Core Innovation)
The system compares sequences over time using three approaches:

1. **Dynamic Time Warping (DTW)**
   - Solves: "Two songs have same bass pattern but at different speeds"
   - Flexibly aligns sequences despite tempo differences
   - Returns DTW distance → converted to 0-100% similarity

2. **Cross-Correlation**
   - Finds repeating patterns and shifted versions
   - Shows where in the song patterns align
   - Peaks indicate strong alignment points

3. **Frame-by-Frame Similarity**
   - For each moment, compare bass levels: how similar is the energy?
   - Shows WHERE in the track patterns match (temporal localization)
   - Average across all frames = overall similarity

**Step 5: Visualize Results**
- Side-by-side spectrograms (what frequencies are loud when)
- Frame similarity graph (shows matching moments over time)
- Detected matched segments with timestamps and match percentages
- Overall similarity score (0-100%)

## Installation

Install dependencies:

```bash
uv sync
```

**Notes:**
- `librosa` requires `soundfile`/`audioread`. For MP3 support, install `ffmpeg`:
  - Ubuntu: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`

## Quick Start

### Web UI

```bash
bash run.sh
```

Then open http://localhost:5000 in your browser. Upload 2 songs via drag-drop to compare.

### Python API

```python
from core import (
    load_audio, normalize_waveform,
    compute_stft_magnitude, isolate_frequency_band,
    match_bass_patterns
)

# Load and process
audio_a = load_audio("song1.mp3", sr=22050)
audio_b = load_audio("song2.mp3", sr=22050)

y_a = normalize_waveform(audio_a.y, method="peak")
y_b = normalize_waveform(audio_b.y, method="peak")

# Get spectrograms
S_mag_a, freqs, times = compute_stft_magnitude(y_a, 22050, n_fft=4096, hop_length=512)
S_mag_b, _, _ = compute_stft_magnitude(y_b, 22050, n_fft=4096, hop_length=512)

# Extract bass band (20-250 Hz)
S_bass_a, bass_freqs = isolate_frequency_band(S_mag_a, freqs, 20, 250)
S_bass_b, _ = isolate_frequency_band(S_mag_b, freqs, 20, 250)

# Match patterns
result = match_bass_patterns(S_bass_a, S_bass_b, sr=22050, hop_length=512)

# Results
print(f"Similarity: {result.overall_similarity:.2%}")
print(f"Matched segments: {result.matched_segments}")
print(f"Frame scores: {result.frame_similarity}")
```

## Project Structure

```
Song2Vec/
├── app.py                ← Flask web app entry point
├── run.sh                ← Start web server
├── requirements.txt      ← Python dependencies
├── README.md
│
├── core/                 ← All audio processing logic
│   ├── __init__.py       (Public API exports)
│   ├── audio.py          (Load, normalize, resample)
│   ├── features.py       (STFT, bass extraction)
│   ├── pattern_matching.py (DTW, correlation, matching)
│   └── similarity.py     (Cosine, euclidean metrics)
│
├── web/                  ← Flask-specific code
│   ├── __init__.py
│   └── api.py            (REST endpoints)
│
├── templates/            ← HTML UI
│   └── index.html        (Plotly spectrograms, interface)
│
├── data/
│   ├── raw/             (Sample audio files)
│   ├── uploads/         (Temporary uploads)
│   └── images/          (Logo, documentation)
```

### Core Modules

**`core/audio.py`** — Audio loading and preprocessing
- `load_audio()` — Load audio with librosa
- `normalize_waveform()` — Peak/RMS normalization
- `resample_audio()` — Change sample rate

**`core/features.py`** — Bass feature extraction
- `compute_stft_magnitude()` — STFT spectrogram
- `isolate_frequency_band()` — Extract frequency range
- `bass_energy()` — Per-frame energy calculation

**`core/pattern_matching.py`** — Temporal pattern matching (core)
- `dtw_distance()` — Dynamic Time Warping alignment
- `cross_correlate_patterns()` — Pattern shift detection
- `frame_wise_similarity()` — Per-frame comparison
- `detect_pattern_matches()` — Find matching segments
- `match_bass_patterns()` — Main comparison function

**`core/similarity.py`** — Classical similarity metrics
- `cosine_similarity()` — Cosine distance
- `euclidean_distance()` — L2 distance

**`web/api.py`** — REST API endpoints
- `/api/compare` — File upload and comparison

## Understanding Results

### Example Results

**Score: 41.25% with matched segments:**
- Overall bass is somewhat different (41%)
- But there's a section that matches strongly (86%)
- Suggests songs share a bass phrase/pattern, but differ overall

### Visualizations Explained

- **Bass Spectrogram**: Heatmap showing frequency loudness over time (brighter = more energy)
- **Alignment Graph**: Shows which moments have similar bass. Shaded area highlights matched regions.
- **Matched Segments**: List of time ranges where bass patterns align strongly

## Design Decisions

**Why DTW instead of sliding window?**
- Bass patterns can stretch/compress (different tempos)
- DTW handles elastic alignment gracefully

**Why only bass (20–250 Hz)?**
- Most distinctive and robust for pattern recognition
- Less affected by high-frequency noise
- Can extend to other frequency bands later

**Why frame-by-frame similarity?**
- Shows WHERE patterns match (temporal localization)
- More informative than a single number
- Helps identify specific matching sections


## Performance Tips

**Faster processing:**
- Reduce sample rate: `sr=11025` (instead of 22050)
- Reduce duration: `duration=15` (instead of full track)
- Smaller FFT: `n_fft=2048` (instead of 4096)

**Better accuracy:**
- Increase sample rate: `sr=44100`
- Increase FFT: `n_fft=8192`
- Process full track duration

## References

- McFee et al., *librosa: Audio and music signal analysis in python* (SciPy 2015)
  - https://librosa.org/

- Bogdanov et al., **Essentia** (open-source MIR feature extraction)
  - https://essentia.upf.edu/

- Müller, *Fundamentals of Music Processing* (MIR reference textbook)

- Lerch, *An Introduction to Audio Content Analysis*
  - https://www.audiocontentanalysis.org/

- ISMIR community (music information retrieval conference)
  - https://ismir.net/

- Choi et al., *Automatic tagging using deep convolutional neural networks* (ISMIR 2016)
  - https://arxiv.org/abs/1606.00298

- Wyse, *Audio Spectrogram Representations for Processing with CNNs* (2017)
  - https://arxiv.org/abs/1706.09559
### Research Papers on Core Techniques


**Dynamic Time Warping (DTW)**

- Sakoe, H., & Chiba, S., *Dynamic programming algorithm optimization for spoken word recognition* (IEEE 1978)
  - https://doi.org/10.1109/TASSP.1978.1163055
  - Foundational DTW algorithm for pattern matching and temporal alignment

- Müller, M., & Kurth, F., *Enhancing beat tracking by dynamic programming* (IEEE TASLP 2006)
  - https://doi.org/10.1109/TASLP.2006.881688
  - DTW application to music beat tracking and tempo handling

- Salvador, S., & Chan, P., *FastDTW: Toward accurate dynamic time warping in linear time and space* (KDD 2004)
  - https://cs.fit.edu/~pkc/papers/tdm04.pdf
  - Efficient DTW for large-scale pattern matching

**Audio Fingerprinting & Music Similarity**

- Wang, A., *An industrial-strength audio search algorithm* (ISMIR 2003)
  - https://www.ismir.net/proceedings/2003/wang03.pdf
  - Shazam's robust audio fingerprinting approach

- Ellis, D. P., *Robust landmark-based audio fingerprinting* (ISMIR 2003)
  - https://www.ismir.net/proceedings/2003/ellis03fp.pdf
  - Spectrogram-based fingerprinting techniques

- Casey, M., Veltkamp, R., & Goto, M., *Content-based music information retrieval: Current directions and future challenges* (JACS 2008)
  - https://doi.org/10.1145/1344411.1344422
  - Comprehensive survey on music similarity and fingerprinting

**Spectrogram Analysis & Bass Detection**

- Müller, M., *Information Retrieval for Music and Motion* (Springer 2007)
  - https://www.springer.com/gp/book/9783540740841
  - Detailed treatment of STFT, spectrograms, and frequency band analysis

- Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S., *The Timbre Toolbox: extracting audio descriptors from musical signals* (JASA 2011)
  - https://doi.org/10.1121/1.3642604
  - Audio feature extraction from spectral analysis

**Cross-Correlation & Pattern Detection**

- Raffel, C., McFee, B., Humphrey, E. J., Salamon, J., Nieto, O., Liang, D., & Ellis, D. P. W., *mir_eval: A transparent implementation of common MIR metrics* (ISMIR 2014)
  - https://www.music-ir.org/mirex/abstracts/2014/mir_eval_presentation.pdf
  - Standardized evaluation of music similarity metrics

**Low-Frequency Analysis & Bass Patterns**

- Scheirer, E., & Slaney, M., *Construction and evaluation of a robust multifeature speech/music discriminator* (ICASSP 1997)
  - https://doi.org/10.1109/ICASSP.1997.599661
  - Audio classification using frequency-based features

- Ness, S. R., Theocharis, A., Tzanetakis, G., & Martins, L. G., *Improving automatic music tagging using deep learning* (IEEE TASLP 2016)
  - https://ieeexplore.ieee.org/document/7378814
  - Modern approaches to audio feature extraction and understanding

## License

[MIT](https://github.com/pranshu05/Song2Vec?tab=MIT-1-ov-file)