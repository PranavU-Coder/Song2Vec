# Song2Vec

![alt text](/data/images/Song2Vec.png)

Compare musical similarity between two songs using audio signal processing and machine learning.

## Current focus: bass similarity (working baseline)

This repo currently focuses on **bass-only similarity** as a solid, explainable baseline.

Pipeline:
1. Load audio (`librosa`)
2. Compute STFT magnitude
3. Keep bass band (default **20–250 Hz**)
4. Build a fixed-length bass feature vector:
   - bass-band log-energy statistics
   - temporal dynamics of the bass energy envelope
   - optional bass **sub-band** stats (default: 20–60, 60–120, 120–250 Hz)
5. Compare features with cosine similarity (or euclidean distance mapped to (0, 1])

## Repository layout

- `data/`: put your audio pairs/datasets here (gitignored if you want)
- `features/`: feature extractors (bass/rhythm/melody/timbre)
- `similarity/`: similarity metrics and scoring
- `visualization/`: plotting helpers
- `utils/`: I/O and shared utilities
- `train/`: model training scaffolds (Siamese network later)
- `compare.py`: CLI entrypoint

## Install

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Notes:
- `librosa` uses `soundfile`/`audioread` under the hood. For MP3 support you may need `ffmpeg` installed on your system.

## Quick start (bass similarity)

```bash
python compare.py path/to/song1.wav path/to/song2.wav
```

Optional arguments:
- `--sr 22050` (set target sampling rate)
- `--metric cosine|euclidean`
- `--normalize none|peak|rms`

Example output:

```
Song Comparison Report
----------------------
Song A: song1.wav
Song B: song2.wav
Bass similarity (cosine): 0.81
```

## What “bass feature vector” means

The bass feature extractor is implemented in `features/bass_features.py`.

The output is a fixed-length vector computed from low-frequency energy statistics and simple temporal dynamics within the 20–250 Hz band.

## References / inspiration (non-exhaustive)

- McFee et al., *librosa: Audio and music signal analysis in python* (SciPy 2015)
   - Docs: https://librosa.org/
   - Paper: https://doi.org/10.25080/Majora-7b98e3ed-003

- Bogdanov et al., **Essentia** (open-source MIR feature extraction, C++/Python)
   - https://essentia.upf.edu/
   - https://github.com/MTG/essentia

- Müller, *Fundamentals of Music Processing* (standard MIR reference / textbook)

- Lerch, *An Introduction to Audio Content Analysis* + open-source reference implementations
   - https://www.audiocontentanalysis.org/
   - Python code: https://github.com/alexanderlerch/pyACA

- ISMIR community (conference + open-access journal TISMIR)
   - https://ismir.net/
   - TISMIR: https://transactions.ismir.net/

- Choi et al., *Automatic tagging using deep convolutional neural networks* (ISMIR 2016)
   - (useful background for learned audio embeddings and spectrogram baselines)
   - https://arxiv.org/abs/1606.00298

- Wyse, *Audio Spectrogram Representations for Processing with CNNs* (2017)
   - (high-level overview of representation choices)
   - https://arxiv.org/abs/1706.09559

- CREPE pitch tracker (useful later if you extend from bass energy → bassline pitch/melody)
   - https://github.com/marl/crepe

## License

MIT.
