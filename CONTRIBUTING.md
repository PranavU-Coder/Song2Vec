# Contributing

Thanks for contributing!

## Dev setup

- Use Python 3.10+
- Create a venv and install deps from `requirements.txt`.

## Quality gates

Before opening a PR:
- Run unit tests: `python -m pytest`
- Keep functions small and documented (docstrings).

## Project conventions

- Feature extraction code lives in `features/`
- Similarity metrics live in `similarity/`
- Training scripts live in `train/`
- Avoid committing audio files; store them in `data/` (gitignored)

## Reporting issues

When reporting an issue, include:
- OS + Python version
- The command you ran
- A minimal reproduction (a short audio clip if it is your own/right-to-share)