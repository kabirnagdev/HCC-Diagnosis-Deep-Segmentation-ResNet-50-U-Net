# Reproducibility Notes

## Environment

- Python 3.10+ recommended
- Dependencies listed in `requirements.txt`

## Determinism

For repeatable experiments:

1. Fix random seeds for `numpy` and `torch`.
2. Record exact package versions.
3. Log hardware details (GPU/CPU, VRAM, RAM).
4. Save all preprocessing hyperparameters.

## Model Configuration Snapshot

Current default values from `models/model_info.json`:

- HU window: `[-75, 150]`
- CLAHE clip limit: `0.01`
- Input resolution: `224`
- Classes: `Background`, `Liver`, `Tumor`

## Reporting Results

When publishing results, include:

- Dataset split strategy
- Number of slices/volumes per class
- Per-class metrics and confidence intervals
- External validation constraints

## Non-clinical Use

All outputs are for research benchmarking and experimentation only.
