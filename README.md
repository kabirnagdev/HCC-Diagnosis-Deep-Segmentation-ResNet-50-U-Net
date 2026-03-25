# HCC Diagnosis via Deep Segmentation and Classification

Research-oriented repository for **Hepatocellular Carcinoma (HCC)** detection from CT scans using a segmentation-informed pipeline and a MobileNetV2-based classifier.

## Highlights

- CT slice classification into: `Background`, `Liver`, `Tumor`
- Medical-image preprocessing: HU windowing + CLAHE
- Inference utility for both single slice and full volume workflows
- Desktop GUI (`Tkinter`) for interactive scan analysis
- Saved model artifacts and metadata for reproducibility

## Project Status

This repository is intended for **research and educational use**. It is **not** a clinical-grade diagnostic system.

## Model Summary

From `models/model_info.json`:

- Architecture: **MobileNet-V2**
- Input size: **224 × 224**
- Classes: **3** (`Background`, `Liver`, `Tumor`)
- Test Accuracy: **95.31%**
- Test Sensitivity: **92.86%**
- Test Dice: **89.01%**

## Repository Structure

- `inference.py` — core inference pipeline and classifier wrapper
- `app.py` — desktop GUI app for CT/image analysis
- `example_usage.py` — scripted usage examples
- `quick_test.py` — lightweight smoke testing with local NIfTI scan
- `test_real_scan.py` — full/partial local scan testing
- `models/` — model artifacts (`.pth`, `model_info.json`)
- `MODEL_DEPLOYMENT_README.md` — deployment and integration notes
- `reference research paper/` — reference materials

## Quick Start

### 1) Create and activate a virtual environment

Use your preferred method (`venv`, `conda`, etc.).

### 2) Install dependencies

Install from `requirements.txt`.

### 3) Run inference (CLI)

Use `inference.py` with a `.nii` or `.nii.gz` volume and optional slice index.

### 4) Launch GUI

Run `app.py` and load a CT scan or image.

## Data Notes

- NIfTI files in this project are used for local testing and experimentation.
- Ensure data handling complies with institutional ethics/privacy requirements.

## Reproducibility

Please see:

- `REPRODUCIBILITY.md`
- `DATA.md`
- `CITATION.cff`

## Safety and Medical Disclaimer

This project is for **AI research prototyping only** and should not be used for medical decision-making without regulatory approval and expert clinical oversight.

## License

This repository is licensed under the **Apache License 2.0** (see `LICENSE`).

## Acknowledgment

Based on and inspired by the Kaggle notebook: **HCC Diagnosis Deep Segmentation: ResNet-50 U Net**.
