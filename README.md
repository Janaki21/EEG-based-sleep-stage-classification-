# EEG-Based Sleep Stage Classification using Neural Complexity Features
---

## Overview

This repository implements an end-to-end, subject-independent pipeline for automatic sleep stage classification from single-channel EEG signals. The core contribution is a systematic comparison of **statistical**, **spectral**, and **nonlinear complexity** feature families — and their combinations — evaluated under a rigorous inter-subject generalization protocol.

**Central finding:** Nonlinear complexity features (Kappa = 0.744) outperform spectral features alone (Kappa = 0.643), and their fusion with spectral features achieves the best overall performance (Macro F1 = 0.731, Kappa = 0.806).

---

## Research Questions

1. Do nonlinear complexity features improve inter-subject generalization over traditional spectral features?
2. Does combining multiple feature families produce better performance than any single family?
3. Can interpretable classical ML models remain competitive with deep learning baselines under strict subject-independent evaluation?

---

## Methodology

### Pipeline

```
Raw EEG (Sleep-EDF)
        │
        ▼
  Preprocessing
  (bandpass filter, z-score normalization)
        │
        ▼
  Segmentation → 30-second epochs (AASM standard)
        │
        ▼
  Feature Extraction
  ├── Statistical  (mean, variance, skewness, kurtosis, Hjorth params)
  ├── Spectral     (delta/theta/alpha/beta power, spectral entropy, SEF95)
  └── Complexity   (SampEn, ApEn, PE, HFD, LZC)
        │
        ▼
  Classification
  ├── Logistic Regression
  └── Random Forest
        │
        ▼
  Stratified Group K-Fold Evaluation
  (subject-level isolation, no data leakage)
```

### Dataset

**Sleep-EDF Database** (Cassette subset) — publicly available via PhysioNet.

| Property | Value |
|---|---|
| Channel used | Fpz-Cz (single-channel EEG) |
| Epoch length | 30 seconds |
| Sleep stages | Wake, N1, N2, N3, REM |
| Annotation standard | AASM |
| Access | [`download_data.py`](download_data.py) |

### Feature Set (28 total)

| Domain | Features |
|---|---|
| Statistical (8) | Mean, variance, skewness, kurtosis, Hjorth activity/mobility/complexity, zero-crossing rate |
| Spectral (10) | Delta/theta/alpha/beta absolute & relative power, spectral entropy, peak frequency, SEF95, spectral flatness |
| Complexity (10) | Sample Entropy, Approximate Entropy, Permutation Entropy, Higuchi Fractal Dimension, Lempel-Ziv Complexity (×2 variants each) |

### Validation Protocol

**Stratified Group K-Fold** cross-validation with subject as the grouping variable — epochs from a single subject appear only in training *or* testing, never both. This directly evaluates inter-subject generalization rather than intra-subject memorization.

---

## Results

### Classification Performance (Mean ± Std across folds)

| Feature Set | Model | Macro F1 | Cohen's Kappa |
|---|---|---|---|
| Statistical | LR | 0.591 ± 0.071 | 0.651 ± 0.141 |
| Spectral | LR | 0.637 ± 0.093 | 0.643 ± 0.161 |
| Complexity | RF | 0.635 ± 0.020 | **0.744 ± 0.010** |
| Stat + Complexity | LR | 0.698 ± 0.062 | 0.770 ± 0.090 |
| Spectral + Complexity | LR | 0.720 ± 0.050 | 0.796 ± 0.059 |
| **All Features** | **LR** | **0.731 ± 0.040** | **0.806 ± 0.056** |

> Complexity-only model achieves higher Kappa than spectral-only despite lower variance — indicating more consistent predictions across subjects.

### Per-Class F1 (Best Model — All Features, LR)

| Stage | F1 Score | Note |
|---|---|---|
| Wake | 0.95 | — |
| N1 | 0.34 | Known challenge; sparse and transitional |
| N2 | 0.92 | — |
| N3 | 0.74 | — |
| REM | 0.75 | — |

### Statistical Testing

Wilcoxon signed-rank tests were applied on fold-level metrics to compare feature set pairs. Performance gains from adding complexity features are directionally consistent across all folds. Formal significance is limited by fold count; effect sizes are reported in the paper.

---

## Visual Results

| Model Comparison | Confusion Matrix | Feature Importance |
|---|---|---|
| ![Model Comparison](assets/model_comparison_macro_f1.png) | ![Confusion Matrix](assets/best_model_confusion_matrix.png) | ![Feature Importance](assets/feature_importance.png) |

---

## Reproducing Results

### Requirements

```bash
Python >= 3.9
pip install -r requirements.txt
```

Key dependencies: `numpy`, `scipy`, `scikit-learn`, `mne`, `antropy`, `pandas`, `matplotlib`, `streamlit`

### Step-by-Step

```bash
# 1. Download Sleep-EDF dataset
python download_data.py

# 2. Preprocess and segment into epochs
python run_phase0_phase1.py

# 3. Extract all feature sets
python run_phase2.py

# 4. Train models and evaluate
python run_phase3.py

# 5. Generate figures and paper tables
python run_phase4.py
python run_phase5.py
```

All outputs (figures, tables, trained models) are written to `outputs/`.

### Interactive Demo

```bash
streamlit run app.py
```

Features: live EEG visualization, sleep stage prediction, per-feature contribution panel, confidence scores.

---

## Repository Structure

```
EEG-based-sleep-stage-classification/
├── data/                    # Raw and processed EEG data
├── outputs/
│   ├── figures/             # Publication-quality plots
│   ├── tables/              # Paper-ready result tables
│   └── models/              # Saved trained models
├── assets/                  # README figures
├── download_data.py         # PhysioNet Sleep-EDF downloader
├── preprocess.py            # Filtering, normalization, segmentation
├── features.py              # Full feature extraction pipeline
├── train_eval.py            # Model training and cross-validation
├── plots.py                 # Visualization routines
├── run_phase0_phase1.py     # Data setup + preprocessing
├── run_phase2.py            # Feature extraction
├── run_phase3.py            # Model training and evaluation
├── run_phase4.py            # Error analysis + ablation
├── run_phase5.py            # Paper figures and tables
├── app.py                   # Streamlit demo
├── config.py                # Global configuration
├── utils.py                 # Shared utilities
└── requirements.txt
```

---

## Limitations

- Single-channel EEG (Fpz-Cz) — no spatial cortical information
- No explicit temporal sequence modeling; sleep stage transitions not captured
- Classical ML baseline; deep learning comparison is a planned extension

---

## Future Work

- Multi-channel EEG integration for spatial feature capture
- Temporal modeling via GRU/attention layers for transition detection
- Validation on larger, multi-site cohorts (SHHS, MASS datasets)
- Deep learning baseline comparison (DeepSleepNet, ATTNSLEEP)

---

