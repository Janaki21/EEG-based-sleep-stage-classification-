# EEG Sleep Stage Classification using Interpretable Feature Learning

## Overview

This project presents a research-oriented pipeline for automatic sleep stage classification from single-channel EEG signals. The focus is on evaluating interpretable feature-based models and comparing them with a deep learning baseline under a subject-independent setting.

The work systematically analyzes statistical, spectral, and nonlinear complexity features, with emphasis on understanding their contribution to classification performance and generalization across subjects.

---

## Research Objective

The primary goal is to investigate:

- Whether nonlinear complexity features improve inter-subject generalization compared to traditional spectral features  
- Whether combining multiple feature families yields better performance than individual feature sets  
- Whether interpretable machine learning models can remain competitive with a simple deep learning baseline  

---

## Key Contributions

- Feature-based EEG sleep stage classification using:
  - Statistical features  
  - Spectral features  
  - Nonlinear complexity features  

- Comparative evaluation across feature families and their combinations  

- Subject-independent evaluation using Stratified Group K-Fold cross-validation  

- Repeated cross-validation for robust performance estimation  

- Statistical validation using:
  - Wilcoxon signed-rank test  
  - Effect size estimation  
  - Confidence intervals  

- Detailed error analysis including:
  - Confusion matrix analysis  
  - Misclassification patterns  
  - N1-stage difficulty analysis  

- Feature importance analysis:
  - Model-based importance  
  - Permutation importance  
  - Domain-level contribution  

- Deep learning baseline using a lightweight CNN for comparison  

- Automated generation of:
  - Paper-ready tables  
  - Publication-quality figures  

---

## Dataset

The pipeline is designed for publicly available EEG sleep datasets with:

- Single-channel EEG (e.g., Fpz-Cz)
- 30-second epochs
- Multi-class sleep staging (Wake, N1, N2, N3, REM)

Processed data and extracted features are stored locally in the `outputs/` directory.

---

## Project Structure

```text
sleep_eeg_project/
├── data/
├── outputs/
│   ├── archive_raw/
│   ├── figures/
│   │   ├── paper_main/
│   │   └── paper_reference/
│   ├── tables/
│   │   ├── paper_main/
│   │   └── paper_reference/
│   ├── models/
│   └── cache/
├── preprocess.py
├── features.py
├── train_eval.py
├── deep_baseline.py
├── error_analysis.py
├── dataset_report.py
├── plots.py
├── make_paper_tables.py
├── organize_outputs.py
├── run_phase4.py
├── run_phase5.py
├── config.py
├── utils.py
├── requirements.txt
└── README.md
