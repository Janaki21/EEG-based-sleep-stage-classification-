# 🧠 EEG-Based Sleep Stage Classification using Neural Complexity

## 📌 Overview

This project builds an end-to-end system to classify brain states from EEG signals using machine learning.

Unlike traditional approaches that rely only on frequency-based features, this work introduces **neural complexity measures** (entropy, fractals) to capture deeper structure in brain activity.

👉 **Key Insight:**
Brain states are not defined only by frequency patterns — they also differ in signal complexity.

---

## 🎯 Problem Statement

Most EEG-based systems focus on:

* Spectral features (alpha, beta, delta waves)

However, this ignores an important question:

> Can the *complexity* of brain signals improve classification?

---

## ⚙️ Methodology

### 🔄 Pipeline

```
Raw EEG Signal
      ↓
Segmentation (30s epochs)
      ↓
Feature Extraction
  ├── Statistical Features
  ├── Spectral Features
  └── Complexity Features ⭐
      ↓
Machine Learning Models
      ↓
Sleep Stage Prediction
```

---

## 🧠 Feature Engineering

### 1. Statistical Features

* Mean, variance, skewness
* Hjorth parameters

### 2. Spectral Features

* Delta, Theta, Alpha, Beta power
* Spectral entropy
* Peak frequency

### 3. Complexity Features (Core Contribution)

* Sample entropy
* Approximate entropy
* Permutation entropy
* Higuchi fractal dimension
* Lempel-Ziv complexity

👉 These capture **structure, irregularity, and information richness** of EEG signals.

---

## 🤖 Models Used

* Logistic Regression
* Random Forest

Evaluation:

* Stratified Group K-Fold (no subject leakage)
* Metrics:

  * Macro F1 Score
  * Balanced Accuracy
  * Cohen’s Kappa

---

## 📊 Results

### 🔥 Key Finding

| Feature Set           | Macro F1 | Kappa    |
| --------------------- | -------- | -------- |
| Statistical           | 0.59     | 0.65     |
| Spectral              | 0.64     | 0.64     |
| Complexity            | 0.63     | 0.74     |
| Stat + Complexity     | 0.70     | 0.77     |
| Spectral + Complexity | 0.72     | 0.80     |
| **All Features**      | **0.73** | **0.81** |

👉 Adding complexity features significantly improves performance.

---

## 📈 Visual Results

### Model Performance

![Model Comparison](outputs/figures/model_comparison_macro_f1.png)

### Confusion Matrix

![Confusion Matrix](outputs/figures/best_model_confusion_matrix.png)

### Class Distribution

![Distribution](outputs/figures/class_distribution.png)

### Feature Importance

![Feature Importance](outputs/figures/feature_importance.png)

---

## 🎬 Interactive Demo

This project includes a Streamlit UI for real-time demonstration.

### Run the demo:

```bash
streamlit run app.py
```

### Features:

* EEG signal visualization
* Brain state prediction
* Confidence scores
* Explanation panel (feature insights)

---

## 🧠 Example Demo Output

```
Input: EEG Signal Segment
Prediction: N2 Sleep
Confidence: 0.82

Top Features:
- spectral_entropy
- sample_entropy
- delta_power
```

---

## 💡 Key Contributions

* Introduced **complexity-based EEG representation**
* Demonstrated performance gain over traditional features
* Built **end-to-end pipeline + interactive UI**
* Ensured **rigorous validation (no data leakage)**

---

## 📦 Project Structure

```
sleep_eeg_project/
│
├── app.py                # Streamlit UI
├── preprocess.py        # EEG preprocessing
├── features.py          # Feature extraction
├── train_eval.py        # Model training
├── plots.py             # Visualization
├── config.py            # Configuration
│
├── outputs/             # Generated results
├── data/                # Dataset (excluded)
│
└── README.md
```

---

## 🚀 Future Work

* Quantum kernel methods (QSVC) for EEG classification
* Real-time EEG integration
* Explainability using SHAP
* Multi-channel EEG modeling

---

## 🧠 Conclusion

This work shows that:

> Brain states are not just frequency patterns — they are complex dynamical systems.

By incorporating complexity measures, we achieve:

* Better classification
* Deeper understanding of neural signals

---

## 📜 License

MIT License

---

## 👤 Author

**Janaki Nageshwaran**
AI & Neuroscience Enthusiast
Focus: Consciousness, AGI, Neural Systems
