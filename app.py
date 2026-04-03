import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from config import OUTPUT_DIR
from features import extract_all_features

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="EEG Intelligence System",
    layout="wide",
)

# -------------------------
# CONSTANTS
# -------------------------
STAGE_MAP = {
    0: "Wake",
    1: "N1 (Light Sleep)",
    2: "N2 (Sleep)",
    3: "N3 (Deep Sleep)",
    4: "REM",
}

CONFUSION_MATRIX_PATH = OUTPUT_DIR / "figures" / "best_model_confusion_matrix.png"


# -------------------------
# HELPERS
# -------------------------
@st.cache_data
def load_numpy_data():
    x_path = OUTPUT_DIR / "X_epochs.npy"
    y_path = OUTPUT_DIR / "y_labels.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing file: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing file: {y_path}")

    X = np.load(x_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    return X, y


@st.cache_data
def load_results():
    results_path = OUTPUT_DIR / "model_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing file: {results_path}")
    return pd.read_csv(results_path)


@st.cache_resource
def load_model():
    model_path = OUTPUT_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing file: {model_path}")
    return joblib.load(model_path)


def render_header():
    st.title("🧠 EEG Intelligence System")
    st.markdown(
        """
### Understanding Brain States using AI and Neural Complexity

This system analyzes EEG signals and predicts brain states using:

- **Statistical features**
- **Spectral features**
- **Neural complexity features** such as entropy and fractal-based measures

**Main idea:** brain states are not only different in frequency content, but also in signal complexity and structure.
"""
    )


def plot_dataset_distribution(y):
    counts = pd.Series(y).value_counts().sort_index()
    labels = [STAGE_MAP.get(int(i), str(i)) for i in counts.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, counts.values)
    ax.set_title("Dataset Class Distribution")
    ax.set_ylabel("Number of Epochs")
    plt.xticks(rotation=20)
    plt.tight_layout()
    st.pyplot(fig)


def plot_model_performance(results_df):
    fig, ax = plt.subplots(figsize=(9, 4))

    for model_name in results_df["model"].unique():
        subset = results_df[results_df["model"] == model_name]
        ax.plot(
            subset["feature_set"],
            subset["macro_f1"],
            marker="o",
            label=model_name,
        )

    ax.set_title("Macro F1 Across Feature Sets")
    ax.set_ylabel("Macro F1")
    ax.legend()
    plt.xticks(rotation=25)
    plt.tight_layout()
    st.pyplot(fig)


def plot_eeg_signal(sample, sample_idx):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(sample)
    ax.set_title(f"EEG Signal - Sample {sample_idx}")
    ax.set_xlabel("Time Points")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    st.pyplot(fig)


def explain_prediction(features_df):
    st.markdown("### 🧠 Why this prediction?")

    feature_values = features_df.iloc[0].copy()
    top_features = feature_values.abs().sort_values(ascending=False).head(8)

    st.markdown("**Most prominent extracted features for this signal:**")
    for feature_name, value in top_features.items():
        st.write(f"• **{feature_name}**: {value:.4f}")

    st.info(
        """
These features come from three groups:

- **Statistical:** magnitude and variation of the signal
- **Spectral:** how power is distributed across brain-wave bands
- **Complexity:** how structured, irregular, or information-rich the signal is

The model combines them to infer the most likely brain state.
"""
    )


def render_confusion_matrix():
    st.subheader("🧩 Confusion Matrix")

    if CONFUSION_MATRIX_PATH.exists():
        st.image(str(CONFUSION_MATRIX_PATH))
    else:
        st.warning("Confusion matrix image not found. Run Phase 5 to generate it.")


def render_overview_tab():
    st.subheader("🔬 What this work is about")
    st.markdown(
        """
Most EEG-based systems rely mainly on frequency bands like alpha, beta, and delta.

This project asks a deeper question:

**Can brain-state classification improve if we also measure signal complexity?**

Instead of only looking at frequency content, this system also examines how regular, irregular, or structured the neural signal is.
"""
    )

    st.subheader("⚙️ What the pipeline does")
    st.markdown(
        """
1. Takes raw EEG recordings
2. Splits them into 30-second epochs
3. Extracts three feature families
   - Statistical
   - Spectral
   - Complexity
4. Trains machine learning models
5. Predicts the brain state for each EEG segment
"""
    )

    st.subheader("💡 Central result")
    st.success(
        """
Complexity features substantially improved classification when combined with traditional handcrafted EEG features.

That means brain states are not defined only by wave frequencies, but also by underlying signal structure.
"""
    )

    st.subheader("🧠 What we have done in simple terms")
    st.markdown(
        """
- We took real EEG sleep recordings.
- We divided them into 30-second labeled segments.
- For each segment, we extracted:
  - basic signal statistics,
  - frequency information,
  - complexity information.
- We trained machine learning models on these features.
- We compared which feature groups worked best.

### Central issue solved
Most EEG systems only rely on frequency bands.  
Our work shows that **complexity features add important information**, which improves brain-state classification.
"""
    )


def render_demo_tab(X, model):
    st.subheader("🎬 Interactive Brain-State Demo")

    if X is None or len(X) == 0:
        st.error("EEG samples are not available.")
        return

    if model is None:
        st.error("Model is not available.")
        return

    left, right = st.columns([1, 1])

    with left:
        sample_idx = st.slider(
            "Select EEG sample",
            min_value=0,
            max_value=len(X) - 1,
            value=0,
            step=1,
        )

    with right:
        analyze = st.button("🔍 Analyze Signal")

    sample = X[sample_idx]

    st.markdown("### 📡 Raw EEG Signal")
    plot_eeg_signal(sample, sample_idx)

    st.markdown(
        """
### Demo explanation

This demo takes one EEG epoch, extracts the same features used during training, and sends them through the trained classifier to estimate the brain state.
"""
    )

    if analyze:
        with st.spinner("Extracting features and predicting..."):
            features_df = extract_all_features(np.array([sample]), sfreq=100.0)
            pred = model.predict(features_df)[0]

            st.markdown("### ✅ Demo Output")
            st.success(f"Predicted Brain State: **{STAGE_MAP[int(pred)]}**")

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features_df)[0]
                confidence = float(np.max(probs))
                st.info(f"Prediction confidence: **{confidence:.2f}**")

                prob_df = pd.DataFrame(
                    {
                        "Brain State": [STAGE_MAP[i] for i in range(len(probs))],
                        "Probability": probs,
                    }
                )

                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(prob_df["Brain State"], prob_df["Probability"])
                ax.set_ylim(0, 1)
                ax.set_ylabel("Probability")
                ax.set_title("Prediction Confidence by Class")
                plt.xticks(rotation=20)
                plt.tight_layout()
                st.pyplot(fig)

            explain_prediction(features_df)


def render_results_tab(y, results_df):
    st.subheader("📊 Experimental Results")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Dataset Distribution")
        plot_dataset_distribution(y)

    with c2:
        st.markdown("#### Model Comparison")
        plot_model_performance(results_df)

    st.markdown("#### Result Table")
    st.dataframe(results_df)

    render_confusion_matrix()


def main():
    render_header()

    X = None
    y = None
    results_df = None
    model = None

    try:
        X, y = load_numpy_data()
        results_df = load_results()
        model = load_model()
    except Exception as e:
        st.error(f"Failed to load project files: {e}")
        st.info(
            "Make sure these files exist inside OUTPUT_DIR: "
            "X_epochs.npy, y_labels.npy, model_results.csv, best_model.joblib"
        )
        return

    tab1, tab2, tab3 = st.tabs(["📘 Overview", "🎬 Live Demo", "📊 Results"])

    with tab1:
        render_overview_tab()

    with tab2:
        render_demo_tab(X, model)

    with tab3:
        render_results_tab(y, results_df)


if __name__ == "__main__":
    main()