import numpy as np
import pandas as pd
import joblib

from config import OUTPUT_DIR
from features import extract_all_features


def run_demo():
    print("=== EEG Brain State Demo ===")

    # Load model
    model = joblib.load(OUTPUT_DIR / "best_model.joblib")

    # Load dataset
    X = np.load(OUTPUT_DIR / "X_epochs.npy")

    # Pick random sample
    idx = np.random.randint(0, len(X))
    sample = X[idx]

    print(f"Using sample index: {idx}")

    # Extract features for this sample
    features_df = extract_all_features(np.array([sample]), sfreq=100.0)

    # Predict
    prediction = model.predict(features_df)[0]

    stage_map = {
        0: "Wake",
        1: "N1",
        2: "N2",
        3: "N3",
        4: "REM"
    }

    print(f"Predicted Brain State: {stage_map[prediction]}")


if __name__ == "__main__":
    run_demo()