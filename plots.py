import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import joblib

from config import OUTPUT_DIR, FIGURES_DIR
from utils import print_header

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def plot_model_comparison():
    df = pd.read_csv(OUTPUT_DIR / "model_results.csv")

    plt.figure(figsize=(10, 6))
    for model_name in df["model"].unique():
        subset = df[df["model"] == model_name]
        plt.plot(subset["feature_set"], subset["macro_f1"], marker="o", label=model_name)

    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Macro F1")
    plt.title("Model Performance Across Feature Sets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_macro_f1.png", dpi=300)
    plt.close()


def plot_confusion_matrix_best():
    y_true = np.load(OUTPUT_DIR / "best_y_true.npy")
    y_pred = np.load(OUTPUT_DIR / "best_y_pred.npy")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=STAGE_NAMES)

    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix - Best Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "best_model_confusion_matrix.png", dpi=300)
    plt.close()


def plot_class_distribution():
    y = np.load(OUTPUT_DIR / "y_labels.npy")
    counts = pd.Series(y).value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(STAGE_NAMES, counts.values)
    plt.ylabel("Number of Epochs")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=300)
    plt.close()


# 🧠 NEW: Feature Importance
def plot_feature_importance():
    model = joblib.load(OUTPUT_DIR / "best_model.joblib")

    if hasattr(model, "named_steps"):
        clf = model.named_steps["clf"]
        feature_names = pd.read_csv(OUTPUT_DIR / "features_all.csv").columns
    else:
        clf = model
        feature_names = pd.read_csv(OUTPUT_DIR / "features_all.csv").columns

    if hasattr(clf, "coef_"):
        importance = np.abs(clf.coef_).mean(axis=0)
    elif hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    else:
        print("Feature importance not available for this model")
        return

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values(by="importance", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(df["feature"], df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=300)
    plt.close()


# 🧠 NEW: Per-class F1
def plot_per_class_f1():
    y_true = np.load(OUTPUT_DIR / "best_y_true.npy")
    y_pred = np.load(OUTPUT_DIR / "best_y_pred.npy")

    f1_scores = f1_score(y_true, y_pred, average=None)

    plt.figure(figsize=(8, 5))
    plt.bar(STAGE_NAMES, f1_scores)
    plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Scores")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "per_class_f1.png", dpi=300)
    plt.close()


def main():
    print_header("GENERATING PLOTS")

    plot_model_comparison()
    plot_confusion_matrix_best()
    plot_class_distribution()
    plot_feature_importance()
    plot_per_class_f1()

    print_header("PLOTS SAVED")
    print(f"Saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()