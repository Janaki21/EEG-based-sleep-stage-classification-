import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUT_DIR
from utils import print_header


def find_output_file(filename):
    candidates = [
        OUTPUT_DIR / filename,
        OUTPUT_DIR / "archive_raw" / filename,
        OUTPUT_DIR / "tables" / "paper_main" / filename,
        OUTPUT_DIR / "tables" / "paper_reference" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename} in outputs, archive_raw, or tables folders.")


def read_csv_flexible(filename):
    return pd.read_csv(find_output_file(filename))


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_model_comparison():
    df = read_csv_flexible("model_results_summary.csv")

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = df["feature_set"] + " | " + df["model"]
    ax.barh(labels, df["macro_f1_mean"])
    ax.set_xlabel("Macro F1")
    ax.set_title("Model Comparison Across Feature Sets")
    save_figure(fig, "figure_model_comparison_macro_f1.png")

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = df["feature_set"] + " | " + df["model"]
    ax.barh(labels, df["kappa_mean"])
    ax.set_xlabel("Cohen's Kappa")
    ax.set_title("Agreement Comparison Across Feature Sets")
    save_figure(fig, "figure_model_comparison_kappa.png")


def plot_confusion_matrix():
    cm = read_csv_flexible("best_model_confusion_matrix_normalized.csv").set_index(
        read_csv_flexible("best_model_confusion_matrix_normalized.csv").columns[0]
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm.values, aspect="auto")
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns)
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index)
    ax.set_xlabel("Predicted Stage")
    ax.set_ylabel("True Stage")
    ax.set_title("Normalized Confusion Matrix (Best Model)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, "figure_confusion_matrix_normalized.png")


def plot_per_class_f1():
    df = read_csv_flexible("best_model_per_class_metrics.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["stage_name"], df["f1_score"])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Sleep Stage")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Scores for Best Model")

    for i, val in enumerate(df["f1_score"]):
        ax.text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)

    save_figure(fig, "figure_per_class_f1.png")


def plot_feature_importance(top_k=15):
    df = read_csv_flexible("best_model_feature_importance.csv").head(top_k)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["feature_name"][::-1], df["importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_k} Features in Best Model")
    save_figure(fig, "figure_feature_importance_top15.png")


def plot_stagewise_recall():
    df = read_csv_flexible("best_model_per_class_metrics.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["stage_name"], df["recall"])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Sleep Stage")
    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recall for Best Model")

    for i, val in enumerate(df["recall"]):
        ax.text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)

    save_figure(fig, "figure_per_class_recall.png")


def plot_top_confusions():
    df = read_csv_flexible("best_model_error_analysis.csv").head(10)

    labels = df["true_stage"] + "→" + df["pred_stage"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels[::-1], df["count"][::-1])
    ax.set_xlabel("Misclassification Count")
    ax.set_title("Top Misclassification Patterns")
    save_figure(fig, "figure_top_confusions.png")


def create_caption_notes():
    captions = {
        "figure_model_comparison_macro_f1.png":
            "Comparison of Macro F1 across all feature-set/model combinations under subject-independent evaluation.",
        "figure_model_comparison_kappa.png":
            "Comparison of Cohen's Kappa across all feature-set/model combinations, highlighting inter-rater style agreement beyond accuracy alone.",
        "figure_confusion_matrix_normalized.png":
            "Normalized confusion matrix for the best-performing model. Rows correspond to true labels and show where transitional stages, especially N1, are confused.",
        "figure_per_class_f1.png":
            "Per-class F1 scores for the best-performing model. The lower N1 score reflects the well-known difficulty of transitional sleep-stage discrimination.",
        "figure_feature_importance_top15.png":
            "Top-ranked features in the best model, illustrating the relative contributions of statistical, spectral, and complexity descriptors.",
        "figure_per_class_recall.png":
            "Per-class recall values for the best model, useful for understanding which stages are most often missed.",
        "figure_top_confusions.png":
            "Largest off-diagonal confusion patterns in the best model, supporting the error-analysis discussion.",
    }

    with open(OUTPUT_DIR / "figure_captions.txt", "w", encoding="utf-8") as f:
        for k, v in captions.items():
            f.write(f"{k}\n{v}\n\n")

    print(f"Saved: {OUTPUT_DIR / 'figure_captions.txt'}")


def create_result_notes():
    summary = read_csv_flexible("model_results_summary.csv")
    best = summary.sort_values("macro_f1_mean", ascending=False).iloc[0]
    per_class = read_csv_flexible("best_model_per_class_metrics.csv")
    errors = read_csv_flexible("best_model_error_analysis.csv").head(5)
    wilcoxon = read_csv_flexible("wilcoxon_tests.csv")

    lines = []
    lines.append("RESULTS INTERPRETATION NOTES")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        f"Best model: {best['feature_set']} + {best['model']} | "
        f"Macro F1 = {best['macro_f1_mean']:.4f}, Kappa = {best['kappa_mean']:.4f}"
    )
    lines.append("")
    lines.append("Per-class observations:")
    for _, row in per_class.iterrows():
        lines.append(
            f"- {row['stage_name']}: precision={row['precision']:.3f}, "
            f"recall={row['recall']:.3f}, f1={row['f1_score']:.3f}, support={int(row['support'])}"
        )
    lines.append("")
    lines.append("Top confusion patterns:")
    for _, row in errors.iterrows():
        lines.append(
            f"- {row['true_stage']} predicted as {row['pred_stage']}: "
            f"count={int(row['count'])}, normalized rate={row['row_normalized_rate']:.3f}"
        )
    lines.append("")
    lines.append("Statistical testing notes:")
    if len(wilcoxon) > 0:
        for _, row in wilcoxon.iterrows():
            lines.append(
                f"- {row['model']} | {row['comparison']} | {row['metric']} | "
                f"delta={row['delta_b_minus_a']:.4f}, dz={row['effect_size_dz']:.4f}, p={row['p_value']:.4f}"
            )

    with open(OUTPUT_DIR / "results_interpretation_notes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Saved: {OUTPUT_DIR / 'results_interpretation_notes.txt'}")


def main():
    print_header("GENERATING PLOTS + ANALYSIS OUTPUTS")

    plot_model_comparison()
    plot_confusion_matrix()
    plot_per_class_f1()
    plot_stagewise_recall()
    plot_feature_importance()
    plot_top_confusions()
    create_caption_notes()
    create_result_notes()

    print_header("PLOTS COMPLETE")


if __name__ == "__main__":
    main()