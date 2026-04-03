from pathlib import Path
import shutil

from config import OUTPUT_DIR


def ensure_dirs():
    folders = [
        OUTPUT_DIR / "figures" / "paper_main",
        OUTPUT_DIR / "figures" / "paper_reference",
        OUTPUT_DIR / "tables" / "paper_main",
        OUTPUT_DIR / "tables" / "paper_reference",
        OUTPUT_DIR / "models",
        OUTPUT_DIR / "cache",
        OUTPUT_DIR / "archive_raw",
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def move_if_exists(src, dst):
    src = Path(src)
    dst = Path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
        print(f"Moved: {src.name} -> {dst}")


def main():
    ensure_dirs()

    # ---------------- FIGURES: main ----------------
    main_figs = [
        "figure_model_comparison_macro_f1.png",
        "figure_model_comparison_kappa.png",
        "figure_confusion_matrix_normalized.png",
        "figure_per_class_f1.png",
        "figure_feature_importance_top15.png",
        "figure_top_confusions.png",
    ]

    for f in main_figs:
        move_if_exists(OUTPUT_DIR / f, OUTPUT_DIR / "figures" / "paper_main" / f)

    # ---------------- FIGURES: reference ----------------
    ref_figs = [
        "figure_per_class_recall.png",
        "n1_correct_psd.png",
        "n1_to_w_psd.png",
        "n1_to_n2_psd.png",
    ]

    for f in ref_figs:
        move_if_exists(OUTPUT_DIR / f, OUTPUT_DIR / "figures" / "paper_reference" / f)

    # ---------------- TEXT notes ----------------
    note_files = [
        "figure_captions.txt",
        "results_interpretation_notes.txt",
        "hypotheses.txt",
        "dataset_report.txt",
    ]

    for f in note_files:
        move_if_exists(OUTPUT_DIR / f, OUTPUT_DIR / "tables" / "paper_reference" / f)

    # ---------------- MODELS ----------------
    move_if_exists(OUTPUT_DIR / "best_model.joblib", OUTPUT_DIR / "models" / "best_model.joblib")

    # ---------------- RAW/ARCHIVE ----------------
    raw_files = [
        "features_all.csv",
        "features_complexity.csv",
        "features_spectral.csv",
        "features_statistical.csv",
        "features_spectral_complexity.csv",
        "features_stat_complexity.csv",
        "model_results.csv",
        "model_results_folds.csv",
        "model_results_summary.csv",
        "per_fold_stage_metrics.csv",
        "permutation_importance.csv",
        "feature_definitions.csv",
        "feature_stability.csv",
        "best_model_confusion_matrix_counts.csv",
        "best_model_confusion_matrix_normalized.csv",
        "best_model_error_analysis.csv",
        "best_model_feature_importance.csv",
        "best_model_per_class_metrics.csv",
        "best_model_summary.csv",
        "deep_baseline_fold_results.csv",
        "deep_baseline_per_class_metrics.csv",
        "deep_baseline_summary.csv",
        "domain_importance_summary.csv",
        "dataset_stage_distribution.csv",
        "dataset_summary.csv",
        "wilcoxon_tests.csv",
        "sota_comparison_template.csv",
        "feature_set_summary.csv",
        "label_map.csv",
        "epoch_summary.csv",
    ]

    for f in raw_files:
        move_if_exists(OUTPUT_DIR / f, OUTPUT_DIR / "archive_raw" / f)

    # repeated cv files
    for file in OUTPUT_DIR.glob("repeated_cv_*.csv"):
        move_if_exists(file, OUTPUT_DIR / "archive_raw" / file.name)

    print("\nOutputs organized successfully.")


if __name__ == "__main__":
    main()