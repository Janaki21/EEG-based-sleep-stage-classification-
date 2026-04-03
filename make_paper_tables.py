from pathlib import Path
import pandas as pd

from config import OUTPUT_DIR


def find_file(filename):
    candidates = [
        OUTPUT_DIR / filename,
        OUTPUT_DIR / "archive_raw" / filename,
        OUTPUT_DIR / "tables" / "paper_main" / filename,
        OUTPUT_DIR / "tables" / "paper_reference" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def safe_read_csv(filename):
    path = find_file(filename)
    if path is not None:
        return pd.read_csv(path)
    return None


def ensure_dirs():
    (OUTPUT_DIR / "tables" / "paper_main").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tables" / "paper_reference").mkdir(parents=True, exist_ok=True)


def make_main_model_comparison():
    df = safe_read_csv("model_results_summary.csv")
    if df is None:
        print("Missing model_results_summary.csv")
        return

    cols = [
        "feature_set", "model",
        "macro_f1_mean", "macro_f1_boot_ci_low", "macro_f1_boot_ci_high",
        "kappa_mean", "kappa_boot_ci_low", "kappa_boot_ci_high"
    ]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()

    out = out.rename(columns={
        "feature_set": "Feature Set",
        "model": "Model",
        "macro_f1_mean": "Macro F1",
        "macro_f1_boot_ci_low": "F1 CI Low",
        "macro_f1_boot_ci_high": "F1 CI High",
        "kappa_mean": "Kappa",
        "kappa_boot_ci_low": "Kappa CI Low",
        "kappa_boot_ci_high": "Kappa CI High",
    })

    out = out.sort_values(["Macro F1", "Kappa"], ascending=False)
    out.to_csv(OUTPUT_DIR / "tables" / "paper_main" / "table_main_model_comparison.csv", index=False)
    print("Saved table_main_model_comparison.csv")


def make_ablation_table():
    df = safe_read_csv("model_results_summary.csv")
    if df is None:
        print("Missing model_results_summary.csv")
        return

    wanted_sets = [
        "statistical",
        "spectral",
        "complexity",
        "stat+complexity",
        "spectral+complexity",
        "all",
    ]

    out = df[df["feature_set"].isin(wanted_sets)].copy()
    out = out[["feature_set", "model", "macro_f1_mean", "kappa_mean"]]
    out = out.rename(columns={
        "feature_set": "Feature Set",
        "model": "Model",
        "macro_f1_mean": "Macro F1",
        "kappa_mean": "Kappa",
    })
    out = out.sort_values(["Model", "Macro F1"], ascending=[True, False])
    out.to_csv(OUTPUT_DIR / "tables" / "paper_main" / "table_ablation_feature_sets.csv", index=False)
    print("Saved table_ablation_feature_sets.csv")


def make_best_model_table():
    df = safe_read_csv("best_model_summary.csv")
    if df is None:
        print("Missing best_model_summary.csv")
        return

    out = df.rename(columns={
        "feature_set": "Best Feature Set",
        "model": "Best Model",
        "macro_f1_mean": "Macro F1",
        "kappa_mean": "Kappa",
    })
    out.to_csv(OUTPUT_DIR / "tables" / "paper_main" / "table_best_model_summary.csv", index=False)
    print("Saved table_best_model_summary.csv")


def make_per_class_table():
    df = safe_read_csv("best_model_per_class_metrics.csv")
    if df is None:
        print("Missing best_model_per_class_metrics.csv")
        return

    out = df[["stage_name", "precision", "recall", "f1_score", "support"]].copy()
    out = out.rename(columns={
        "stage_name": "Stage",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "support": "Support",
    })
    out.to_csv(OUTPUT_DIR / "tables" / "paper_main" / "table_best_model_per_class.csv", index=False)
    print("Saved table_best_model_per_class.csv")


def make_deep_vs_classical_table():
    classical = safe_read_csv("best_model_summary.csv")
    deep = safe_read_csv("deep_baseline_summary.csv")

    rows = []

    if classical is not None:
        row = classical.iloc[0]
        rows.append({
            "Approach": "Interpretable ML",
            "Method": f"{row['feature_set']} + {row['model']}",
            "Macro F1": row["macro_f1_mean"],
            "Kappa": row["kappa_mean"],
            "Notes": "Best classical/interpretable pipeline",
        })

    if deep is not None:
        row = deep.iloc[0]
        rows.append({
            "Approach": "Deep Learning Baseline",
            "Method": row["model"],
            "Macro F1": row["macro_f1_mean"],
            "Kappa": row["kappa_mean"],
            "Notes": "Grouped-CV simple CNN baseline",
        })

    if not rows:
        print("Missing best_model_summary.csv and deep_baseline_summary.csv")
        return

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "tables" / "paper_main" / "table_classical_vs_deep.csv", index=False)
    print("Saved table_classical_vs_deep.csv")


def make_dataset_table():
    ds = safe_read_csv("dataset_summary.csv")
    stage = safe_read_csv("dataset_stage_distribution.csv")

    if ds is not None:
        ds.to_csv(OUTPUT_DIR / "tables" / "paper_main" / "table_dataset_summary.csv", index=False)
        print("Saved table_dataset_summary.csv")

    if stage is not None:
        out = stage.rename(columns={
            "stage_name": "Stage",
            "n_epochs": "Epoch Count",
        })
        out.to_csv(OUTPUT_DIR / "tables" / "paper_main" / "table_dataset_stage_distribution.csv", index=False)
        print("Saved table_dataset_stage_distribution.csv")


def make_stats_table():
    df = safe_read_csv("wilcoxon_tests.csv")
    if df is None:
        print("Missing wilcoxon_tests.csv")
        return

    out = df.rename(columns={
        "model": "Model",
        "comparison": "Comparison",
        "metric": "Metric",
        "mean_a": "Mean A",
        "mean_b": "Mean B",
        "delta_b_minus_a": "Delta (B-A)",
        "effect_size_dz": "Effect Size dz",
        "statistic": "Statistic",
        "p_value": "p-value",
    })
    out.to_csv(OUTPUT_DIR / "tables" / "paper_reference" / "table_wilcoxon_tests.csv", index=False)
    print("Saved table_wilcoxon_tests.csv")


def make_importance_tables():
    perm = safe_read_csv("permutation_importance.csv")
    dom = safe_read_csv("domain_importance_summary.csv")
    feat = safe_read_csv("best_model_feature_importance.csv")

    if perm is not None:
        perm.head(20).to_csv(
            OUTPUT_DIR / "tables" / "paper_reference" / "table_top20_permutation_importance.csv",
            index=False
        )
        print("Saved table_top20_permutation_importance.csv")

    if dom is not None:
        dom.to_csv(
            OUTPUT_DIR / "tables" / "paper_main" / "table_domain_importance_summary.csv",
            index=False
        )
        print("Saved table_domain_importance_summary.csv")

    if feat is not None:
        feat.head(20).to_csv(
            OUTPUT_DIR / "tables" / "paper_reference" / "table_top20_native_feature_importance.csv",
            index=False
        )
        print("Saved table_top20_native_feature_importance.csv")


def make_error_table():
    df = safe_read_csv("best_model_error_analysis.csv")
    if df is None:
        print("Missing best_model_error_analysis.csv")
        return

    out = df.head(15).rename(columns={
        "true_stage": "True Stage",
        "pred_stage": "Predicted Stage",
        "count": "Count",
        "row_normalized_rate": "Normalized Error Rate",
    })
    out.to_csv(OUTPUT_DIR / "tables" / "paper_reference" / "table_top_error_patterns.csv", index=False)
    print("Saved table_top_error_patterns.csv")


def make_sota_template_filled_with_ours():
    rows = []

    best = safe_read_csv("best_model_summary.csv")
    deep = safe_read_csv("deep_baseline_summary.csv")

    if best is not None:
        b = best.iloc[0]
        rows.append({
            "Method": "This Work (Interpretable ML)",
            "Type": "Handcrafted features + classical ML",
            "Dataset": "Sleep-EDF",
            "Protocol": "Repeated StratifiedGroupKFold, subject-independent",
            "Macro F1": b["macro_f1_mean"],
            "Kappa": b["kappa_mean"],
            "Interpretability": "High",
            "Notes": f"{b['feature_set']} + {b['model']}",
        })

    if deep is not None:
        d = deep.iloc[0]
        rows.append({
            "Method": "This Work (Simple CNN Baseline)",
            "Type": "Deep learning baseline",
            "Dataset": "Sleep-EDF",
            "Protocol": "StratifiedGroupKFold, subject-independent",
            "Macro F1": d["macro_f1_mean"],
            "Kappa": d["kappa_mean"],
            "Interpretability": "Low",
            "Notes": "SimpleCNN baseline",
        })

    rows.extend([
        {
            "Method": "DeepSleepNet",
            "Type": "",
            "Dataset": "",
            "Protocol": "",
            "Macro F1": "",
            "Kappa": "",
            "Interpretability": "",
            "Notes": "Fill from paper",
        },
        {
            "Method": "ATTNSLEEP",
            "Type": "",
            "Dataset": "",
            "Protocol": "",
            "Macro F1": "",
            "Kappa": "",
            "Interpretability": "",
            "Notes": "Fill from paper",
        },
        {
            "Method": "SleepTransformer",
            "Type": "",
            "Dataset": "",
            "Protocol": "",
            "Macro F1": "",
            "Kappa": "",
            "Interpretability": "",
            "Notes": "Fill from paper",
        },
    ])

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "tables" / "paper_main" / "table_sota_comparison_working.csv", index=False)
    print("Saved table_sota_comparison_working.csv")


def main():
    ensure_dirs()
    make_main_model_comparison()
    make_ablation_table()
    make_best_model_table()
    make_per_class_table()
    make_deep_vs_classical_table()
    make_dataset_table()
    make_stats_table()
    make_importance_tables()
    make_error_table()
    make_sota_template_filled_with_ours()
    print("\nAll paper tables created successfully.")


if __name__ == "__main__":
    main()