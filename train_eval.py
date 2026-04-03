from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon, t
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.inspection import permutation_importance

from config import OUTPUT_DIR, RANDOM_STATE, N_SPLITS
from utils import print_header

warnings.filterwarnings("ignore")


STAGE_NAMES = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}
LABEL_ORDER = [0, 1, 2, 3, 4]


# ----------------------------
# Path helpers
# ----------------------------
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


def load_npy_flexible(filename):
    return np.load(find_output_file(filename), allow_pickle=True)


# ----------------------------
# Utility functions
# ----------------------------
def mean_std_ci(values, alpha=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0.0

    if n <= 1:
        return float(mean), float(std), float(mean), float(mean)

    sem = std / np.sqrt(n)
    tcrit = t.ppf((1 + alpha) / 2.0, df=n - 1)
    ci_low = mean - tcrit * sem
    ci_high = mean + tcrit * sem
    return float(mean), float(std), float(ci_low), float(ci_high)


def bootstrap_ci(values, n_bootstrap=2000, alpha=0.95, random_state=42):
    rng = np.random.default_rng(random_state)
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(values[0]), float(values[0])

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))

    lower = np.percentile(boot_means, ((1 - alpha) / 2) * 100)
    upper = np.percentile(boot_means, (1 - (1 - alpha) / 2) * 100)
    return float(lower), float(upper)


def paired_effect_size_dz(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = y - x
    sd = np.std(d, ddof=1) if len(d) > 1 else 0.0
    if sd == 0:
        return 0.0
    return float(np.mean(d) / sd)


def safe_wilcoxon(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        return {"statistic": np.nan, "p_value": np.nan}

    if np.allclose(x, y):
        return {"statistic": 0.0, "p_value": 1.0}

    try:
        stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return {"statistic": float(stat), "p_value": float(p)}
    except Exception:
        return {"statistic": np.nan, "p_value": np.nan}


def get_domain_map(columns):
    domain_map = {}
    for col in columns:
        if col in [
            "mean", "std", "var", "skew", "kurtosis", "rms", "zcr",
            "hjorth_activity", "hjorth_mobility", "hjorth_complexity"
        ]:
            domain_map[col] = "statistical"
        elif col in [
            "abs_delta", "abs_theta", "abs_alpha", "abs_sigma", "abs_beta",
            "rel_delta", "rel_theta", "rel_alpha", "rel_sigma", "rel_beta",
            "spectral_entropy", "peak_frequency", "sef95"
        ]:
            domain_map[col] = "spectral"
        elif col in [
            "approx_entropy", "sample_entropy", "perm_entropy",
            "higuchi_fd", "lz_complexity"
        ]:
            domain_map[col] = "complexity"
        else:
            domain_map[col] = "unknown"
    return domain_map


# ----------------------------
# Data loading
# ----------------------------
def load_data():
    y = load_npy_flexible("y_labels.npy")
    subject_ids = load_npy_flexible("subject_ids.npy")

    stat = read_csv_flexible("features_statistical.csv")
    spec = read_csv_flexible("features_spectral.csv")
    comp = read_csv_flexible("features_complexity.csv")

    return stat, spec, comp, y, subject_ids


def get_feature_sets(stat, spec, comp):
    return {
        "statistical": stat,
        "spectral": spec,
        "complexity": comp,
        "stat+complexity": pd.concat([stat, comp], axis=1),
        "spectral+complexity": pd.concat([spec, comp], axis=1),
        "all": pd.concat([stat, spec, comp], axis=1),
    }


# ----------------------------
# Models
# ----------------------------
def build_model(model_name="logreg"):
    if model_name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

    elif model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")


# ----------------------------
# One evaluation run
# ----------------------------
def evaluate_model_once(X, y, groups, model_name="logreg", seed=RANDOM_STATE):
    sgkf = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=seed,
    )

    fold_rows = []
    per_fold_reports = []

    y_true_fullorder = np.array(y).copy()
    y_pred_fullorder = np.full_like(y_true_fullorder, fill_value=-1)
    fold_ids_fullorder = np.full_like(y_true_fullorder, fill_value=-1)

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_model(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_test_samples": len(test_idx),
                "accuracy": acc,
                "macro_f1": macro_f1,
                "balanced_acc": balanced_acc,
                "kappa": kappa,
            }
        )

        report = classification_report(
            y_test,
            y_pred,
            labels=LABEL_ORDER,
            target_names=[STAGE_NAMES[i] for i in LABEL_ORDER],
            output_dict=True,
            zero_division=0,
        )

        for label_name in STAGE_NAMES.values():
            per_fold_reports.append(
                {
                    "fold": fold_idx,
                    "stage_name": label_name,
                    "precision": report[label_name]["precision"],
                    "recall": report[label_name]["recall"],
                    "f1_score": report[label_name]["f1-score"],
                    "support": report[label_name]["support"],
                }
            )

        y_pred_fullorder[test_idx] = y_pred
        fold_ids_fullorder[test_idx] = fold_idx

    fold_df = pd.DataFrame(fold_rows)
    per_fold_stage_df = pd.DataFrame(per_fold_reports)

    return {
        "fold_df": fold_df,
        "per_fold_stage_df": per_fold_stage_df,
        "y_true_fullorder": y_true_fullorder,
        "y_pred_fullorder": y_pred_fullorder,
        "fold_ids_fullorder": fold_ids_fullorder,
    }


# ----------------------------
# Repeated grouped CV
# ----------------------------
def repeated_group_cv(X, y, groups, model_name="logreg", n_repeats=5):
    repeated_rows = []
    repeated_stage_rows = []

    best_repeat_result = None
    best_repeat_score = -np.inf

    for repeat_idx in range(n_repeats):
        seed = RANDOM_STATE + repeat_idx
        run = evaluate_model_once(X, y, groups, model_name=model_name, seed=seed)

        fold_df = run["fold_df"].copy()
        fold_df["repeat"] = repeat_idx + 1
        fold_df["seed"] = seed
        repeated_rows.append(fold_df)

        stage_df = run["per_fold_stage_df"].copy()
        stage_df["repeat"] = repeat_idx + 1
        stage_df["seed"] = seed
        repeated_stage_rows.append(stage_df)

        repeat_mean_f1 = fold_df["macro_f1"].mean()
        if repeat_mean_f1 > best_repeat_score:
            best_repeat_score = repeat_mean_f1
            best_repeat_result = run

    repeated_df = pd.concat(repeated_rows, ignore_index=True)
    repeated_stage_df = pd.concat(repeated_stage_rows, ignore_index=True)

    return {
        "repeated_df": repeated_df,
        "repeated_stage_df": repeated_stage_df,
        "best_repeat_result": best_repeat_result,
    }


# ----------------------------
# Metrics / reports
# ----------------------------
def summarize_metrics(repeated_df):
    summary = {}
    for metric in ["accuracy", "macro_f1", "balanced_acc", "kappa"]:
        values = repeated_df[metric].values
        mean, std, ci_low, ci_high = mean_std_ci(values)
        boot_low, boot_high = bootstrap_ci(values)

        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std
        summary[f"{metric}_ci_low"] = ci_low
        summary[f"{metric}_ci_high"] = ci_high
        summary[f"{metric}_boot_ci_low"] = boot_low
        summary[f"{metric}_boot_ci_high"] = boot_high
    return summary


def compute_pooled_per_class_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        zero_division=0,
    )

    rows = []
    for idx, label in enumerate(LABEL_ORDER):
        rows.append(
            {
                "stage_id": label,
                "stage_name": STAGE_NAMES[label],
                "precision": precision[idx],
                "recall": recall[idx],
                "f1_score": f1[idx],
                "support": int(support[idx]),
            }
        )
    return pd.DataFrame(rows)


def compute_confusion_details(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    cm_norm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER, normalize="true")

    cm_df = pd.DataFrame(
        cm,
        index=[STAGE_NAMES[i] for i in LABEL_ORDER],
        columns=[STAGE_NAMES[i] for i in LABEL_ORDER],
    )
    cm_norm_df = pd.DataFrame(
        cm_norm,
        index=[STAGE_NAMES[i] for i in LABEL_ORDER],
        columns=[STAGE_NAMES[i] for i in LABEL_ORDER],
    )

    error_rows = []
    for i, true_label in enumerate(LABEL_ORDER):
        for j, pred_label in enumerate(LABEL_ORDER):
            if i == j:
                continue
            error_rows.append(
                {
                    "true_stage": STAGE_NAMES[true_label],
                    "pred_stage": STAGE_NAMES[pred_label],
                    "count": int(cm[i, j]),
                    "row_normalized_rate": float(cm_norm[i, j]),
                }
            )

    error_df = pd.DataFrame(error_rows).sort_values(
        ["count", "row_normalized_rate"], ascending=False
    )

    return cm_df, cm_norm_df, error_df


def create_dataset_summary(y, groups):
    counts = pd.Series(y).value_counts().sort_index()

    rows = []
    for label in LABEL_ORDER:
        rows.append(
            {
                "stage_id": label,
                "stage_name": STAGE_NAMES[label],
                "n_epochs": int(counts.get(label, 0)),
            }
        )

    stage_df = pd.DataFrame(rows)
    dataset_df = pd.DataFrame(
        [
            {
                "n_total_epochs": int(len(y)),
                "n_unique_subject_groups": int(pd.Series(groups).nunique()),
                "n_classes": len(LABEL_ORDER),
            }
        ]
    )

    return dataset_df, stage_df


def fit_final_model(X, y, model_name):
    model = build_model(model_name)
    model.fit(X, y)
    return model


def compute_feature_importance(final_model, X, y, model_name, feature_set_name):
    if model_name == "logreg":
        clf = final_model.named_steps["clf"]
        coef_matrix = np.abs(clf.coef_)
        native_importance = coef_matrix.mean(axis=0)
    elif model_name == "rf":
        native_importance = final_model.feature_importances_
    else:
        native_importance = np.zeros(X.shape[1])

    native_df = pd.DataFrame(
        {
            "feature_name": X.columns,
            "importance": native_importance,
            "feature_set": feature_set_name,
            "model": model_name,
            "importance_type": "native",
        }
    ).sort_values("importance", ascending=False)

    perm = permutation_importance(
        final_model,
        X,
        y,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    perm_df = pd.DataFrame(
        {
            "feature_name": X.columns,
            "importance": perm.importances_mean,
            "importance_std": perm.importances_std,
            "feature_set": feature_set_name,
            "model": model_name,
            "importance_type": "permutation",
        }
    ).sort_values("importance", ascending=False)

    domain_map = get_domain_map(X.columns)
    perm_df["domain"] = perm_df["feature_name"].map(domain_map)
    native_df["domain"] = native_df["feature_name"].map(domain_map)

    domain_summary = (
        perm_df.groupby("domain", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .rename(columns={"importance": "total_permutation_importance"})
    )

    stability_df = pd.DataFrame(
        {
            "feature_name": X.columns,
            "perm_mean": perm.importances_mean,
            "perm_std": perm.importances_std,
        }
    ).sort_values("perm_mean", ascending=False)

    return native_df, perm_df, domain_summary, stability_df


def save_hypothesis_file():
    text = """Primary hypothesis:
Nonlinear complexity features improve inter-subject generalization beyond spectral-only features in single-channel EEG sleep stage classification.

Secondary hypothesis:
Combining spectral and complexity features yields better performance than either family alone.

Interpretive hypothesis:
Complexity-sensitive features are especially useful for transitional or heterogeneous states such as N1 and REM, while more regular states such as N3 remain easier to classify with conventional descriptors.
"""
    with open(OUTPUT_DIR / "hypotheses.txt", "w", encoding="utf-8") as f:
        f.write(text)


def save_sota_comparison_template():
    rows = [
        {
            "method": "This work - interpretable ML",
            "model_type": "Handcrafted + LogisticRegression/RandomForest",
            "channel_setup": "Single-channel Fpz-Cz",
            "cv_protocol": "Repeated subject-independent StratifiedGroupKFold",
            "macro_f1": "",
            "kappa": "",
            "notes": "Fill with final best result from model_results_summary.csv",
        },
        {
            "method": "DeepSleepNet",
            "model_type": "Deep learning",
            "channel_setup": "",
            "cv_protocol": "",
            "macro_f1": "",
            "kappa": "",
            "notes": "Fill from cited paper",
        },
        {
            "method": "ATTNSLEEP",
            "model_type": "Deep learning",
            "channel_setup": "",
            "cv_protocol": "",
            "macro_f1": "",
            "kappa": "",
            "notes": "Fill from cited paper",
        },
        {
            "method": "SleepTransformer",
            "model_type": "Deep learning",
            "channel_setup": "",
            "cv_protocol": "",
            "macro_f1": "",
            "kappa": "",
            "notes": "Fill from cited paper",
        },
    ]
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "sota_comparison_template.csv", index=False)


def run_wilcoxon_tests(repeated_results_dict):
    comparisons = [
        ("LogisticRegression", "spectral", "complexity"),
        ("LogisticRegression", "spectral", "spectral+complexity"),
        ("LogisticRegression", "statistical", "stat+complexity"),
        ("LogisticRegression", "spectral+complexity", "all"),
        ("RandomForest", "spectral", "complexity"),
        ("RandomForest", "spectral", "spectral+complexity"),
        ("RandomForest", "statistical", "stat+complexity"),
        ("RandomForest", "spectral+complexity", "all"),
    ]

    rows = []

    for model_label, a_name, b_name in comparisons:
        key_a = (a_name, model_label)
        key_b = (b_name, model_label)

        if key_a not in repeated_results_dict or key_b not in repeated_results_dict:
            continue

        df_a = repeated_results_dict[key_a].sort_values(["repeat", "fold"]).reset_index(drop=True)
        df_b = repeated_results_dict[key_b].sort_values(["repeat", "fold"]).reset_index(drop=True)

        min_len = min(len(df_a), len(df_b))
        df_a = df_a.iloc[:min_len]
        df_b = df_b.iloc[:min_len]

        for metric in ["macro_f1", "kappa", "balanced_acc"]:
            test = safe_wilcoxon(df_a[metric].values, df_b[metric].values)
            dz = paired_effect_size_dz(df_a[metric].values, df_b[metric].values)

            rows.append(
                {
                    "model": model_label,
                    "comparison": f"{a_name} vs {b_name}",
                    "metric": metric,
                    "mean_a": df_a[metric].mean(),
                    "mean_b": df_b[metric].mean(),
                    "delta_b_minus_a": df_b[metric].mean() - df_a[metric].mean(),
                    "effect_size_dz": dz,
                    "statistic": test["statistic"],
                    "p_value": test["p_value"],
                }
            )

    return pd.DataFrame(rows)


# ----------------------------
# Main experiment runner
# ----------------------------
def run_all_experiments(n_repeats=5):
    stat, spec, comp, y, groups = load_data()
    feature_sets = get_feature_sets(stat, spec, comp)

    dataset_df, stage_dist_df = create_dataset_summary(y, groups)
    dataset_df.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
    stage_dist_df.to_csv(OUTPUT_DIR / "dataset_stage_distribution.csv", index=False)

    summary_rows = []
    repeated_fold_rows_all = []
    repeated_stage_rows_all = []
    repeated_results_dict = {}

    best_result = None
    best_score = -np.inf

    print_header("RUNNING EXPERIMENTS")

    for feature_name, X in feature_sets.items():
        print(f"\nEvaluating: {feature_name}")

        for model_name, model_label in [("logreg", "LogisticRegression"), ("rf", "RandomForest")]:
            repeated = repeated_group_cv(
                X, y, groups, model_name=model_name, n_repeats=n_repeats
            )

            repeated_df = repeated["repeated_df"].copy()
            repeated_stage_df = repeated["repeated_stage_df"].copy()
            best_repeat_result = repeated["best_repeat_result"]

            repeated_df["feature_set"] = feature_name
            repeated_df["model"] = model_label
            repeated_stage_df["feature_set"] = feature_name
            repeated_stage_df["model"] = model_label

            repeated_fold_rows_all.append(repeated_df)
            repeated_stage_rows_all.append(repeated_stage_df)
            repeated_results_dict[(feature_name, model_label)] = repeated_df[
                ["repeat", "fold", "accuracy", "macro_f1", "balanced_acc", "kappa"]
            ].copy()

            summary = summarize_metrics(repeated_df)

            row = {
                "feature_set": feature_name,
                "model": model_label,
            }
            row.update(summary)
            summary_rows.append(row)

            repeated_df.to_csv(
                OUTPUT_DIR / f"repeated_cv_{feature_name}_{model_label}.csv",
                index=False
            )

            print(
                f"  {model_label} -> "
                f"F1: {summary['macro_f1_mean']:.4f} "
                f"[{summary['macro_f1_boot_ci_low']:.4f}, {summary['macro_f1_boot_ci_high']:.4f}], "
                f"Kappa: {summary['kappa_mean']:.4f} "
                f"[{summary['kappa_boot_ci_low']:.4f}, {summary['kappa_boot_ci_high']:.4f}]"
            )

            if summary["macro_f1_mean"] > best_score:
                best_score = summary["macro_f1_mean"]
                best_result = {
                    "feature_set": feature_name,
                    "model_label": model_label,
                    "model_name_internal": model_name,
                    "X_data": X,
                    "y_true_fullorder": best_repeat_result["y_true_fullorder"],
                    "y_pred_fullorder": best_repeat_result["y_pred_fullorder"],
                    "fold_ids_fullorder": best_repeat_result["fold_ids_fullorder"],
                    "macro_f1_mean": summary["macro_f1_mean"],
                    "kappa_mean": summary["kappa_mean"],
                }

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["macro_f1_mean", "kappa_mean"], ascending=False
    )
    repeated_fold_results_df = pd.concat(repeated_fold_rows_all, ignore_index=True)
    repeated_stage_results_df = pd.concat(repeated_stage_rows_all, ignore_index=True)

    summary_df.to_csv(OUTPUT_DIR / "model_results_summary.csv", index=False)
    repeated_fold_results_df.to_csv(OUTPUT_DIR / "model_results_folds.csv", index=False)
    repeated_stage_results_df.to_csv(OUTPUT_DIR / "per_fold_stage_metrics.csv", index=False)

    backward_df = summary_df.rename(
        columns={
            "accuracy_mean": "accuracy",
            "macro_f1_mean": "macro_f1",
            "balanced_acc_mean": "balanced_acc",
            "kappa_mean": "kappa",
        }
    )[["feature_set", "model", "accuracy", "macro_f1", "balanced_acc", "kappa"]]
    backward_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)

    np.save(OUTPUT_DIR / "best_y_true.npy", best_result["y_true_fullorder"])
    np.save(OUTPUT_DIR / "best_y_pred.npy", best_result["y_pred_fullorder"])
    np.save(OUTPUT_DIR / "best_fold_ids.npy", best_result["fold_ids_fullorder"])

    best_meta = pd.DataFrame(
        [
            {
                "feature_set": best_result["feature_set"],
                "model": best_result["model_label"],
                "macro_f1_mean": best_result["macro_f1_mean"],
                "kappa_mean": best_result["kappa_mean"],
            }
        ]
    )
    best_meta.to_csv(OUTPUT_DIR / "best_model_summary.csv", index=False)

    pooled_per_class_df = compute_pooled_per_class_metrics(
        best_result["y_true_fullorder"], best_result["y_pred_fullorder"]
    )
    pooled_per_class_df.to_csv(OUTPUT_DIR / "best_model_per_class_metrics.csv", index=False)

    cm_df, cm_norm_df, error_df = compute_confusion_details(
        best_result["y_true_fullorder"], best_result["y_pred_fullorder"]
    )
    cm_df.to_csv(OUTPUT_DIR / "best_model_confusion_matrix_counts.csv")
    cm_norm_df.to_csv(OUTPUT_DIR / "best_model_confusion_matrix_normalized.csv")
    error_df.to_csv(OUTPUT_DIR / "best_model_error_analysis.csv", index=False)

    final_model = fit_final_model(
        best_result["X_data"],
        y,
        best_result["model_name_internal"]
    )
    joblib.dump(final_model, OUTPUT_DIR / "best_model.joblib")

    native_imp_df, perm_imp_df, domain_imp_df, stability_df = compute_feature_importance(
        final_model,
        best_result["X_data"],
        y,
        best_result["model_name_internal"],
        best_result["feature_set"]
    )

    native_imp_df.to_csv(OUTPUT_DIR / "best_model_feature_importance.csv", index=False)
    perm_imp_df.to_csv(OUTPUT_DIR / "permutation_importance.csv", index=False)
    domain_imp_df.to_csv(OUTPUT_DIR / "domain_importance_summary.csv", index=False)
    stability_df.to_csv(OUTPUT_DIR / "feature_stability.csv", index=False)

    wilcoxon_df = run_wilcoxon_tests(repeated_results_dict)
    wilcoxon_df.to_csv(OUTPUT_DIR / "wilcoxon_tests.csv", index=False)

    save_hypothesis_file()
    save_sota_comparison_template()

    print_header("FINAL SUMMARY RESULTS")
    print(summary_df[[
        "feature_set", "model",
        "macro_f1_mean", "macro_f1_boot_ci_low", "macro_f1_boot_ci_high",
        "kappa_mean", "kappa_boot_ci_low", "kappa_boot_ci_high"
    ]])

    print_header("BEST MODEL")
    print(best_meta)

    print_header("PER-CLASS METRICS (BEST MODEL)")
    print(pooled_per_class_df)

    print_header("TOP ERROR PATTERNS")
    print(error_df.head(10))

    print_header("WILCOXON + EFFECT SIZE")
    print(wilcoxon_df)

    print_header("DOMAIN IMPORTANCE SUMMARY")
    print(domain_imp_df)

    return {
        "summary_df": summary_df,
        "repeated_fold_results_df": repeated_fold_results_df,
        "repeated_stage_results_df": repeated_stage_results_df,
        "pooled_per_class_df": pooled_per_class_df,
        "cm_df": cm_df,
        "cm_norm_df": cm_norm_df,
        "error_df": error_df,
        "native_importance_df": native_imp_df,
        "perm_importance_df": perm_imp_df,
        "domain_importance_df": domain_imp_df,
        "stability_df": stability_df,
        "wilcoxon_df": wilcoxon_df,
    }


def main():
    outputs = run_all_experiments(n_repeats=5)

    print_header("TRAINING COMPLETE")
    print("\nSaved files:")
    for name in [
        "model_results_summary.csv",
        "model_results_folds.csv",
        "per_fold_stage_metrics.csv",
        "best_model_per_class_metrics.csv",
        "best_model_confusion_matrix_counts.csv",
        "best_model_confusion_matrix_normalized.csv",
        "best_model_error_analysis.csv",
        "best_model_feature_importance.csv",
        "permutation_importance.csv",
        "domain_importance_summary.csv",
        "feature_stability.csv",
        "wilcoxon_tests.csv",
        "dataset_summary.csv",
        "dataset_stage_distribution.csv",
        "sota_comparison_template.csv",
        "hypotheses.txt",
        "best_model.joblib",
    ]:
        print(f"- {OUTPUT_DIR / name}")


if __name__ == "__main__":
    main()