import numpy as np
import pandas as pd
import joblib

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
)

from scipy.stats import wilcoxon

from config import OUTPUT_DIR, RANDOM_STATE, N_SPLITS
from utils import print_header

STAGE_NAMES = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}


def load_data():
    y = np.load(OUTPUT_DIR / "y_labels.npy")
    subject_ids = np.load(OUTPUT_DIR / "subject_ids.npy")

    stat = pd.read_csv(OUTPUT_DIR / "features_statistical.csv")
    spec = pd.read_csv(OUTPUT_DIR / "features_spectral.csv")
    comp = pd.read_csv(OUTPUT_DIR / "features_complexity.csv")

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


def build_model(model_name="logreg"):
    if model_name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500))
        ])
    elif model_name == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def evaluate_model(X, y, groups, model_name="logreg"):
    sgkf = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows = []
    all_true, all_pred = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_model(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        fold_rows.append({
            "fold": fold_idx,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "balanced_acc": balanced_acc,
            "kappa": kappa,
        })

        all_true.extend(y_test)
        all_pred.extend(y_pred)

    fold_df = pd.DataFrame(fold_rows)

    summary = {
        "accuracy_mean": fold_df["accuracy"].mean(),
        "accuracy_std": fold_df["accuracy"].std(ddof=1),
        "macro_f1_mean": fold_df["macro_f1"].mean(),
        "macro_f1_std": fold_df["macro_f1"].std(ddof=1),
        "balanced_acc_mean": fold_df["balanced_acc"].mean(),
        "balanced_acc_std": fold_df["balanced_acc"].std(ddof=1),
        "kappa_mean": fold_df["kappa"].mean(),
        "kappa_std": fold_df["kappa"].std(ddof=1),
        "y_true": np.array(all_true),
        "y_pred": np.array(all_pred),
        "fold_df": fold_df,
    }

    return summary


def compute_per_class_f1(y_true, y_pred):
    labels = [0, 1, 2, 3, 4]
    per_class = f1_score(y_true, y_pred, labels=labels, average=None)

    rows = []
    for label, score in zip(labels, per_class):
        rows.append({
            "stage_id": label,
            "stage_name": STAGE_NAMES[label],
            "f1_score": score,
        })

    return pd.DataFrame(rows)


def safe_wilcoxon(x, y):
    """
    Wilcoxon on paired fold scores.
    With only 3 folds, power is low; this is still useful as a formal check.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("Wilcoxon inputs must have same length.")

    if np.allclose(x, y):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
        }

    try:
        stat, p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        return {
            "statistic": float(stat),
            "p_value": float(p),
        }
    except Exception:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
        }


def run_wilcoxon_tests(fold_results_dict):
    """
    Key paired comparisons:
    1. spectral vs spectral+complexity
    2. statistical vs stat+complexity
    3. spectral+complexity vs all
    Uses Macro F1 and Kappa
    """
    comparisons = [
        ("LogisticRegression", "spectral", "spectral+complexity"),
        ("LogisticRegression", "statistical", "stat+complexity"),
        ("LogisticRegression", "spectral+complexity", "all"),
        ("RandomForest", "spectral", "spectral+complexity"),
        ("RandomForest", "statistical", "stat+complexity"),
        ("RandomForest", "spectral+complexity", "all"),
    ]

    rows = []

    for model_label, a_name, b_name in comparisons:
        key_a = (a_name, model_label)
        key_b = (b_name, model_label)

        fold_a = fold_results_dict[key_a]
        fold_b = fold_results_dict[key_b]

        macro_test = safe_wilcoxon(
            fold_a["macro_f1"].values,
            fold_b["macro_f1"].values
        )
        kappa_test = safe_wilcoxon(
            fold_a["kappa"].values,
            fold_b["kappa"].values
        )

        rows.append({
            "model": model_label,
            "comparison": f"{a_name} vs {b_name}",
            "metric": "macro_f1",
            "statistic": macro_test["statistic"],
            "p_value": macro_test["p_value"],
        })

        rows.append({
            "model": model_label,
            "comparison": f"{a_name} vs {b_name}",
            "metric": "kappa",
            "statistic": kappa_test["statistic"],
            "p_value": kappa_test["p_value"],
        })

    return pd.DataFrame(rows)


def run_all_experiments():
    stat, spec, comp, y, groups = load_data()
    feature_sets = get_feature_sets(stat, spec, comp)

    summary_rows = []
    fold_rows_all = []
    fold_results_dict = {}
    best_result = None

    print_header("RUNNING EXPERIMENTS")

    for feature_name, X in feature_sets.items():
        print(f"\nEvaluating: {feature_name}")

        for model_name, model_label in [("logreg", "LogisticRegression"), ("rf", "RandomForest")]:
            res = evaluate_model(X, y, groups, model_name=model_name)

            row = {
                "feature_set": feature_name,
                "model": model_label,
                "accuracy_mean": res["accuracy_mean"],
                "accuracy_std": res["accuracy_std"],
                "macro_f1_mean": res["macro_f1_mean"],
                "macro_f1_std": res["macro_f1_std"],
                "balanced_acc_mean": res["balanced_acc_mean"],
                "balanced_acc_std": res["balanced_acc_std"],
                "kappa_mean": res["kappa_mean"],
                "kappa_std": res["kappa_std"],
            }
            summary_rows.append(row)

            fold_df = res["fold_df"].copy()
            fold_df["feature_set"] = feature_name
            fold_df["model"] = model_label
            fold_rows_all.append(fold_df)

            fold_results_dict[(feature_name, model_label)] = res["fold_df"]

            print(
                f"  {model_label} -> "
                f"F1: {res['macro_f1_mean']:.4f} ± {res['macro_f1_std']:.4f}, "
                f"Kappa: {res['kappa_mean']:.4f} ± {res['kappa_std']:.4f}"
            )

            if best_result is None or res["macro_f1_mean"] > best_result["macro_f1_mean"]:
                best_result = {
                    "feature_set": feature_name,
                    "model": model_label,
                    "macro_f1_mean": res["macro_f1_mean"],
                    "macro_f1_std": res["macro_f1_std"],
                    "kappa_mean": res["kappa_mean"],
                    "kappa_std": res["kappa_std"],
                    "y_true": res["y_true"],
                    "y_pred": res["y_pred"],
                    "X_data": X,
                    "model_name_internal": model_name,
                }

    summary_df = pd.DataFrame(summary_rows)
    fold_results_df = pd.concat(fold_rows_all, ignore_index=True)

    # Save core results
    summary_df.to_csv(OUTPUT_DIR / "model_results_summary.csv", index=False)
    fold_results_df.to_csv(OUTPUT_DIR / "model_results_folds.csv", index=False)

    # Backward-compatible file for UI/plots
    backward_df = summary_df.rename(columns={
        "accuracy_mean": "accuracy",
        "macro_f1_mean": "macro_f1",
        "balanced_acc_mean": "balanced_acc",
        "kappa_mean": "kappa",
    })[
        ["feature_set", "model", "accuracy", "macro_f1", "balanced_acc", "kappa"]
    ]
    backward_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)

    # Save best predictions
    np.save(OUTPUT_DIR / "best_y_true.npy", best_result["y_true"])
    np.save(OUTPUT_DIR / "best_y_pred.npy", best_result["y_pred"])

    best_meta = pd.DataFrame([{
        "feature_set": best_result["feature_set"],
        "model": best_result["model"],
        "macro_f1_mean": best_result["macro_f1_mean"],
        "macro_f1_std": best_result["macro_f1_std"],
        "kappa_mean": best_result["kappa_mean"],
        "kappa_std": best_result["kappa_std"],
    }])
    best_meta.to_csv(OUTPUT_DIR / "best_model_summary.csv", index=False)

    # Save best model fitted on full data
    final_model = build_model(best_result["model_name_internal"])
    final_model.fit(best_result["X_data"], y)
    joblib.dump(final_model, OUTPUT_DIR / "best_model.joblib")

    # Per-class F1 for best model
    per_class_df = compute_per_class_f1(best_result["y_true"], best_result["y_pred"])
    per_class_df.to_csv(OUTPUT_DIR / "best_model_per_class_f1.csv", index=False)

    # Wilcoxon tests
    wilcoxon_df = run_wilcoxon_tests(fold_results_dict)
    wilcoxon_df.to_csv(OUTPUT_DIR / "wilcoxon_tests.csv", index=False)

    print_header("FINAL SUMMARY RESULTS")
    print(summary_df)

    print_header("BEST MODEL")
    print(best_meta)

    print_header("PER-CLASS F1 (BEST MODEL)")
    print(per_class_df)

    print_header("WILCOXON TESTS")
    print(wilcoxon_df)

    return summary_df, fold_results_df, per_class_df, wilcoxon_df


def main():
    summary_df, fold_results_df, per_class_df, wilcoxon_df = run_all_experiments()

    print_header("TRAINING COMPLETE")
    print("\nSaved files:")
    print(f"- {OUTPUT_DIR / 'model_results_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'model_results_folds.csv'}")
    print(f"- {OUTPUT_DIR / 'best_model_per_class_f1.csv'}")
    print(f"- {OUTPUT_DIR / 'wilcoxon_tests.csv'}")


if __name__ == "__main__":
    main()