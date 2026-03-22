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

from config import OUTPUT_DIR, RANDOM_STATE, N_SPLITS
from utils import print_header


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
    return RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    )


def evaluate_model(X, y, groups, model_name="logreg"):
    sgkf = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    accs, f1s, bals, kappas = [], [], [], []
    all_true, all_pred = [], []

    for train_idx, test_idx in sgkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_model(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="macro"))
        bals.append(balanced_accuracy_score(y_test, y_pred))
        kappas.append(cohen_kappa_score(y_test, y_pred))

        all_true.extend(y_test)
        all_pred.extend(y_pred)

    return {
        "accuracy": np.mean(accs),
        "macro_f1": np.mean(f1s),
        "balanced_acc": np.mean(bals),
        "kappa": np.mean(kappas),
        "y_true": np.array(all_true),
        "y_pred": np.array(all_pred),
    }


def run_all_experiments():
    stat, spec, comp, y, groups = load_data()
    feature_sets = get_feature_sets(stat, spec, comp)

    results = []
    best_result = None

    print_header("RUNNING EXPERIMENTS")

    for name, X in feature_sets.items():
        print(f"\nEvaluating: {name}")

        for model_name, model_label in [("logreg", "LogisticRegression"), ("rf", "RandomForest")]:
            res = evaluate_model(X, y, groups, model_name=model_name)

            row = {
                "feature_set": name,
                "model": model_label,
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
                "balanced_acc": res["balanced_acc"],
                "kappa": res["kappa"],
            }
            results.append(row)

            print(f"  {model_label} -> F1: {res['macro_f1']:.4f}, Kappa: {res['kappa']:.4f}")

            if best_result is None or res["macro_f1"] > best_result["macro_f1"]:
                best_result = {
                    "feature_set": name,
                    "model": model_label,
                    "macro_f1": res["macro_f1"],
                    "kappa": res["kappa"],
                    "y_true": res["y_true"],
                    "y_pred": res["y_pred"],
                    "X_data": X,
                }

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "model_results.csv", index=False)

    np.save(OUTPUT_DIR / "best_y_true.npy", best_result["y_true"])
    np.save(OUTPUT_DIR / "best_y_pred.npy", best_result["y_pred"])

    best_meta = pd.DataFrame([{
        "feature_set": best_result["feature_set"],
        "model": best_result["model"],
        "macro_f1": best_result["macro_f1"],
        "kappa": best_result["kappa"],
    }])
    best_meta.to_csv(OUTPUT_DIR / "best_model_summary.csv", index=False)

    final_model = build_model("logreg" if best_result["model"] == "LogisticRegression" else "rf")
    final_model.fit(best_result["X_data"], y)
    joblib.dump(final_model, OUTPUT_DIR / "best_model.joblib")

    print_header("FINAL RESULTS")
    print(results_df)

    print_header("BEST MODEL")
    print(best_meta)

    return results_df


def main():
    results = run_all_experiments()
    print_header("TRAINING COMPLETE")
    print(results.head())


if __name__ == "__main__":
    main()