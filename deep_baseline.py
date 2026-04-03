from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)

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


def get_device():
    # MPS is fine here now because we avoid the adaptive-pool shape issue
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SimpleCNN(nn.Module):
    def __init__(self, input_length, n_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # dynamically compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            out = self.features(dummy)
            flattened_dim = out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def zscore_epochs(X):
    X = X.astype(np.float32)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std


def make_class_weights(y):
    counts = np.bincount(y, minlength=len(LABEL_ORDER)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_one_fold(model, train_loader, val_loader, device, class_weights, epochs=6, lr=1e-3):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_val_f1 = -np.inf

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            preds = model(xb)
            loss = criterion(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_true, val_pred = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                val_pred.extend(preds.tolist())
                val_true.extend(yb.numpy().tolist())

        val_f1 = f1_score(val_true, val_pred, average="macro", zero_division=0)
        mean_train_loss = float(np.mean(train_losses)) if len(train_losses) > 0 else np.nan

        print(
            f"    Epoch {epoch + 1}/{epochs} | "
            f"train_loss={mean_train_loss:.4f} | val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_fold_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def run_grouped_cnn_cv(X, y, groups, epochs=6, batch_size=64):
    device = get_device()
    print(f"Using device: {device}")

    sgkf = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows = []
    y_pred_fullorder = np.full_like(y, fill_value=-1)
    input_length = X.shape[1]

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), start=1):
        print(f"\nFold {fold_idx}/{N_SPLITS}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        y_test_t = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = SimpleCNN(input_length=input_length, n_classes=len(LABEL_ORDER)).to(device)
        class_weights = make_class_weights(y_train)

        model = train_one_fold(
            model,
            train_loader,
            test_loader,
            device,
            class_weights,
            epochs=epochs,
            lr=1e-3,
        )

        model.eval()
        all_true, all_pred = [], []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_pred.extend(preds.tolist())
                all_true.extend(yb.numpy().tolist())

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)

        metrics = evaluate_fold_metrics(all_true, all_pred)
        metrics["fold"] = fold_idx
        metrics["n_test_samples"] = len(test_idx)
        fold_rows.append(metrics)

        y_pred_fullorder[test_idx] = all_pred

        print(
            f"  Fold result -> "
            f"F1: {metrics['macro_f1']:.4f}, "
            f"Kappa: {metrics['kappa']:.4f}, "
            f"Balanced Acc: {metrics['balanced_acc']:.4f}"
        )

    fold_df = pd.DataFrame(fold_rows)

    precision, recall, f1, support = precision_recall_fscore_support(
        y,
        y_pred_fullorder,
        labels=LABEL_ORDER,
        zero_division=0,
    )

    per_class_rows = []
    for idx, label in enumerate(LABEL_ORDER):
        per_class_rows.append(
            {
                "stage_id": label,
                "stage_name": STAGE_NAMES[label],
                "precision": precision[idx],
                "recall": recall[idx],
                "f1_score": f1[idx],
                "support": int(support[idx]),
            }
        )

    per_class_df = pd.DataFrame(per_class_rows)

    summary_df = pd.DataFrame(
        [
            {
                "model": "SimpleCNN",
                "accuracy_mean": fold_df["accuracy"].mean(),
                "accuracy_std": fold_df["accuracy"].std(ddof=1),
                "macro_f1_mean": fold_df["macro_f1"].mean(),
                "macro_f1_std": fold_df["macro_f1"].std(ddof=1),
                "balanced_acc_mean": fold_df["balanced_acc"].mean(),
                "balanced_acc_std": fold_df["balanced_acc"].std(ddof=1),
                "kappa_mean": fold_df["kappa"].mean(),
                "kappa_std": fold_df["kappa"].std(ddof=1),
            }
        ]
    )

    return {
        "fold_df": fold_df,
        "per_class_df": per_class_df,
        "summary_df": summary_df,
        "y_pred_fullorder": y_pred_fullorder,
    }


def main():
    print_header("DEEP BASELINE: SIMPLE CNN")

    X = np.load(OUTPUT_DIR / "X_epochs.npy")
    y = np.load(OUTPUT_DIR / "y_labels.npy")
    groups = np.load(OUTPUT_DIR / "subject_ids.npy")

    X = zscore_epochs(X)

    results = run_grouped_cnn_cv(X, y, groups, epochs=6, batch_size=64)

    results["fold_df"].to_csv(OUTPUT_DIR / "deep_baseline_fold_results.csv", index=False)
    results["per_class_df"].to_csv(OUTPUT_DIR / "deep_baseline_per_class_metrics.csv", index=False)
    results["summary_df"].to_csv(OUTPUT_DIR / "deep_baseline_summary.csv", index=False)
    np.save(OUTPUT_DIR / "deep_baseline_y_pred.npy", results["y_pred_fullorder"])

    print_header("DEEP BASELINE SUMMARY")
    print(results["summary_df"])

    print_header("DEEP BASELINE PER-CLASS METRICS")
    print(results["per_class_df"])

    print_header("DEEP BASELINE COMPLETE")
    print(f"Saved: {OUTPUT_DIR / 'deep_baseline_fold_results.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'deep_baseline_per_class_metrics.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'deep_baseline_summary.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'deep_baseline_y_pred.npy'}")


if __name__ == "__main__":
    main()