import numpy as np
import pandas as pd
from config import OUTPUT_DIR

def main():
    y = np.load(OUTPUT_DIR / "y_labels.npy")
    subjects = np.load(OUTPUT_DIR / "subject_ids.npy")

    df = pd.DataFrame({
        "label": y,
        "subject": subjects
    })

    summary = {
        "total_epochs": len(y),
        "unique_subjects": df["subject"].nunique(),
    }

    stage_counts = df["label"].value_counts().sort_index()

    with open(OUTPUT_DIR / "dataset_report.txt", "w") as f:
        f.write(str(summary) + "\n\n")
        f.write("Stage distribution:\n")
        f.write(stage_counts.to_string())

    print("Dataset report saved.")

if __name__ == "__main__":
    main()