import re
import numpy as np
import pandas as pd
import mne

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    EEG_CHANNEL,
    EPOCH_DURATION,
    LOW_FREQ,
    HIGH_FREQ,
    SLEEP_STAGE_MAP,
    STAGE_NAMES,
)
from utils import load_json, print_header


def extract_subject_id_from_path(psg_path: str) -> str:
    name = psg_path.split("/")[-1]
    match = re.match(r"([A-Za-z0-9]+)-PSG\.edf", name)
    if match:
        return match.group(1)
    return name.replace(".edf", "")


def load_manifest():
    manifest_path = DATA_DIR / "sleep_edf_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return load_json(manifest_path)


def resolve_eeg_channel(raw, requested_channel):
    if requested_channel in raw.ch_names:
        return requested_channel

    normalized = {ch.replace("EEG ", "").strip(): ch for ch in raw.ch_names}
    requested_norm = requested_channel.replace("EEG ", "").strip()

    if requested_norm in normalized:
        resolved = normalized[requested_norm]
        print(f"Resolved EEG channel '{requested_channel}' -> '{resolved}'")
        return resolved

    raise ValueError(
        f"Requested channel '{requested_channel}' not found. Available channels: {raw.ch_names}"
    )


def preprocess_single_record(psg_path, hyp_path):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    annot = mne.read_annotations(hyp_path)
    raw.set_annotations(annot)

    selected_channel = resolve_eeg_channel(raw, EEG_CHANNEL)
    raw.pick([selected_channel])

    raw.filter(LOW_FREQ, HIGH_FREQ, fir_design="firwin", verbose=False)

    valid_annotations = []
    for onset, duration, desc in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description
    ):
        if desc in SLEEP_STAGE_MAP:
            valid_annotations.append((onset, duration, desc))

    if len(valid_annotations) == 0:
        return None

    onset, duration, description = zip(*valid_annotations)
    filtered_annot = mne.Annotations(
        onset=list(onset),
        duration=list(duration),
        description=list(description),
        orig_time=raw.annotations.orig_time,
    )
    raw.set_annotations(filtered_annot)

    events, event_id = mne.events_from_annotations(
        raw,
        event_id=SLEEP_STAGE_MAP,
        chunk_duration=EPOCH_DURATION,
        verbose=False,
    )

    if len(events) == 0:
        return None

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=0.0,
        tmax=EPOCH_DURATION - 1.0 / raw.info["sfreq"],
        baseline=None,
        preload=True,
        verbose=False,
    )

    X = epochs.get_data(copy=True)[:, 0, :]
    y = epochs.events[:, 2]

    return X, y, raw.info["sfreq"]


def build_dataset():
    manifest = load_manifest()

    all_X = []
    all_y = []
    subject_ids = []
    rows = []
    sfreq_ref = None

    print_header("PREPROCESSING ALL RECORDS")

    for i, item in enumerate(manifest):
        psg_path = item["psg_path"]
        hyp_path = item["hypnogram_path"]
        subject_id = extract_subject_id_from_path(psg_path)

        print(f"[{i+1}/{len(manifest)}] Processing {subject_id}")

        result = preprocess_single_record(psg_path, hyp_path)
        if result is None:
            print(f"  Skipped {subject_id} (no valid epochs)")
            continue

        X, y, sfreq = result

        if sfreq_ref is None:
            sfreq_ref = sfreq
        elif sfreq != sfreq_ref:
            raise ValueError(f"Sampling rate mismatch: {sfreq} vs {sfreq_ref}")

        all_X.append(X)
        all_y.append(y)
        subject_ids.extend([subject_id] * len(y))

        stage_counts = pd.Series(y).value_counts().to_dict()
        rows.append(
            {
                "subject_id": subject_id,
                "n_epochs": len(y),
                "stage_counts": str(stage_counts),
            }
        )

        print(f"  epochs={len(y)}, shape={X.shape}")

    if len(all_X) == 0:
        raise ValueError("No valid epochs were created.")

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    subject_ids = np.array(subject_ids)

    summary_df = pd.DataFrame(rows)

    np.save(OUTPUT_DIR / "X_epochs.npy", X_all)
    np.save(OUTPUT_DIR / "y_labels.npy", y_all)
    np.save(OUTPUT_DIR / "subject_ids.npy", subject_ids)

    summary_df.to_csv(OUTPUT_DIR / "epoch_summary.csv", index=False)

    metadata = {
        "n_samples": int(len(y_all)),
        "n_timepoints_per_epoch": int(X_all.shape[1]),
        "sampling_frequency": float(sfreq_ref),
        "class_counts": {
            STAGE_NAMES[int(k)]: int(v)
            for k, v in pd.Series(y_all).value_counts().sort_index().items()
        },
    }

    pd.DataFrame(
        [{"stage_id": k, "stage_name": v} for k, v in STAGE_NAMES.items()]
    ).to_csv(OUTPUT_DIR / "label_map.csv", index=False)

    print_header("DATASET BUILT")
    print(f"X shape: {X_all.shape}")
    print(f"y shape: {y_all.shape}")
    print(f"Unique subjects: {len(np.unique(subject_ids))}")
    print("Class counts:")
    for k, v in metadata["class_counts"].items():
        print(f"  {k}: {v}")

    return metadata


if __name__ == "__main__":
    build_dataset()