import mne

from config import DATA_DIR, EEG_CHANNEL
from utils import load_json, print_header

def inspect_first_record():
    manifest_path = DATA_DIR / "sleep_edf_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run download_data.py first."
        )

    manifest = load_json(manifest_path)

    if not manifest:
        raise ValueError("Manifest is empty.")

    first = manifest[0]
    psg_path = first["psg_path"]
    hyp_path = first["hypnogram_path"]

    print_header("READING FIRST PSG FILE")
    print(f"PSG file: {psg_path}")
    print(f"Hypnogram file: {hyp_path}")

    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
    annot = mne.read_annotations(hyp_path)

    print_header("BASIC RAW INFO")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Sampling frequency: {raw.info['sfreq']}")
    print(f"Duration (sec): {raw.times[-1]:.2f}")
    print("Available channels:")
    for ch in raw.ch_names:
        print(f" - {ch}")

    print_header("ANNOTATION SUMMARY")
    print(f"Number of annotations: {len(annot)}")
    unique_desc = sorted(set(annot.description))
    print("Unique annotation labels:")
    for desc in unique_desc:
        print(f" - {desc}")

    print_header("CHANNEL CHECK")
    if EEG_CHANNEL in raw.ch_names:
        print(f"Target EEG channel '{EEG_CHANNEL}' found.")
    else:
        print(f"Target EEG channel '{EEG_CHANNEL}' NOT found.")

    raw.set_annotations(annot)

    print_header("FIRST FEW ANNOTATIONS")
    for i in range(min(10, len(raw.annotations))):
        ann = raw.annotations[i]
        print(
            f"[{i}] onset={ann['onset']:.2f}s duration={ann['duration']:.2f}s desc={ann['description']}"
        )

if __name__ == "__main__":
    inspect_first_record()
