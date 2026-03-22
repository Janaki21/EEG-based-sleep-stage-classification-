from pathlib import Path
from mne.datasets.sleep_physionet.age import fetch_data

from config import DATA_DIR, SUBJECTS, RECORDING_NIGHTS
from utils import save_json, print_header

def download_sleep_edf(subjects=None, recording_nights=None):
    subjects = SUBJECTS if subjects is None else subjects
    recording_nights = RECORDING_NIGHTS if recording_nights is None else recording_nights

    print_header("DOWNLOADING SLEEP-EDF DATASET")
    print(f"Subjects: {subjects}")
    print(f"Recording nights: {recording_nights}")
    print(f"Download directory: {DATA_DIR.resolve()}")

    records = fetch_data(
        subjects=subjects,
        recording=recording_nights,
        path=str(DATA_DIR),
        on_missing="warn",
    )

    print(f"\nDownloaded / resolved {len(records)} subject-night pairs.")
    return records

def build_manifest(records):
    manifest = []
    for pair in records:
        if len(pair) != 2:
            continue
        psg_path, hyp_path = pair
        manifest.append(
            {
                "psg_path": str(Path(psg_path).resolve()),
                "hypnogram_path": str(Path(hyp_path).resolve()),
                "psg_exists": Path(psg_path).exists(),
                "hyp_exists": Path(hyp_path).exists(),
            }
        )
    return manifest

def main():
    records = download_sleep_edf()
    manifest = build_manifest(records)

    manifest_path = DATA_DIR / "sleep_edf_manifest.json"
    save_json(manifest, manifest_path)

    print_header("DOWNLOAD SUMMARY")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Entries: {len(manifest)}")

    for i, item in enumerate(manifest[:5]):
        print(f"[{i}] PSG: {item['psg_path']}")
        print(f"    HYP: {item['hypnogram_path']}")
        print(f"    OK : PSG={item['psg_exists']} HYP={item['hyp_exists']}")

if __name__ == "__main__":
    main()
