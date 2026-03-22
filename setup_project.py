from pathlib import Path
import textwrap

# Create everything relative to the folder where this script is saved
PROJECT_ROOT = Path(__file__).resolve().parent

FILES = {
    "requirements.txt": """
mne
numpy
pandas
scipy
scikit-learn
matplotlib
joblib
antropy
pyedflib
""",

    "config.py": """
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"
CACHE_DIR = OUTPUT_DIR / "cache"

for path in [DATA_DIR, OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, CACHE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

SUBJECTS = list(range(10))   # start small
RECORDING_NIGHTS = [1]
EEG_CHANNEL = "Fpz-Cz"
RANDOM_STATE = 42
N_SPLITS = 5
""",

    "utils.py": """
from pathlib import Path
import json

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def print_header(title: str):
    line = "=" * 80
    print(f"\\n{line}\\n{title}\\n{line}")
""",

    "download_data.py": """
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

    print(f"\\nDownloaded / resolved {len(records)} subject-night pairs.")
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
""",

    "inspect_data.py": """
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
""",

    "run_phase0_phase1.py": """
from utils import print_header
import download_data
import inspect_data

def main():
    print_header("PHASE 0 + PHASE 1")
    download_data.main()
    inspect_data.inspect_first_record()
    print_header("DONE")
    print("Phase 0 and Phase 1 completed successfully.")

if __name__ == "__main__":
    main()
"""
}

FOLDERS = [
    "data",
    "outputs",
    "outputs/figures",
    "outputs/tables",
    "outputs/models",
    "outputs/cache",
]

def main():
    print(f"Creating project in: {PROJECT_ROOT}")

    for folder in FOLDERS:
        folder_path = PROJECT_ROOT / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {folder_path}")

    for filename, content in FILES.items():
        file_path = PROJECT_ROOT / filename
        file_path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
        print(f"Created file: {file_path}")

    print("\\nSETUP COMPLETE")
    print("Files now present:")
    for p in sorted(PROJECT_ROOT.iterdir()):
        print(f" - {p.name}")

    print("\\nNext commands to run:")
    print("python -m pip install -r requirements.txt")
    print("python run_phase0_phase1.py")

if __name__ == "__main__":
    main()