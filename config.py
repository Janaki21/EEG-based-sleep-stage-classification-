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

SUBJECTS = list(range(3))
RECORDING_NIGHTS = [1]

EEG_CHANNEL = "EEG Fpz-Cz"
EPOCH_DURATION = 30.0
LOW_FREQ = 0.5
HIGH_FREQ = 30.0
RANDOM_STATE = 42
N_SPLITS = 3

SLEEP_STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

STAGE_NAMES = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}