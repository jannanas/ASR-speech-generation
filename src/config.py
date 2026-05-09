from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.parent / "data"

PIPELINE_VERSION = "1"
CACHE_DIR = DATA_DIR / ".cache" / PIPELINE_VERSION

N_MFCC = 13