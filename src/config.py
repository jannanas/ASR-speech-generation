from pathlib import Path

PIPELINE_VERSION = "2"
N_MFCC = 13

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / ".cache" / PIPELINE_VERSION
OUTPUT_DIR = ROOT_DIR / "output"
SAMPLE_RATE = 16000
