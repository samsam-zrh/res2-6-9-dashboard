import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rs_project import run_pipeline


if __name__ == "__main__":
    summary = run_pipeline()
    print("Pipeline termine.")
    print(summary)
