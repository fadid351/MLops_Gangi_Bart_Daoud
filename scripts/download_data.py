"""Télécharge adult.data et adult.test dans data/raw/.

Usage :
    python scripts/download_data.py            # ne retélécharge pas si déjà présents
    python scripts/download_data.py --force    # force le téléchargement
"""

import argparse
import sys
from pathlib import Path

# Rend le package src importable quand on lance `python scripts/download_data.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_RAW  # noqa: E402
from src.data.load_data import download_data  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="retélécharger même si présent")
    args = parser.parse_args()
    for path in download_data(DATA_RAW, overwrite=args.force):
        print(f"OK : {path}")
