"""
Fetch the real Kaggle dataset for training in CI.

Authenticates with Kaggle via KAGGLE_USERNAME / KAGGLE_KEY env vars
(automatically picked up by kagglehub), downloads the dataset, and copies
the CSV into ./dataset/ so PHISHGUARD_RAW_DATA_FILE resolves correctly.

This replaces the GitHub Release dependency, which was unreliable to
upload from consumer links.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("Installing kagglehub...", flush=True)
    os.system(f"{sys.executable} -m pip install kagglehub")
    import kagglehub

DATASET_HANDLE = "sarahhamer/malicious-and-benign-urls-2010-2026"
TARGET_DIR = Path("dataset")
TARGET_NAME = "malicious_benign_URL_dataset_v2.csv"

TARGET_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading {DATASET_HANDLE} ...", flush=True)
cache_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
print(f"Dataset cached at: {cache_path}", flush=True)

candidates = list(cache_path.rglob("*.csv"))
if not candidates:
    print(f"No CSV found under {cache_path}", file=sys.stderr)
    sys.exit(1)

source_csv = max(candidates, key=lambda p: p.stat().st_size)
print(f"Using CSV: {source_csv} ({source_csv.stat().st_size / 1_000_000:.1f} MB)", flush=True)

target = TARGET_DIR / TARGET_NAME
shutil.copy2(source_csv, target)
print(f"Copied to: {target}", flush=True)