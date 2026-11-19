"""Decompress the request files and check every protocol-locked file against its lock.

Run once from the repository root before any analysis command:
    python restore_data.py
"""
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = ["hw2_missing_fields", "hw2_year_generalization"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


ok = True
for name in TASKS:
    root = ROOT / "experiments" / name
    run_id = json.loads((root / "config.json").read_text())["run_id"]
    lock = json.loads((root / "runs" / run_id / "protocol_lock.json").read_text())["hashes"]
    for gz in sorted((root / "data").glob("*.jsonl.gz")):
        out = gz.with_suffix("")
        if not out.exists() or sha(out) != lock[f"data/{out.name}"]:
            with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
                shutil.copyfileobj(fi, fo)
    bad = [rel for rel, h in lock.items() if not (root / rel).exists() or sha(root / rel) != h]
    print(f"{name}: {len(lock) - len(bad)}/{len(lock)} locked files match {run_id}/protocol_lock.json")
    if bad:
        ok = False
        print("  mismatch:", bad)
sys.exit(0 if ok else 1)
