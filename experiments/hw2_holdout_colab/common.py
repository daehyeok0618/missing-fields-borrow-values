"""Atomic artifacts and a finite source whitelist."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def source_paths():
    paths = [p for ext in ("*.py", "*.md", "*.txt") for p in HERE.glob(ext)]
    paths += list((HERE / "source").glob("*.json")) + list((HERE / "source").glob("*.jsonl"))
    paths += list((HERE / "tests").glob("*.py"))
    paths += [p for p in (HERE / "parent_source").rglob("*") if p.is_file()
              and p.suffix in (".py", ".json", ".jsonl", ".md", ".txt") and "__pycache__" not in p.parts]
    paths += list((HERE / "source").glob("*.txt"))
    return sorted(p for p in paths if p.is_file())


def source_hashes():
    return {str(p.relative_to(HERE)): file_hash(p) for p in source_paths()}
