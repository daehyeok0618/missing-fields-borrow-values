from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def cfg():
    return json.loads((ROOT / "config.json").read_text())


def run_dir():
    return ROOT / "runs" / cfg()["run_id"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def write_jsonl(path, rows, append=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a" if append else "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def offline():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HOME"] = str(REPO / ".cache/huggingface")
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")


def tokenizer():
    from transformers import AutoTokenizer
    offline()
    m = cfg()["model"]
    return AutoTokenizer.from_pretrained(m["id"], revision=m["revision"],
                                        cache_dir=m["cache_dir"], local_files_only=True)


def render(messages, tok):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def normalize(value):
    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(c for c in value if unicodedata.category(c)[0] not in "PS")
    return " ".join(value.split())


def occurs(value, context):
    v, c = normalize(value), normalize(context)
    return bool(v and re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", c))


def parse(raw):
    def pairs(items):
        obj = {}
        for k, v in items:
            if k in obj:
                raise ValueError("duplicate_key")
            obj[k] = v
        return obj
    try:
        obj = json.loads(raw.strip(), object_pairs_hook=pairs,
                         parse_constant=lambda s: (_ for _ in ()).throw(ValueError("non_json_constant")))
        if not isinstance(obj, dict) or set(obj) != {"value"}:
            raise ValueError("schema")
        v = obj["value"]
        if v is not None and (not isinstance(v, str) or not normalize(v)):
            raise ValueError("value_type_or_empty")
        return {"valid": True, "value": v, "normalized_value": normalize(v) if v is not None else None, "error": None}
    except (ValueError, TypeError) as exc:
        return {"valid": False, "value": None, "normalized_value": None,
                "error": "malformed_json" if isinstance(exc, json.JSONDecodeError) else str(exc)}


def classify(request, parsed):
    if not parsed["valid"]:
        return "INVALID"
    v = parsed["normalized_value"]
    if v is None:
        return "ABSTAIN"
    if request["state"] == "PRESENT" and v == str(request["target_value"]):
        return "TARGET_CORRECT"
    if request["family"] not in ("T0", "T1") and v in (str(request["v1"]), str(request["v2"])):
        return "DONOR_MATCH"
    context = "\n".join(m["content"] for m in request["messages"])
    return "OTHER_IN_CONTEXT" if occurs(parsed["value"], context) else "CONTEXT_ABSENT"


def delta_attr(s1, s2, v1, v2):
    a = s1["normalized_value"] if s1["valid"] else None
    b = s2["normalized_value"] if s2["valid"] else None
    return 0.5 * ((a == str(v1)) - (b == str(v1)) + (b == str(v2)) - (a == str(v2)))


def semantic_parse(parsed):
    """Fixed alias list, defined before inference; never repairs invalid JSON."""
    if parsed["valid"] and parsed["normalized_value"] in cfg()["analysis"]["abstention_aliases"]:
        return {**parsed, "value": None, "normalized_value": None}
    return dict(parsed)


def verify_lock():
    lock = json.loads((run_dir() / "protocol_lock.json").read_text())
    mismatches = [p for p, h in lock["hashes"].items() if not (ROOT / p).exists() or sha(ROOT / p) != h]
    if mismatches:
        raise RuntimeError({"protocol_changed": mismatches})
    return lock
