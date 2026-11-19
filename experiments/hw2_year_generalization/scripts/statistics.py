from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from .common import ROOT, cfg, classify, delta_attr, offline, parse, read_jsonl, run_dir, sha, verify_lock, write_json, write_jsonl

CATEGORIES = ("ABSTAIN", "DONOR_MATCH", "OTHER_IN_CONTEXT", "CONTEXT_ABSENT", "INVALID")


def csv_file(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def estimate(vector, indices):
    vector = np.asarray(vector, dtype=float)
    draws = vector[indices].mean(axis=1)
    lo, hi = np.quantile(draws, [.025, .975])
    return {"estimate": float(vector.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}


def base_effects(rows, templates=None, orders=None):
    rows = [r for r in rows if r["family"] == "MISSING" and
            (templates is None or r["template"] in templates) and (orders is None or r["order"] in orders)]
    bases = sorted({r["base_id"] for r in rows})
    by_base = defaultdict(list)
    for r in rows:
        by_base[r["base_id"]].append(r)
    output = []
    for bid in bases:
        rs = by_base[bid]
        pairs = defaultdict(dict)
        for r in rs:
            pairs[(r["template"], r["order"], r["placement"])][r["assignment"]] = r
        attr, forward, reverse = [], [], []
        for pair in pairs.values():
            a, b = pair["S1"], pair["S2"]
            attr.append(delta_attr(a["parse"], b["parse"], a["v1"], a["v2"]))
            av = a["parse"]["normalized_value"] if a["parse"]["valid"] else None
            bv = b["parse"]["normalized_value"] if b["parse"]["valid"] else None
            forward.append(av == str(a["v1"]) and bv == str(a["v2"]))
            reverse.append(av == str(a["v2"]) and bv == str(a["v1"]))
        effect = {"base_id": bid, "delta_attr": float(np.mean(attr)),
                  "forward_switch": float(np.mean(forward)), "reverse_switch": float(np.mean(reverse))}
        for placement in ("MIDDLE", "ADJACENT"):
            subset = [r for r in rs if r["placement"] == placement]
            for short, category in (("C", "DONOR_MATCH"), ("U", "CONTEXT_ABSENT"), ("I", "INVALID"), ("A", "ABSTAIN")):
                effect[f"{short}_{placement}"] = float(np.mean([r["category"] == category for r in subset]))
        effect["delta_C"] = effect["C_ADJACENT"] - effect["C_MIDDLE"]
        effect["delta_U"] = effect["U_ADJACENT"] - effect["U_MIDDLE"]
        effect["delta_type"] = effect["delta_C"] - effect["delta_U"]
        output.append(effect)
    return output

