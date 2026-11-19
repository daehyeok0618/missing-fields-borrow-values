"""CPU-only summaries; strict decisions and fail-closed service outputs stay separate."""
import argparse
from collections import Counter
import csv
import json
import random
from pathlib import Path

from . import core
from .common import digest, file_hash, read_jsonl, write_json
from .protocol import PROTOCOL, GROUPS

METHODS = tuple(PROTOCOL["methods"])
FIELDS = ("n", "correct", "missing", "falsefill", "present", "present_correct",
          "std_correct_present", "present_loss", "loss_to_null", "loss_to_wrong_value",
          "invalid", "answered", "answered_wrong")
RATIOS = {"accuracy": ("correct", "n"), "missing_falsefill_rate": ("falsefill", "missing"),
          "present_accuracy": ("present_correct", "present"),
          "present_loss_among_std_correct": ("present_loss", "std_correct_present"),
          "invalid_rate": ("invalid", "n"), "coverage": ("answered", "n"),
          "selective_error": ("answered_wrong", "answered")}


def correct(parsed, label, policy):
    if not parsed["valid"]:
        return False
    if label["value"] is None:
        return core.abstains(parsed, policy)
    return core.canonical_number(parsed["value"]) == str(label["value"])


def indicators(record, label, method, policy, mode):
    p = record["methods" if mode == "decision" else "served_methods"][method]
    is_missing = label["value"] is None
    right = correct(p, label, policy)
    abstain = core.abstains(p, policy)
    answered = bool(p["valid"] and not abstain)
    standard_correct = bool(not is_missing and correct(record["methods"]["standard"], label, policy))
    lost = standard_correct and not right
    return {"n": 1, "correct": int(right), "missing": int(is_missing),
            "falsefill": int(is_missing and answered), "present": int(not is_missing),
            "present_correct": int(not is_missing and right), "std_correct_present": int(standard_correct),
            "present_loss": int(lost), "loss_to_null": int(lost and abstain),
            "loss_to_wrong_value": int(lost and answered), "invalid": int(not p["valid"]),
            "answered": int(answered), "answered_wrong": int(answered and not right)}


def weights_for(clusters, repetitions, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    weights = np.zeros((repetitions, len(clusters)), dtype=int)
    for task in sorted({t for t, _ in clusters}):
        indices = [i for i, (t, _) in enumerate(clusters) if t == task]
        weights[:, indices] = rng.multinomial(len(indices), [1 / len(indices)] * len(indices), size=repetitions)
    return weights


def summarize(records, gold, task, group, repetitions, mode="decision", policy="semantic"):
    import numpy as np
    clusters = sorted({(r["task"], r["base_id"]) for r in records})
    lookup = {key: i for i, key in enumerate(clusters)}
    seed = PROTOCOL["bootstrap_seed"] ^ int(digest([task, group, mode, policy])[:8], 16)
    weights = weights_for(clusters, repetitions, seed)
    metric_rows, zero_bounds, all_draws, aggregates = [], [], {}, {}
    for method in METHODS:
        counts = np.zeros((len(clusters), len(FIELDS)), dtype=int)
        for r in records:
            flags = indicators(r, gold[r["id"]], method, policy, mode)
            counts[lookup[(r["task"], r["base_id"])]] += [flags[f] for f in FIELDS]
        total = counts.sum(axis=0)
        resampled = weights @ counts
        aggregates[method] = dict(zip(FIELDS, map(int, total)))
        for metric, (num, den) in RATIOS.items():
            ni, di = FIELDS.index(num), FIELDS.index(den)
            value = float(total[ni] / total[di]) if total[di] else None
            draws = np.divide(resampled[:, ni], resampled[:, di], out=np.full(repetitions, np.nan), where=resampled[:, di] != 0)
            finite = draws[np.isfinite(draws)]
            ci = list(map(float, np.percentile(finite, [2.5, 97.5]))) if len(finite) else [None, None]
            all_draws[(method, metric)] = draws
            metric_rows.append({"task": task, "group": group, "mode": mode, "policy": policy,
                "method": method, "metric": metric, "n_base": len(clusters), "n": len(records),
                "numerator": int(total[ni]), "denominator": int(total[di]), "estimate": value,
                "ci_low": ci[0], "ci_high": ci[1]})
        for event, denominator in (("present_loss", "std_correct_present"), ("falsefill", "missing"), ("invalid", "n")):
            ei, di = FIELDS.index(event), FIELDS.index(denominator)
            n_base = int(np.count_nonzero(counts[:, di]))
            if total[ei] == 0 and n_base:
                zero_bounds.append({"task": task, "group": group, "mode": mode, "policy": policy,
                    "method": method, "event": event, "n_applicable_base": n_base,
                    "one_sided95_base_any_event_upper": 1 - .05 ** (1 / n_base)})
    pairs = [("id_gate_constrained", m) for m in
             ("target_only_routed_gate_constrained", "target_only_gate_constrained", "standard_gate_constrained", "id_selection", "alias_field_parser")]
    pairs += [("target_only_gate_constrained", "target_only"), ("standard_gate_constrained", "standard"),
              ("id_selection", "target_only"), ("alias_field_parser", "field_parser")]
    contrasts = []
    for method, reference in pairs:
        for metric, (num, den) in RATIOS.items():
            a, b = aggregates[method], aggregates[reference]
            difference = a[num] / a[den] - b[num] / b[den] if a[den] and b[den] else None
            draws = all_draws[(method, metric)] - all_draws[(reference, metric)]
            finite = draws[np.isfinite(draws)]
            ci = list(map(float, np.percentile(finite, [2.5, 97.5]))) if len(finite) else [None, None]
            contrasts.append({"task": task, "group": group, "mode": mode, "policy": policy,
                "method": method, "reference": reference, "metric": metric,
                "n_base": len(clusters), "difference_method_minus_reference": difference,
                "ci_low": ci[0], "ci_high": ci[1]})
    return metric_rows, contrasts, zero_bounds, aggregates


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


