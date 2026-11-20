"""Print every table and every Analysis-section number of the paper from the result files.

Nothing is typed in: all values are read from the outputs of the locked analysis scripts
(experiments/*/results/) and of the probe holdout analysis, and are printed in the units used in
the paper. Each value is also checked against the number printed in the paper (EXPECTED below);
the script stops with an error if any value differs.

Run from the repository root (after the analysis commands in README.md):
  python paper/analysis/paper_tables.py            # all tables and numbers
  python paper/analysis/paper_tables.py main       # Table 2 only (also: outcomes, probe, analysis)
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAL = ROOT / "experiments/hw2_missing_fields/results"
YEAR = ROOT / "experiments/hw2_year_generalization/results"
PROBE = ROOT / "experiments/hw2_holdout_colab/imported_runs/holdout_v1_828a82fc0e26/results/results"
TASKS = [("Salary", SAL), ("Founding Year", YEAR)]
CHECKS = []


def check(label, got, expected):
    """Record a printed value and compare it with the paper (same rounding as the paper)."""
    CHECKS.append((label, got, expected))
    if got != expected:
        raise SystemExit(f"MISMATCH {label}: computed {got}, paper {expected}")


def rows(path, **match):
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if all(r.get(k) == v for k, v in match.items())]


def pct(x, nd=2):
    return f"{100 * float(x):+.{nd}f}"


def request_level(results_dir):
    with open(results_dir / "request_level.jsonl") as f:
        return [json.loads(line) for line in f]


def digits(x):
    return "".join(ch for ch in str(x) if ch.isdigit())


# ---------------------------------------------------------------- Table 2 (main results)
def table_main():
    print("Table 2. Main results (percentage points, 95% bootstrap interval over 96 bases)")
    expected = {
        "Salary": {"delta_attr": ("+16.75", "+14.37", "+19.23"), "delta_C": ("+15.97", "+13.37", "+18.75"),
                   "delta_U": ("+0.43", "+0.00", "+0.95"), "delta_type": ("+15.54", "+12.76", "+18.49")},
        "Founding Year": {"delta_attr": ("+71.14", "+67.45", "+74.65"), "delta_C": ("-21.96", "-25.78", "-18.14"),
                          "delta_U": ("+4.17", "+2.52", "+5.99"), "delta_type": ("-26.13", "-31.16", "-21.35")},
    }
    for task, res in TASKS:
        extra = {"policy": "STRICT"} if task == "Founding Year" else {}
        print(f"  {task}")
        for metric in ("delta_attr", "delta_C", "delta_U", "delta_type"):
            (r,) = rows(res / "contrasts.csv", subset="ALL", metric=metric, **extra)
            got = (pct(r["estimate"]), pct(r["ci95_low"]), pct(r["ci95_high"]))
            check(f"{task} {metric}", got, expected[task][metric])
            print(f"    {metric:<11} {got[0]:>7}  [{got[1]}, {got[2]}]")
    sal = json.loads((SAL / "summary.json").read_text())
    yr = json.loads((YEAR / "summary.json").read_text())
    print(f"  Salary decisions: H1 supported={sal['h1_supported']}, H2 supported={sal['h2_supported']}")
    print(f"  Founding Year decisions: {json.dumps(yr['decisions'].get('STRICT', yr['decisions']), sort_keys=True)[:300]}")
    print("  Controls (% correct, strict policy)")
    ctrl = {"Salary": {k: v["estimate"] for k, v in sal["controls"].items()},
            "Founding Year": {r["family"]: float(r["estimate"]) for r in rows(YEAR / "control_accuracy.csv", policy="STRICT")}}
    exp_ctrl = {"Salary": {"T0": "100.0", "T1": "100.0", "PRESENT": "98.8", "N0": "99.0"},
                "Founding Year": {"T0": "100.0", "T1": "100.0", "PRESENT": "99.8", "N0": "63.1"}}
    for task in ("Salary", "Founding Year"):
        got = {k: f"{100 * ctrl[task][k]:.1f}" for k in ("T0", "T1", "PRESENT", "N0")}
        for k in got:
            check(f"{task} control {k}", got[k], exp_ctrl[task][k])
        print(f"    {task:<14} T0 abstention {got['T0']}, T1 accuracy {got['T1']}, "
              f"Present accuracy {got['PRESENT']}, N0 abstention {got['N0']}")


# ---------------------------------------------------------------- Table 5 (Appendix B)
def table_outcomes():
    print("Table 5 (Appendix B). Outcomes of unsupported-field inputs (%), strict policy")
    expected = {("Salary", "MIDDLE"): ("8.6", "9.0", "82.3", "0.1", "0.0"),
                ("Salary", "ADJACENT"): ("21.3", "25.0", "53.2", "0.5", "0.0"),
                ("Founding Year", "MIDDLE"): ("1.6", "95.7", "2.0", "0.8", "0.0"),
                ("Founding Year", "ADJACENT"): ("7.9", "73.7", "13.5", "4.9", "0.0")}
    cats = ("ABSTAIN", "DONOR_MATCH", "OTHER_IN_CONTEXT", "CONTEXT_ABSENT", "INVALID")
    for task, res in TASKS:
        extra = {"policy": "STRICT"} if task == "Founding Year" else {}
        for placement in ("MIDDLE", "ADJACENT"):
            est = {r["category"]: float(r["estimate"])
                   for r in rows(res / "outcome_rates.csv", family="MISSING", placement=placement, **extra)}
            got = tuple(f"{100 * est[c]:.1f}" for c in cats)
            check(f"{task} {placement} outcomes", got, expected[(task, placement)])
            print(f"  {task:<14} {placement:<9} abstain {got[0]:>5}  donor {got[1]:>5}  other {got[2]:>5}  "
                  f"absent {got[3]:>4}  invalid {got[4]}")
        reqs = [r for r in request_level(res) if r["family"] == "MISSING" and r["category"] == "OTHER_IN_CONTEXT"]
        null_str = sum(1 for r in reqs if isinstance(r["parse"].get("value"), str)
                       and r["parse"]["value"].strip().lower() == "null")
        check(f"{task} other-in-context that are 'null'", (null_str, len(reqs)),
              {"Salary": (1560, 1561), "Founding Year": (178, 178)}[task])
        print(f"  {task}: {null_str} of {len(reqs)} other in-context outputs are the string \"null\"")


# ---------------------------------------------------------------- Table 3 (probe)
PROBE_ROWS = [("Standard", "standard"), ("R+P", "target_only"), ("R+P+presence", "target_only_presence"),
              ("R+ID", "id_selection"), ("Standard+V", "standard_gate_constrained"),
              ("R+P+V", "target_only_routed_gate_constrained"), ("R+ID+V", "id_gate_constrained"),
              ("Literal parser", "field_parser"), ("Schema parser", "alias_field_parser"),
              ("Always null", "always_abstain")]
PROBE_EXPECTED = {"Standard": (113, 24, 21, 576, 734), "R+P": (156, 88, 130, 576, 950),
                  "R+P+presence": (192, 88, 130, 576, 986), "R+ID": (192, 160, 134, 576, 1062),
                  "Standard+V": (192, 186, 192, 576, 1146), "R+P+V": (192, 188, 192, 576, 1148),
                  "R+ID+V": (192, 188, 192, 576, 1148), "Literal parser": (192, 192, 192, 288, 864),
                  "Schema parser": (192, 192, 192, 576, 1152), "Always null": (192, 192, 192, 0, 576)}


def table_probe():
    print("Table 3. Probe holdout: correct outputs (16 bases, both wordings, both tasks pooled)")
    count = Counter()
    for r in rows(PROBE / "metrics.csv", split="holdout_all", mode="decision", policy="semantic", metric="accuracy"):
        count[(r["method"], r["group"])] += int(r["numerator"])
    print(f"  {'':<15}{'basic':>6}{'other':>6}{'multi':>6}{'Present':>8}{'All':>6}")
    for label, method in PROBE_ROWS:
        present = sum(count[(method, g)] for g in ("basic_present", "collision_present", "multi_present"))
        got = (count[(method, "basic_missing")], count[(method, "otherfield_missing")],
               count[(method, "multi_missing")], present, count[(method, "all_conditions")])
        check(f"probe {label}", got, PROBE_EXPECTED[label])
        print(f"  {label:<15}{got[0]:>6}{got[1]:>6}{got[2]:>6}{got[3]:>8}{got[4]:>6}")


# ---------------------------------------------------------------- Section 6 numbers
def analysis_numbers():
    print("Section 6 numbers")
    # 6.2 abstention notation
    for task, res in TASKS:
        miss = [r for r in request_level(res) if r["family"] == "MISSING"]
        for placement in ("MIDDLE", "ADJACENT", None):
            sel = [r for r in miss if placement is None or r["placement"] == placement]
            strict = sum(r["category"] == "ABSTAIN" for r in sel) / len(sel)
            with_null = sum(r["category"] == "ABSTAIN" or (r["category"] == "OTHER_IN_CONTEXT" and isinstance(
                r["parse"].get("value"), str) and r["parse"]["value"].strip().lower() == "null") for r in sel) / len(sel)
            print(f"  6.2 {task:<14} {placement or 'both':<9} abstention strict {100 * strict:.1f}% -> "
                  f"with \"null\" string {100 * with_null:.1f}%")
            if task == "Salary" and placement:
                check(f"6.2 Salary {placement}", (f"{100 * strict:.1f}", f"{100 * with_null:.1f}"),
                      {"MIDDLE": ("8.6", "90.8"), "ADJACENT": ("21.3", "74.5")}[placement])
    (y,) = rows(YEAR / "control_accuracy.csv", policy="SEMANTIC", family="MISSING")
    (ys,) = rows(YEAR / "control_accuracy.csv", policy="STRICT", family="MISSING")
    check("6.2 Year abstention strict -> nine-string policy",
          (f"{100 * float(ys['estimate']):.1f}", f"{100 * float(y['estimate']):.1f}"), ("4.7", "12.5"))
    print(f"  6.2 Founding Year abstention strict {100 * float(ys['estimate']):.1f}% -> nine-string policy "
          f"{100 * float(y['estimate']):.1f}%")
    sens = rows(SAL / "abstention_notation_sensitivity.csv")
    print("  6.2 Salary post-hoc notation sensitivity (abstention_notation_sensitivity.csv):")
    for r in sens:
        print("     ", {k: r[k] for k in r})

    # 6.3 borrowing with and without a matching label
    sal = request_level(SAL)
    n0_donor = sum(r["family"] == "N0" and r["category"] == "DONOR_MATCH" for r in sal)
    n0_n = sum(r["family"] == "N0" for r in sal)
    check("6.3 Salary N0 donor matches", (n0_donor, n0_n), (1, 1152))
    print(f"  6.3 Salary N0 donor value: {n0_donor} of {n0_n}")
    for task, res in TASKS:
        prim = [r for r in request_level(res) if r["family"] == "MISSING" and r["category"] == "DONOR_MATCH"]
        for placement in (None, "MIDDLE", "ADJACENT"):
            sel = [r for r in prim if placement is None or r["placement"] == placement]
            queried = sum(digits(r["parse"]["value"]) == digits(r["v1" if r["assignment"] == "S1" else "v2"])
                          for r in sel)
            print(f"  6.3 {task:<14} {placement or 'both':<9} donor matches following the queried label: "
                  f"{queried} of {len(sel)} = {100 * queried / len(sel):.1f}%")
            if task == "Salary" and placement is None:
                check("6.3 Salary queried share", (queried, len(sel)), (389, 392))
            if task == "Founding Year" and placement:
                check(f"6.3 Year queried share {placement}", f"{100 * queried / len(sel):.1f}", "92.0")
    pairs = defaultdict(dict)
    for r in sal:
        if r["family"] == "MISSING":
            pairs[(r["base_id"], r["template"], r["order"], r["placement"])][r["assignment"]] = r

    def is_null(r):
        v = r["parse"].get("value")
        return r["category"] == "ABSTAIN" or (isinstance(v, str) and v.strip().lower() == "null")
    one_sided = sum(1 for p in pairs.values()
                    if (p["S1"]["category"] == "DONOR_MATCH") != (p["S2"]["category"] == "DONOR_MATCH")
                    and is_null(p["S2"] if p["S1"]["category"] == "DONOR_MATCH" else p["S1"]))
    check("6.3 Salary one-sided pairs", (one_sided, len(pairs)), (364, 1152))
    print(f"  6.3 Salary pairs with a donor value in one input and null/\"null\" in the other: "
          f"{one_sided} of {len(pairs)}")
    for task, res in TASKS:
        extra = {"policy": "STRICT"} if task == "Founding Year" else {}
        (r,) = rows(res / "contrasts.csv", subset="ALL", metric="forward_switch", **extra)
        got = (f"{100 * float(r['estimate']):.2f}", f"{100 * float(r['ci95_low']):.2f}",
               f"{100 * float(r['ci95_high']):.2f}")
        check(f"6.3 {task} complete switches", got,
              {"Salary": ("0.95", "0.35", "1.74"), "Founding Year": ("60.76", "56.08", "65.45")}[task])
        print(f"  6.3 {task:<14} complete switches v1(S1)->v2(S2): {got[0]}% [{got[1]}, {got[2]}]")
    for placement, exp in (("MIDDLE", ("323", "576", "56.1", "52.6", "59.7")),
                           ("ADJACENT", ("86", "576", "14.9", "12.0", "18.1"))):
        (r,) = rows(YEAR / "outcome_rates.csv", family="N0", placement=placement, category="DONOR_MATCH",
                    policy="STRICT")
        got = (r["count"], r["n"], f"{100 * float(r['estimate']):.1f}", f"{100 * float(r['ci95_low']):.1f}",
               f"{100 * float(r['ci95_high']):.1f}")
        check(f"6.3 Year N0 donor {placement}", got, exp)
        print(f"  6.3 Founding Year N0 {placement:<9} donor value {got[0]}/{got[1]} = {got[2]}% [{got[3]}, {got[4]}]")
    # Founding Year primary inputs by placement (strict categories): donor values with the queried or the
    # competing label, null or "null" outputs, and context-absent outputs, all over the inputs of one placement.
    year = [r for r in request_level(YEAR) if r["family"] == "MISSING"]
    donor_rate = {}
    for placement, exp in (("MIDDLE", (1152, 1014, 88, "88.0", "7.6", "3.6", "0.8")),
                           ("ADJACENT", (1152, 781, 68, "67.8", "5.9", "21.4", "4.9"))):
        sel = [r for r in year if r["placement"] == placement]
        n = len(sel)
        donors = [r for r in sel if r["category"] == "DONOR_MATCH"]
        queried = sum(digits(r["parse"]["value"]) == digits(r["v1" if r["assignment"] == "S1" else "v2"])
                      for r in donors)
        competing = sum(digits(r["parse"]["value"]) == digits(r["v2" if r["assignment"] == "S1" else "v1"])
                        for r in donors)
        if queried + competing != len(donors):
            raise SystemExit(f"6.3 Year {placement}: a donor match carries neither donor value")
        null = sum(is_null(r) for r in sel)
        absent = sum(r["category"] == "CONTEXT_ABSENT" for r in sel)
        got = (n, queried, competing, f"{100 * queried / n:.1f}", f"{100 * competing / n:.1f}",
               f"{100 * null / n:.1f}", f"{100 * absent / n:.1f}")
        check(f"6.3 Year primary {placement}", got, exp)
        donor_rate[placement] = len(donors) / n
        print(f"  6.3 Founding Year {placement:<9} of {n} inputs: queried-label donor {queried} ({got[3]}%), "
              f"competing-label donor {competing} ({got[4]}%), null or \"null\" {got[5]}%, context-absent {got[6]}%")
    drop = 100 * (donor_rate["MIDDLE"] - donor_rate["ADJACENT"])
    check("6.3 Year drop in donor matching", f"{drop:.0f}", "22")
    print(f"  6.3 Founding Year drop in donor matching from Middle to Adjacent: {drop:.2f} points")

    # 6.4 probe error decomposition (basic Missing condition)
    count = Counter()
    for r in rows(PROBE / "metrics.csv", split="holdout_all", mode="decision", policy="semantic", metric="accuracy",
                  group="basic_missing"):
        count[r["method"]] += int(r["numerator"])
    errors = 192 - count["standard"]
    removed_by_routing = count["target_only"] - count["standard"]
    removed_by_presence = count["target_only_presence"] - count["target_only"]
    check("6.4 basic Missing decomposition", (errors, removed_by_routing, removed_by_presence), (79, 43, 36))
    print(f"  6.4 basic Missing: Standard errors {errors}; routing removes {removed_by_routing}; "
          f"presence check removes the remaining {removed_by_presence}")

    # 6.4 paired accuracy differences with 95% bootstrap intervals (all holdout inputs of one task)
    for task, key in (("Salary", "salary"), ("Founding Year", "year")):
        for (method, reference), label, nd, exp in (
                (("standard_gate_constrained", "standard"), "Standard+V - Standard", 1,
                 {"salary": ("30.9", "27.8", "34.0"), "year": ("40.6", "38.5", "42.4")}),
                (("target_only_routed_gate_constrained", "standard_gate_constrained"), "R+P+V - Standard+V", 2,
                 {"salary": ("0.35", "0.00", "1.04"), "year": ("0.00", "0.00", "0.00")})):
            diff, lo, hi = probe_contrast(key, method, reference)
            if (method, reference) == ("standard_gate_constrained", "standard"):
                # this pair is also written by the probe analysis; the recomputation must reproduce it exactly
                (r,) = rows(PROBE / "paired_contrasts.csv", split="holdout_all", task=key, group="all_conditions",
                            mode="decision", policy="semantic", method=method, reference=reference, metric="accuracy")
                if (diff, lo, hi) != tuple(float(r[k]) for k in ("difference_method_minus_reference", "ci_low",
                                                                 "ci_high")):
                    raise SystemExit(f"6.4 {task} {label}: recomputation differs from paired_contrasts.csv")
            got = tuple(f"{100 * x:.{nd}f}" for x in (diff, lo, hi))
            check(f"6.4 {task} {label}", got, exp[key])
            print(f"  6.4 {task:<14} {label}: {got[0]} points [{got[1]}, {got[2]}]")


def probe_contrast(task, method, reference, group="all_conditions", mode="decision", policy="semantic"):
    """Accuracy difference method - reference on the probe holdout with its 95% bootstrap interval.

    Uses the probe analysis code and its bootstrap seed for (task, group, mode, policy), so the draws are the
    ones behind paired_contrasts.csv; that file stores only some pairs (R+P+V - Standard+V is not among them).
    """
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from experiments.hw2_holdout_colab import statistics as st
    from experiments.hw2_holdout_colab.common import digest, read_jsonl
    from experiments.hw2_holdout_colab.protocol import PROTOCOL
    run = PROBE.parent
    gold = json.loads((run / "gold.json").read_text())
    records = [r for r in read_jsonl(run / "decisions.jsonl") if r["task"] == task and r["split"].startswith("holdout_")
               and (group == "all_conditions" or r["group"] == group)]
    clusters = sorted({(r["task"], r["base_id"]) for r in records})
    lookup = {c: i for i, c in enumerate(clusters)}
    seed = PROTOCOL["bootstrap_seed"] ^ int(digest([task, group, mode, policy])[:8], 16)
    weights = st.weights_for(clusters, PROTOCOL["bootstrap_repetitions"], seed)

    def accuracy(m):
        counts = np.zeros((len(clusters), 2), dtype=int)
        for r in records:
            flags = st.indicators(r, gold[r["id"]], m, policy, mode)
            counts[lookup[(r["task"], r["base_id"])]] += [flags["correct"], flags["n"]]
        total, resampled = counts.sum(axis=0), weights @ counts
        return float(total[0] / total[1]), resampled[:, 0] / resampled[:, 1]

    (a, draws_a), (b, draws_b) = accuracy(method), accuracy(reference)
    lo, hi = map(float, np.percentile(draws_a - draws_b, [2.5, 97.5]))
    return a - b, lo, hi


PARTS = {"main": table_main, "outcomes": table_outcomes, "probe": table_probe, "analysis": analysis_numbers}

if __name__ == "__main__":
    for name in (sys.argv[1:] or PARTS):
        PARTS[name]()
        print()
    print(f"All {len(CHECKS)} values match the numbers printed in the paper.")
