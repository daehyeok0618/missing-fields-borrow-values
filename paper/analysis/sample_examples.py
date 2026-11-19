"""Randomly sampled unsupported-field outputs for the Analysis section (not selected by outcome).

For each task, draws one request uniformly at random from each of the 8 design cells
(S1/S2 x donor order AB/BA x Middle/Adjacent) of the 2,304 unsupported-field (family MISSING)
requests of the T4 main run, with a fixed seed per task; the template is whatever the draw gives.
Outcomes are read from request_level.jsonl (strict policy, as produced by the locked analysis
scripts); nothing is re-scored here except naming which donor a donor match came from.

Disclosure: a first, unstratified draw (4 requests per task, seed 20261116, one shared RNG) was
discarded before writing because all 8 draws had donor order AB; the stratified design below was
fixed before it was run, and its draw is reported unchanged.

Run from the repository root:
  python paper/analysis/sample_examples.py
Prints the sample and writes paper/analysis/sampled_examples.json and sampled_examples.tex.
"""
import json
import random
from pathlib import Path

SEED = 20261116  # task i uses random.Random(SEED + i)
CELLS = [(a, o, p) for a in ("S1", "S2") for o in ("AB", "BA") for p in ("MIDDLE", "ADJACENT")]
TASKS = [("Salary", "experiments/hw2_missing_fields/results/request_level.jsonl"),
         ("Founding Year", "experiments/hw2_year_generalization/results/request_level.jsonl")]
TEMPLATE = {"KV": "key--value", "PROSE": "prose", "BULLET": "bulleted"}
OUTCOME = {"ABSTAIN": "abstention", "DONOR_MATCH": "donor match",
           "OTHER_IN_CONTEXT": "other in-context", "CONTEXT_ABSENT": "context-absent",
           "INVALID": "invalid"}


def digits(x):
    return "".join(ch for ch in str(x) if ch.isdigit())


def describe(r):
    out = OUTCOME[r["category"]]
    if r["category"] == "DONOR_MATCH":
        got = digits(r["parse"]["value"])
        j = 1 if got == digits(r["v1"]) else 2 if got == digits(r["v2"]) else None
        assert j is not None, r["request_id"]
        queried = 1 if r["assignment"] == "S1" else 2
        out = f"donor $v_{j}$ ({'queried' if j == queried else 'competing'} label)"
    return out


def main():
    root = Path(__file__).resolve().parents[2]
    picked = []
    for i, (task, rel) in enumerate(TASKS):
        rng = random.Random(SEED + i)
        rows = {}
        with open(root / rel) as f:
            for line in f:
                r = json.loads(line)
                if r["family"] == "MISSING":
                    rows[r["request_id"]] = r
        assert len(rows) == 2304, (task, len(rows))
        for a, o, pl in CELLS:
            cell = sorted(k for k, r in rows.items()
                          if (r["assignment"], r["order"], r["placement"]) == (a, o, pl))
            assert len(cell) == 288, (task, a, o, pl, len(cell))
            r = rows[rng.choice(cell)]
            picked.append(dict(task=task, request_id=r["request_id"], template=r["template"],
                               assignment=r["assignment"], order=r["order"],
                               placement=r["placement"], raw_output=r["raw_output"],
                               category=r["category"], v1=r["v1"], v2=r["v2"],
                               outcome=describe(r)))
    out_dir = Path(__file__).resolve().parent
    (out_dir / "sampled_examples.json").write_text(json.dumps(picked, indent=1, ensure_ascii=False))
    lines, task = [], None
    for p in picked:
        if p["task"] != task:
            task = p["task"]
            lines.append(f"\\midrule\n\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{task}}}}} \\\\")
        val = json.loads(p["raw_output"])["value"]
        shown = "null" if val is None else f'"{val}"'
        cond = f"$S_{p['assignment'][1]}$ {p['order']} {p['placement'][:3].capitalize()}."
        lines.append(f"{cond} & {TEMPLATE[p['template']]} & \\texttt{{{shown}}} & "
                     f"{p['outcome']} \\\\")
        print(p["task"], p["request_id"], repr(p["raw_output"]), p["category"], "|", p["outcome"],
              "| v1", p["v1"], "v2", p["v2"])
    (out_dir / "sampled_examples.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
