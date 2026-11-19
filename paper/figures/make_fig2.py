"""Figure 2: outcome composition of the unsupported-field (MISSING) inputs by placement.

All numbers are read from the result files of the T4 main runs; nothing is typed in:
  experiments/hw2_missing_fields/results/outcome_rates.csv        (Salary)
  experiments/hw2_year_generalization/results/outcome_rates.csv   (Founding Year, policy STRICT)
  .../results/request_level.jsonl  (how many other in-context outputs are the string "null")
  .../results/contrasts.csv        (Delta_C, subset ALL; Year policy STRICT)
The script checks that the category counts in outcome_rates.csv sum to n and agree with
request_level.jsonl, and stops if any unsupported-field output is INVALID (none are, so the
figure has no INVALID segment).

Run from the repository root:
  python paper/figures/make_fig2.py
Writes paper/figures/fig2_outcomes.pdf (for LaTeX) and fig2_preview.png, and prints the
numbers drawn in the figure.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

TASKS = [  # (label, result dir, policy filter for outcome_rates/contrasts or None)
    ("Salary", "experiments/hw2_missing_fields/results", None),
    ("Founding Year", "experiments/hw2_year_generalization/results", "STRICT"),
]
PLACEMENTS = [("MIDDLE", "Middle"), ("ADJACENT", "Adjacent")]
# Left to right: donor match starts at 0 so C_M and C_A can be compared directly;
# abstention ends at 100, with the (mostly "null"-string) other in-context block next to it.
ORDER = ["DONOR_MATCH", "CONTEXT_ABSENT", "OTHER_IN_CONTEXT", "ABSTAIN"]
STYLE = {  # face, hatch, label text color; palette checked with the dataviz validator
    "DONOR_MATCH": ("#E07B39", None, "#1A1A1A"),       # same orange as the donor in Figure 1
    "CONTEXT_ABSENT": ("#6B6B6B", None, "#FFFFFF"),
    "OTHER_IN_CONTEXT": ("#72A9EB", "//////", "#1A1A1A"),
    "ABSTAIN": ("#2A78D6", None, "#FFFFFF"),
}
LEGEND = {
    "DONOR_MATCH": "Donor match",
    "CONTEXT_ABSENT": "Context-absent value",
    "OTHER_IN_CONTEXT": 'Other in-context (nearly all "null")',
    "ABSTAIN": "Abstention (null)",
}
HATCH_COLOR = "#2A78D6"
INK, GRAY = "#1A1A1A", "#5C5C5C"
LABEL_PAD_PT = 1.5  # print a segment's percentage only if it fits with this padding per side
GAP_PT = 0.5  # white stroke between segments


def load(root):
    data = {}
    for task, rel, policy in TASKS:
        res = root / rel
        counts = {p: Counter() for p, _ in PLACEMENTS}
        n = {}
        with open(res / "outcome_rates.csv", newline="") as f:
            for r in csv.DictReader(f):
                if policy is not None and r["policy"] != policy:
                    continue
                if r["family"] != "MISSING" or r["placement"] not in counts:
                    continue
                counts[r["placement"]][r["category"]] += int(r["count"])
                n.setdefault(r["placement"], int(r["n"]))
                assert n[r["placement"]] == int(r["n"])
        null_str, other_jsonl = Counter(), Counter()
        with open(res / "request_level.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r["family"] != "MISSING" or r["category"] != "OTHER_IN_CONTEXT":
                    continue
                other_jsonl[r["placement"]] += 1
                if r["parse"]["value"] == "null":  # the JSON string "null", not JSON null
                    null_str[r["placement"]] += 1
        delta_c = None
        with open(res / "contrasts.csv", newline="") as f:
            for r in csv.DictReader(f):
                if policy is not None and r["policy"] != policy:
                    continue
                if r["subset"] == "ALL" and r["metric"] == "delta_C":
                    delta_c = float(r["estimate"])
        assert delta_c is not None
        for p, _ in PLACEMENTS:
            c = counts[p]
            assert sum(c.values()) == n[p], (task, p, c, n[p])
            assert c["INVALID"] == 0, (task, p, "INVALID outputs present; add a segment")
            assert set(c) <= set(ORDER) | {"INVALID"}, (task, p, c)
            assert other_jsonl[p] == c["OTHER_IN_CONTEXT"], (task, p, other_jsonl[p])
        data[task] = dict(counts=counts, n=n, null_str=null_str, delta_c=delta_c)
    return data


def report(data):
    print(f"{'task':<14}{'placement':<10}{'n':>6}" + "".join(f"{k:>18}" for k in ORDER)
          + f"{'null-string':>13}")
    for task, d in data.items():
        for p, _ in PLACEMENTS:
            c, n = d["counts"][p], d["n"][p]
            cells = "".join(f"{c[k]:>8} ({100 * c[k] / n:5.1f}%)" for k in ORDER)
            print(f"{task:<14}{p:<10}{n:>6}{cells}{d['null_str'][p]:>13}")
        print(f"{task:<14}Delta_C = {100 * d['delta_c']:+.2f} pp")


def draw(data, out_pdf, out_png):
    plt.rcParams.update({
        "font.family": ["Liberation Sans", "DejaVu Sans"],  # Liberation Sans = Arial metrics
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        "mathtext.cal": "Liberation Sans",
        "mathtext.sf": "Liberation Sans",
        "mathtext.tt": "Liberation Mono",
        "font.size": 6.5,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "xtick.labelsize": 6,
        "hatch.linewidth": 0.45,
        "pdf.fonttype": 42,
        "text.hinting": "no_hinting",
    })
    W, H = 3.03, 1.72  # ACL \columnwidth is 7.7 cm
    fig = plt.figure(figsize=(W, H))
    left, right, bottom, top = 0.60, 0.10, 0.30, 0.30
    ax = fig.add_axes([left / W, bottom / H, 1 - (left + right) / W, 1 - (bottom + top) / H])

    bar_h = 0.74
    y, rows, headers = 0.0, [], []
    for task in data:
        headers.append((task, y))
        y += 0.95
        for p, plabel in PLACEMENTS:
            rows.append((task, p, plabel, y))
            y += 1.0
        y += 0.25
    y_max = y - 0.25 - 1.0 + 0.5

    seg_labels = []
    for task, p, plabel, yy in rows:
        c, n = data[task]["counts"][p], data[task]["n"][p]
        x = 0.0
        for k in ORDER:
            w = 100.0 * c[k] / n
            face, hatch, tcol = STYLE[k]
            if w > 0:
                ax.barh(yy, w, left=x, height=bar_h, color=face, edgecolor="white",
                        linewidth=GAP_PT, zorder=2)
                if hatch:
                    ax.barh(yy, w, left=x, height=bar_h, color="none", edgecolor=HATCH_COLOR,
                            hatch=hatch, linewidth=0, zorder=3)
                    ax.barh(yy, w, left=x, height=bar_h, color="none", edgecolor="white",
                            linewidth=GAP_PT, zorder=4)
            if w > 0:
                bbox = dict(boxstyle="square,pad=0.12", fc=face, ec="none") if hatch else None
                t = ax.text(x + w / 2, yy, f"{w:.1f}", ha="center", va="center", color=tcol,
                            fontsize=6, zorder=5, bbox=bbox)
                seg_labels.append((t, w))
            x += w
        ax.text(-1.5, yy, plabel, ha="right", va="center", color=INK)

    fig_x = blended_transform_factory(fig.transFigure, ax.transData)
    for task, yy in headers:
        ax.text(0.012, yy, task, transform=fig_x, ha="left", va="center", fontweight="bold",
                color=INK)
        dc = 100 * data[task]["delta_c"]
        dc_txt = f"{dc:+.2f}".replace("-", "\u2212")
        ax.text(100, yy, rf"$\Delta_C$ = {dc_txt} pp", ha="right", va="center", color=GRAY,
                fontsize=6)

    ax.set_xlim(0, 100)
    ax.set_ylim(y_max, -0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(axis="x", colors=GRAY, labelcolor=INK)
    n_all = {data[t]["n"][p] for t in data for p, _ in PLACEMENTS}
    assert len(n_all) == 1
    ax.set_xlabel(f"% of the {n_all.pop():,} unsupported-field inputs in each row",
                  fontsize=6, color=INK, labelpad=2)

    # Drop percentage labels that do not fit inside their segment (values stay in Table 3).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pct = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    pad_px = 2 * LABEL_PAD_PT * fig.dpi / 72
    for t, w in seg_labels:
        if t.get_window_extent(renderer).width + pad_px > w * px_per_pct:
            t.remove()

    handles = []
    for k in ORDER:
        face, hatch, _ = STYLE[k]
        h = Patch(facecolor=face, edgecolor=HATCH_COLOR if hatch else face, hatch=hatch,
                  linewidth=0, label=LEGEND[k])
        handles.append(h)
    # ncol=2 fills column-wise: reorder so rows read left to right in block order.
    leg_order = [handles[0], handles[2], handles[1], handles[3]]
    fig.legend(handles=leg_order, loc="upper left", bbox_to_anchor=(0.005, 0.995), ncol=2,
               frameon=False, fontsize=6, handlelength=1.3, handleheight=0.9,
               handletextpad=0.4, columnspacing=1.0, labelspacing=0.35, borderaxespad=0.0,
               borderpad=0.1)

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=400)
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=here.parents[1],
                    help="repository root that contains experiments/")
    ap.add_argument("--out", type=Path, default=here / "fig2_outcomes.pdf")
    ap.add_argument("--png", type=Path, default=here / "fig2_preview.png")
    a = ap.parse_args()
    data = load(a.root)
    report(data)
    draw(data, a.out, a.png)
    print(f"wrote {a.out} and {a.png}")


if __name__ == "__main__":
    main()
