"""Figure 3: H1 and H2 contrasts by template and by donor order (secondary estimates).

All numbers are read from the analysis outputs of the T4 main runs; nothing is typed in:
  experiments/hw2_missing_fields/results/contrasts.csv        (Salary)
  experiments/hw2_year_generalization/results/contrasts.csv   (Founding Year, policy STRICT)
Rows use the subsets that the locked analysis scripts already estimated: ALL (primary),
template=KV/PROSE/BULLET and order=AB/BA (secondary; pointwise 95 % bootstrap intervals over
the 96 base examples, 2,000 resamples). No new estimate is computed here.

Run from the repository root:
  python paper/figures/make_fig3.py
Writes paper/figures/fig3_subsets.pdf (for LaTeX) and fig3_preview.png, and prints the values.
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TASKS = [  # (label, result dir, policy filter or None, color, marker)
    ("Salary", "experiments/hw2_missing_fields/results", None, "#6250D6", "o"),
    ("Founding Year", "experiments/hw2_year_generalization/results", "STRICT", "#008300", "s"),
]
ROWS = [  # (subset in contrasts.csv, row label, gap before the row)
    ("ALL", "All", 0.0),
    ("template=KV", "Key–value", 0.45),
    ("template=PROSE", "Prose", 0.0),
    ("template=BULLET", "Bulleted", 0.0),
    ("order=AB", "Order AB", 0.45),
    ("order=BA", "Order BA", 0.0),
]
PANELS = [("delta_attr", r"$\Delta_{\mathrm{attr}}$ (H1)"), ("delta_C", r"$\Delta_C$ (H2)")]
INK, GRAY, LIGHT = "#1A1A1A", "#5C5C5C", "#EFEFEF"
OFFSET = 0.17  # vertical offset of the two tasks within a row


def load(root):
    est = {}
    for task, rel, policy, _, _ in TASKS:
        with open(root / rel / "contrasts.csv", newline="") as f:
            for r in csv.DictReader(f):
                if policy is not None and r["policy"] != policy:
                    continue
                key = (task, r["subset"], r["metric"])
                assert key not in est, key
                assert int(r["n_base"]) == 96, r
                est[key] = tuple(100 * float(r[c]) for c in ("estimate", "ci95_low", "ci95_high"))
    for task, *_ in TASKS:
        for subset, _, _ in ROWS:
            for metric, _ in PANELS:
                assert (task, subset, metric) in est, (task, subset, metric)
    return est


def report(est):
    for metric, _ in PANELS:
        print(metric)
        for subset, label, _ in ROWS:
            cells = "".join(
                f"   {task}: {est[(task, subset, metric)][0]:+7.2f} "
                f"[{est[(task, subset, metric)][1]:+7.2f}, {est[(task, subset, metric)][2]:+7.2f}]"
                for task, *_ in TASKS)
            print(f"  {label:<10}{cells}")


def nice_limits(lo, hi, step=10, pad=3):
    """Axis limits on multiples of `step` that contain every interval and zero."""
    return (min(0.0, math.floor((lo - pad) / step) * step),
            max(0.0, math.ceil((hi + pad) / step) * step))


def draw(est, out_pdf, out_png):
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
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,
        "text.hinting": "no_hinting",
    })
    W, H = 3.03, 1.95  # ACL \columnwidth is 7.7 cm
    left, right, bottom, top, gap = 0.58, 0.08, 0.30, 0.34, 0.16
    pw = (W - left - right - gap) / 2
    fig = plt.figure(figsize=(W, H))

    ys, y = [], 0.0
    for _, _, g in ROWS:
        y += g
        ys.append(y)
        y += 1.0
    y_lo, y_hi = -0.55, ys[-1] + 0.55

    for i, (metric, title) in enumerate(PANELS):
        ax = fig.add_axes([(left + i * (pw + gap)) / W, bottom / H, pw / W, 1 - (bottom + top) / H])
        lo = min(est[(t, s, metric)][1] for t, *_ in TASKS for s, _, _ in ROWS)
        hi = max(est[(t, s, metric)][2] for t, *_ in TASKS for s, _, _ in ROWS)
        x0, x1 = nice_limits(lo, hi)
        ax.axhspan(ys[0] - 0.5, ys[0] + 0.5, color=LIGHT, lw=0, zorder=0)  # primary row
        ax.axvline(0, color=GRAY, lw=0.5, ls=(0, (2, 1.5)), zorder=1)
        for j, (task, _, _, color, marker) in enumerate(TASKS):
            dy = -OFFSET if j == 0 else OFFSET
            for (subset, _, _), yy in zip(ROWS, ys):
                e, l, h = est[(task, subset, metric)]
                ax.plot([l, h], [yy + dy, yy + dy], color=color, lw=0.9, solid_capstyle="butt",
                        zorder=2)
                ax.plot([e], [yy + dy], marker=marker, ms=3.0, color=color, mec="white",
                        mew=0.4, zorder=3)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y_hi, y_lo)
        ax.set_xticks([t for t in range(-100, 101, 20) if x0 <= t <= x1])
        ax.set_yticks([])
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRAY)
        ax.tick_params(axis="x", colors=GRAY, labelcolor=INK, pad=1.5)
        ax.set_title(title, fontsize=6.5, color=INK, pad=3)
        if i == 0:
            for (_, label, _), yy in zip(ROWS, ys):
                ax.text(-0.04, yy, label, transform=ax.get_yaxis_transform(), ha="right",
                        va="center", color=INK, fontweight="bold" if label == "All" else "normal")
    fig.text((left + (W - left - right) / 2) / W, 0.03, "Percentage points (95% bootstrap interval)",
             ha="center", va="bottom", fontsize=6, color=INK)

    handles = [Line2D([0], [0], color=c, lw=0.9, marker=m, ms=3.0, mec="white", mew=0.4, label=t)
               for t, _, _, c, m in TASKS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.995), ncol=2,
               frameon=False, fontsize=6.5, handlelength=1.6, handletextpad=0.4,
               columnspacing=1.4, borderaxespad=0.0, borderpad=0.1)

    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=400)
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=here.parents[1],
                    help="repository root that contains experiments/")
    ap.add_argument("--out", type=Path, default=here / "fig3_subsets.pdf")
    ap.add_argument("--png", type=Path, default=here / "fig3_preview.png")
    a = ap.parse_args()
    est = load(a.root)
    report(est)
    draw(est, a.out, a.png)
    print(f"wrote {a.out} and {a.png}")


if __name__ == "__main__":
    main()
