"""Figure 1 (task overview) as an editable PowerPoint slide + a matplotlib preview.

Every element is defined once in SPEC (inches, slide = 6.3 x 2.55 in = ACL \\textwidth),
then written to .pptx with python-pptx and drawn with matplotlib for layout QA
(Liberation Sans/Mono are metric-compatible with Arial/Courier New).

Example: Founding Year main set, first base example HW2YM000, KV template, order AB,
Middle placement (chosen by position, not by outcome). Outputs are the T4 run's
raw outputs from experiments/hw2_year_generalization/results/request_level.jsonl:
  S1 AB MIDDLE -> {"value": "1904"} DONOR_MATCH
  S2 AB MIDDLE -> {"value": "1937"} DONOR_MATCH
  N0 AB MIDDLE -> {"value": null}  ABSTAIN
Token spans (MIDDLE): target 128-156, donors 964-1038, question 1976-1995, total 2000;
(ADJACENT): donors 1902-1976.
"""
import sys
from pathlib import Path

W, H = 6.3, 2.55
SANS, MONO = "Arial", "Courier New"

INK, GRAY = "222222", "666666"
TGT_FILL, TGT_LINE, TGT_TAG = "E8EEF4", "8DA2B5", "4A6378"
DON_FILL, DON_LINE, ORANGE = "FDEBDD", "E59866", "C8551B"
BG_FILL, BG_LINE = "F2F2F2", "B5B5B5"
NEU_FILL, NEU_LINE = "F2F2F2", "BDBDBD"
RED, GREEN = "B71C1C", "2E7D32"
FRAME, QLINE = "9AA5B1", "555555"
SEG_T, SEG_B, SEG_D, SEG_Q = "8DA2B5", "E3E3E3", "E07B39", "333333"


def R(text, size=6.5, bold=False, italic=False, color=INK, font=SANS, sub=False):
    return dict(text=text, size=size, bold=bold, italic=italic, color=color, font=font, sub=sub)


SPEC = []


def box(name, x, y, w, h, fill=None, line=None, lw=0.75, dash=False, rounded=False, paras=None,
        align="l", valign="t", margin=(0.04, 0.04, 0.025, 0.02)):
    SPEC.append(dict(kind="box", name=name, x=x, y=y, w=w, h=h, fill=fill, line=line, lw=lw, dash=dash,
                     rounded=rounded, paras=paras or [], align=align, valign=valign, margin=margin))


def text(name, x, y, w, h, paras, align="l", valign="t", margin=(0, 0, 0, 0)):
    box(name, x, y, w, h, paras=paras, align=align, valign=valign, margin=margin)


def arrow(name, x1, y1, x2, y2, color=GRAY, lw=0.75):
    SPEC.append(dict(kind="arrow", name=name, x1=x1, y1=y1, x2=x2, y2=y2, color=color, lw=lw))


# ---------------------------------------------------------------- (a) example
AX, AW = 0.05, 2.40
IX, IW = 0.12, 2.26
text("a_title", AX, 0.03, AW, 0.16, [[R("(a) A donor's value fills a missing field", 8, bold=True)]])
box("a_frame", AX, 0.22, AW, 1.82, line=FRAME, rounded=True)
text("a_label", IX, 0.245, IW, 0.11, [[R("Input (abbreviated; 2,000 tokens)", 6, italic=True, color=GRAY)]])

box("a_target", IX, 0.37, IW, 0.40, fill=TGT_FILL, line=TGT_LINE, paras=[
    [R("Organization: Nelporlum Quiltor Institute")],
    [R("Activity: map conservation")],
    [R("Region: western district")]])
text("a_target_tag", IX + IW - 0.30, 0.39, 0.26, 0.11, [[R("T", 6.5, bold=True, color=TGT_TAG)]], align="r")
text("a_target_note", IX + IW - 1.00, 0.645, 0.96, 0.11,
     [[R("no founding year", 6, italic=True, color=RED)]], align="r")

box("a_bg1", IX, 0.81, IW, 0.12, fill=BG_FILL, line=BG_LINE, dash=True, align="c", valign="m",
    margin=(0, 0, 0, 0), paras=[[R("background text (Alice in Wonderland)", 6, italic=True, color=GRAY)]])

box("a_d1", IX, 0.97, IW, 0.27, fill=DON_FILL, line=DON_LINE, paras=[
    [R("Organization: Horxelwyn Torrin Institute")],
    [R("Recorded "), R("founding year", bold=True, color=ORANGE), R(": "), R("1904", bold=True)]])
text("a_d1_tag", IX + IW - 0.30, 0.99, 0.26, 0.11,
     [[R("D", 6.5, bold=True, color=ORANGE), R("1", 6.5, bold=True, color=ORANGE, sub=True)]], align="r")
box("a_d2", IX, 1.27, IW, 0.27, fill=DON_FILL, line=DON_LINE, paras=[
    [R("Organization: Lumhorpor Gilhor Institute")],
    [R("Recorded audit year: "), R("1937", bold=True)]])
text("a_d2_tag", IX + IW - 0.30, 1.29, 0.26, 0.11,
     [[R("D", 6.5, bold=True, color=ORANGE), R("2", 6.5, bold=True, color=ORANGE, sub=True)]], align="r")

box("a_bg2", IX, 1.58, IW, 0.12, fill=BG_FILL, line=BG_LINE, dash=True, align="c", valign="m",
    margin=(0, 0, 0, 0), paras=[[R("background text (Alice in Wonderland)", 6, italic=True, color=GRAY)]])
box("a_question", IX, 1.74, IW, 0.26, fill="FFFFFF", line=QLINE, paras=[
    [R("Question: ", bold=True), R("What is the recorded founding year for")],
    [R("Nelporlum Quiltor Institute?")]])

text("a_output", AX + 0.02, 2.08, AW, 0.12, [[
    R("Model output: ", bold=True), R('{"value": "1904"}', 6.5, bold=True, color=ORANGE, font=MONO),
    R("   donor match", 6.5, bold=True, color=RED)]])
text("a_correct", AX + 0.02, 2.21, AW, 0.12, [[
    R("Correct output: ", bold=True), R('{"value": null}', 6.5, bold=True, color=GREEN, font=MONO)]])
text("a_note", AX + 0.02, 2.345, AW, 0.11, [[
    R("1904 is in the input, so a presence check accepts it.", 6, italic=True, color=GRAY)]])

# ---------------------------------------------------------------- (b) H1
BX = 2.62
text("b_title", BX, 0.03, W - BX - 0.05, 0.16,
     [[R("(b) H1: exchange the attribute labels; the values stay in place", 8, bold=True)]])
C_COND, C_D1, C_D2, C_OUT = BX, 2.97, 4.14, 5.52
CW = 1.12
text("b_h_d1", C_D1, 0.235, CW, 0.11, [[R("D", 6, color=GRAY), R("1", 6, color=GRAY, sub=True)]], align="c")
text("b_h_d2", C_D2, 0.235, CW, 0.11, [[R("D", 6, color=GRAY), R("2", 6, color=GRAY, sub=True)]], align="c")
text("b_h_out", C_OUT, 0.235, 0.73, 0.11, [[R("Model output", 6, color=GRAY)]])

ROWS = [
    ("S", "1", ("founding year", True), ("audit year", False), ("1904", ORANGE, "(D1's value)")),
    ("S", "2", ("audit year", False), ("founding year", True), ("1937", ORANGE, "(D2's value)")),
    ("N", "0", ("audit year", False), ("review year", False), ("null", GREEN, "(abstains)")),
]
for i, (c, sub, d1, d2, out) in enumerate(ROWS):
    y = 0.37 + i * 0.31
    hgt = 0.24
    text(f"b_row{i}_cond", C_COND, y, 0.32, hgt,
         [[R(c, 7, italic=True), R(sub, 7, sub=True)]], valign="m")
    for col, (lab, queried), val, nm in ((C_D1, d1, "1904", "d1"), (C_D2, d2, "1937", "d2")):
        box(f"b_row{i}_{nm}", col, y, CW, hgt,
            fill=DON_FILL if queried else NEU_FILL, line=DON_LINE if queried else NEU_LINE, rounded=True,
            align="c", valign="m", margin=(0.02, 0.02, 0, 0),
            paras=[[R(lab, bold=queried, color=ORANGE if queried else INK), R(": "), R(val, bold=True)]])
    arrow(f"b_row{i}_arrow", 5.30, y + hgt / 2, 5.47, y + hgt / 2)
    val, colr, note = out
    text(f"b_row{i}_out", C_OUT, y, 0.76, hgt, [[R(val, 7, bold=True, color=colr, font=MONO)],
                                                [R(note, 5.5, color=GRAY)]], valign="m")
text("b_note", BX, 1.31, W - BX - 0.05, 0.11, [[
    R("Same entities, values, order, and length in every row; the correct output is always null.",
      6, italic=True, color=GRAY)]])

# ---------------------------------------------------------------- (c) H2
text("c_title", BX, 1.50, W - BX - 0.05, 0.16,
     [[R("(c) H2: move the same donor block within the input", 8, bold=True)]])
X0, SW, NTOK = 3.20, 3.00, 2000
s = SW / NTOK


def seg(name, y, a, b, fill, label=None, lab_color=INK):
    paras = []
    if label:
        paras = [[R(label[0], 6, bold=label[0] == "T", color=lab_color)]
                 + ([R(label[1], 6, color=lab_color, sub=True)] if len(label) > 1 else [])]
    box(name, X0 + a * s, y, (b - a) * s, 0.15, fill=fill, line=None, align="c", valign="m",
        margin=(0, 0, 0, 0), paras=paras)


for row, (lab, y, dpos) in enumerate((("Middle", 1.84, (964, 1038)), ("Adjacent", 2.15, (1902, 1976)))):
    text(f"c_{lab}_label", BX, y, 0.55, 0.15, [[R(lab, 6.5)]], valign="m")
    seg(f"c_{lab}_T", y, 0, 156, SEG_T, ("T",), "FFFFFF")
    seg(f"c_{lab}_Bpre", y, 156, 964, SEG_B, ("B", "pre"))
    if lab == "Middle":
        seg(f"c_{lab}_Bsuf", y, 1038, 1976, SEG_B, ("B", "suf"))
    else:
        seg(f"c_{lab}_Bsuf", y, 964, 1902, SEG_B, ("B", "suf"))
    seg(f"c_{lab}_D", y, dpos[0], dpos[1], SEG_D)
    seg(f"c_{lab}_q", y, 1976, NTOK, SEG_Q)
    text(f"c_{lab}_qlab", X0 + SW + 0.02, y, 0.08, 0.15, [[R("q", 6.5, italic=True)]], valign="m")
    dlab = [[R("D", 6, bold=True, color=ORANGE), R("1", 6, bold=True, color=ORANGE, sub=True),
             R(" D", 6, bold=True, color=ORANGE), R("2", 6, bold=True, color=ORANGE, sub=True)]]
    if lab == "Middle":
        cx = X0 + (dpos[0] + dpos[1]) / 2 * s
        text(f"c_{lab}_dlab", cx - 0.6, y - 0.125, 1.2, 0.11,
             [dlab[0] + [R("  (midpoint at 50%)", 6, color=GRAY)]], align="c")
    else:
        text(f"c_{lab}_dlab", X0 + dpos[1] * s - 1.6, y - 0.125, 1.6, 0.11,
             [dlab[0] + [R("  (immediately before q)", 6, color=GRAY)]], align="r")
for v, al in ((0, "l"), (1000, "c"), (2000, "r")):
    lab = {0: "0", 1000: "1,000", 2000: "2,000 tokens"}[v]
    x = X0 + v * s
    wbox = 0.8
    xl = {"l": x, "c": x - wbox / 2, "r": x - wbox}[al]
    text(f"c_tick{v}", xl, 2.32, wbox, 0.11, [[R(lab, 6, color=GRAY)]], align=al)


# ================================================================ writers
def write_pptx(path):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
    from pptx.enum.dml import MSO_LINE_DASH_STYLE
    from pptx.oxml.ns import qn
    from lxml import etree

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    AL = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}
    VA = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}

    def drop_style(shape):
        st = shape._element.find(qn("p:style"))
        if st is not None:
            shape._element.remove(st)

    for e in SPEC:
        if e["kind"] == "arrow":
            c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(e["x1"]), Inches(e["y1"]),
                                           Inches(e["x2"]), Inches(e["y2"]))
            drop_style(c)
            c.name = e["name"]
            c.line.color.rgb = RGBColor.from_string(e["color"])
            c.line.width = Pt(e["lw"])
            ln = c.line._get_or_add_ln()
            tail = etree.SubElement(ln, qn("a:tailEnd"))
            tail.set("type", "triangle"); tail.set("w", "sm"); tail.set("len", "sm")
            continue
        is_shape = e["fill"] or e["line"]
        if is_shape:
            shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if e["rounded"] else MSO_SHAPE.RECTANGLE,
                                         Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
            drop_style(shp)
            if e["rounded"]:
                shp.adjustments[0] = (0.06 if e["name"] == "a_frame" else 0.03) / min(e["w"], e["h"])
            if e["fill"]:
                shp.fill.solid(); shp.fill.fore_color.rgb = RGBColor.from_string(e["fill"])
            else:
                shp.fill.background()
            if e["line"]:
                shp.line.color.rgb = RGBColor.from_string(e["line"]); shp.line.width = Pt(e["lw"])
                if e["dash"]:
                    shp.line.dash_style = MSO_LINE_DASH_STYLE.DASH
            else:
                shp.line.fill.background()
        else:
            shp = slide.shapes.add_textbox(Inches(e["x"]), Inches(e["y"]), Inches(e["w"]), Inches(e["h"]))
        shp.name = e["name"]
        tf = shp.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        ml, mr, mt, mb = e["margin"]
        tf.margin_left, tf.margin_right, tf.margin_top, tf.margin_bottom = (Inches(ml), Inches(mr),
                                                                            Inches(mt), Inches(mb))
        tf.vertical_anchor = VA[e["valign"]]
        for pi, para in enumerate(e["paras"]):
            p = tf.paragraphs[0] if pi == 0 else tf.add_paragraph()
            p.alignment = AL[e["align"]]
            p.space_before = Pt(0); p.space_after = Pt(0)
            for run in para:
                r = p.add_run()
                r.text = run["text"]
                f = r.font
                f.name = run["font"]; f.size = Pt(run["size"]); f.bold = run["bold"]; f.italic = run["italic"]
                f.color.rgb = RGBColor.from_string(run["color"])
                if run["sub"]:
                    r._r.get_or_add_rPr().set("baseline", "-25000")
    prs.save(path)


def write_preview(path):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["text.hinting"] = "no_hinting"
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath

    FAM = {SANS: "Liberation Sans", MONO: "Liberation Mono"}
    ttp = TextToPath()
    fig = plt.figure(figsize=(W, H), dpi=250)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, ec="#dddddd", lw=0.3))
    problems = []

    def prop(run):
        sz = run["size"] * (0.65 if run["sub"] else 1)
        return FontProperties(family=FAM[run["font"]], size=sz,
                              weight="bold" if run["bold"] else "normal",
                              style="italic" if run["italic"] else "normal")

    def width_in(run):
        # TextToPath drops trailing spaces; measure with a sentinel so spaces count.
        p = prop(run)
        w1 = ttp.get_text_width_height_descent(run["text"] + "|", p, ismath=False)[0]
        w0 = ttp.get_text_width_height_descent("|", p, ismath=False)[0]
        return (w1 - w0) / 72

    for e in SPEC:
        if e["kind"] == "arrow":
            ax.add_patch(FancyArrowPatch((e["x1"], e["y1"]), (e["x2"], e["y2"]), arrowstyle="-|>",
                                         mutation_scale=5, lw=e["lw"], color="#" + e["color"]))
            continue
        if e["fill"] or e["line"]:
            kw = dict(fc="#" + e["fill"] if e["fill"] else "none",
                      ec="#" + e["line"] if e["line"] else "none", lw=e["lw"],
                      ls="--" if e["dash"] else "-")
            if e["rounded"]:
                rad = 0.06 if e["name"] == "a_frame" else 0.03
                ax.add_patch(FancyBboxPatch((e["x"] + rad, e["y"] + rad), e["w"] - 2 * rad, e["h"] - 2 * rad,
                                            boxstyle=f"round,pad={rad}", **kw))
            else:
                ax.add_patch(Rectangle((e["x"], e["y"]), e["w"], e["h"], **kw))
        if not e["paras"]:
            continue
        ml, mr, mt, mb = e["margin"]
        inner_w = e["w"] - ml - mr
        lh = [1.2 * max(r["size"] for r in p) / 72 for p in e["paras"]]
        tot = sum(lh)
        avail_h = e["h"] - mt - mb
        if tot > avail_h + 0.005:
            problems.append(f"{e['name']}: text height {tot:.3f} > {avail_h:.3f}")
        y = {"t": e["y"] + mt, "m": e["y"] + mt + (avail_h - tot) / 2, "b": e["y"] + e["h"] - mb - tot}[e["valign"]]
        for p, h_ in zip(e["paras"], lh):
            pw = sum(width_in(r) for r in p)
            if pw > inner_w + 0.005:
                problems.append(f"{e['name']}: line width {pw:.3f} > {inner_w:.3f}: "
                                + "".join(r["text"] for r in p))
            x = {"l": e["x"] + ml, "c": e["x"] + ml + (inner_w - pw) / 2, "r": e["x"] + e["w"] - mr - pw}[e["align"]]
            base = y + h_ * 0.8
            for r in p:
                yy = base + (0.03 * r["size"] / 6.5 if r["sub"] else 0)
                ax.text(x, yy, r["text"], fontproperties=prop(r), color="#" + r["color"], va="baseline", ha="left")
                x += width_in(r)
            y += h_
        if e["x"] < 0 or e["y"] < 0 or e["x"] + e["w"] > W + 1e-6 or e["y"] + e["h"] > H + 1e-6:
            problems.append(f"{e['name']}: outside slide")
    fig.savefig(path, dpi=250)
    return problems


if __name__ == "__main__":
    out_pptx, out_png = Path(sys.argv[1]), Path(sys.argv[2])
    write_pptx(out_pptx)
    probs = write_preview(out_png)
    print("PROBLEMS:" if probs else "no overflow detected")
    for p in probs:
        print("  ", p)
