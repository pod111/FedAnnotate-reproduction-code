# -*- coding: utf-8 -*-
"""
make_fig4.py
============
Figure 4 of the paper: the privacy-utility trade-off measured on two public
Chinese corpora, plus the voter-strength contrast that explains why the two
curves differ in shape.

    python -m fedannotate.make_fig4

Reads ``_tradeoff.json`` (produced by ``run_tradeoff``) rather than hard-coding
numbers, so the figure cannot silently drift away from the experiments.

Panel (a)  macro-F1 against epsilon, error bars are the standard deviation over
           2 split seeds x 5 noise seeds.
Panel (b)  the same measurements normalised by each corpus's own epsilon=64
           score, which removes the level difference between the corpora and
           leaves only the shape difference.
Panel (c)  voter strength against measured privacy cost.

The claim the figure supports is stated in Sec. 4.3: privacy cost depends on the
error composition of the label model, not on epsilon alone. Where the voter is
strong the privacy noise dominates and the curve decays; where the voter is weak
the error is dominated by systematic bias and the curve stays flat.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from . import core

_DATA = None


def load():
    """Read _tradeoff.json written by run_tradeoff."""
    global _DATA
    if _DATA is None:
        with open(core.artifact_path("_tradeoff.json"), encoding="utf-8") as fh:
            _DATA = json.load(fh)
    return _DATA


def series(corpus, alpha):
    for row in load()["rows"]:
        if row["dataset"] == corpus and abs(row["alpha"] - alpha) < 1e-9:
            return [(s["epsilon"], s["macroF1"], s["sd"]) for s in row["scan"]]
    raise KeyError(f"{corpus} alpha={alpha} absent from _tradeoff.json")


# Gazetteer size and held-out coverage, as reported in Sec. 4.3.
VOTER = {
    "MSRA-NER": {"lex": 1172, "cov": 0.519},
    "News-NER": {"lex": 58, "cov": 0.253},
}
LABEL = {"MSRA-NER": "MSRA-NER", "News-NER": "政务新闻"}

C1, C2 = "#3B6FB6", "#E08A3C"
C3, C4 = "#4C9A6E", "#B565A7"
GRID = dict(alpha=0.25, linewidth=0.6)

SERIES = [
    ("MSRA-NER", 1.0, C1, "o", "-"),
    ("MSRA-NER", 0.3, C2, "s", "-"),
    ("News-NER", 1.0, C3, "^", "--"),
    ("News-NER", 0.3, C4, "D", "--"),
]


def _style(ax):
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(labelsize=9)


def _names(corpus, alpha):
    return f"{LABEL[corpus]}  α={alpha:.1f}"


def panel_a(ax):
    x = np.arange(6)
    ticks = []
    for corpus, alpha, color, marker, ls in SERIES:
        pts = series(corpus, alpha)
        ticks = [p[0] for p in pts]
        y = [p[1] for p in pts]
        e = [p[2] for p in pts]
        ax.errorbar(x, y, yerr=e, label=_names(corpus, alpha), color=color,
                    marker=marker, linestyle=ls, linewidth=1.6, markersize=5,
                    capsize=2.5, elinewidth=0.9, markeredgecolor="white",
                    markeredgewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"ε={v}" for v in ticks], fontsize=9)
    ax.set_xlabel("隐私预算 ε（向右递减）", fontsize=10)
    ax.set_ylabel("Macro-F1", fontsize=10)
    lo = min(p[1] for c, a, _, _, _ in SERIES for p in series(c, a))
    hi = max(p[1] for c, a, _, _, _ in SERIES for p in series(c, a))
    ax.set_ylim(lo - 0.05, hi + 0.05)
    ax.set_title("(a) 两个中文语料的效用—隐私权衡（均值，误差棒为标准差）",
                 fontsize=10.5, pad=8)
    ax.legend(fontsize=8.2, loc="lower left", framealpha=0.94)
    _style(ax)


def panel_b(ax):
    x = np.arange(6)
    ticks = []
    for corpus, alpha, color, marker, ls in SERIES:
        pts = series(corpus, alpha)
        ticks = [p[0] for p in pts]
        base = pts[0][1]
        rel = [100.0 * p[1] / base for p in pts]
        ax.plot(x, rel, label=_names(corpus, alpha), color=color, marker=marker,
                linestyle=ls, linewidth=1.6, markersize=5,
                markeredgecolor="white", markeredgewidth=0.6)
        ax.annotate(f"{rel[-1]:.0f}", xy=(x[-1], rel[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.8, color=color,
                    va="center")
    ax.axhline(100.0, color="#999999", linewidth=0.9, linestyle=":")
    ax.annotate("无隐私基线", xy=(0.02, 100.0), xytext=(0, 5),
                textcoords="offset points", fontsize=7.8, color="#666666")
    ax.set_xticks(x)
    ax.set_xticklabels([f"ε={v}" for v in ticks], fontsize=9)
    ax.set_xlim(-0.35, 5.8)
    ax.set_xlabel("隐私预算 ε", fontsize=10)
    ax.set_ylabel("相对无隐私的 Macro-F1（%）", fontsize=10)
    ax.set_title("(b) 归一化视图：隐私代价取决于误差构成，而非仅取决于 ε",
                 fontsize=10.5, pad=8)
    ax.legend(fontsize=8.2, loc="lower left", framealpha=0.94)
    _style(ax)


def panel_c(ax):
    names = list(VOTER.keys())
    lex = [VOTER[n]["lex"] for n in names]
    cov = [VOTER[n]["cov"] * 100 for n in names]
    drop = []
    for n in names:
        pts = series(n, 1.0)
        base = pts[0][1]
        # negative means epsilon=2 is no worse than epsilon=64
        drop.append((1 - pts[-1][1] / base) * 100)
    y = np.arange(len(names))
    h = 0.3
    ax.barh(y + h / 2, drop, h,
            label="ε=64→2 的质量变化（%，负值表示无损失）",
            color="#C0504D", edgecolor="white", linewidth=0.6)
    ax.barh(y - h / 2, cov, h, label="词典覆盖率（%）",
            color="#7F9DB9", edgecolor="white", linewidth=0.6)
    span = max(max(cov), max(drop))
    for i, (d, c) in enumerate(zip(drop, cov)):
        ax.text(d + (0.8 if d >= 0 else -0.8), i + h / 2, f"{d:+.0f}%",
                va="center", ha="left" if d >= 0 else "right", fontsize=8)
        ax.text(c + 0.8, i - h / 2, f"{c:.0f}%", va="center", fontsize=8)
    ax.axvline(0, color="#999999", linewidth=0.8, linestyle=":")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{LABEL[n]}\n(词典 {VOTER[n]['lex']} 条)" for n in names],
                       fontsize=8.6)
    ax.set_xlabel("百分比（%）", fontsize=10)
    ax.set_xlim(min(0, min(drop)) - span * 0.32, span * 1.32)
    ax.set_title("(c) 投票器强度与隐私代价的对应关系", fontsize=10.5, pad=8)
    ax.legend(fontsize=7.6, loc="lower center", bbox_to_anchor=(0.5, -0.32),
              frameon=False)
    _style(ax)
    ax.grid(axis="x", **GRID)
    ax.grid(axis="y", visible=False)


def main():
    os.makedirs(core.figure_path(""), exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10

    fig = plt.figure(figsize=(11.0, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.95],
                          wspace=0.34, left=0.06, right=0.985,
                          top=0.90, bottom=0.13)
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))
    panel_c(fig.add_subplot(gs[0, 2]))

    png = core.figure_path("fig4_chinese_eps_tradeoff.png")
    fig.savefig(png, format="PNG", dpi=600, bbox_inches="tight", facecolor="white")
    im = Image.open(png).convert("RGB")
    im.save(core.figure_path("fig4_chinese_eps_tradeoff.tiff"), format="TIFF",
            dpi=(600, 600), compression="tiff_lzw")
    im.save(core.figure_path("fig4_chinese_eps_tradeoff.jpeg"), format="JPEG",
            dpi=(600, 600), quality=95)
    print(f"saved -> {png}  ({im.size[0]}x{im.size[1]})")


if __name__ == "__main__":
    main()