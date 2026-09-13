#!/usr/bin/env python3
"""
make_fig3.py — Fig. 3, calibration of content-based grouping against true sessions
=================================================================================
Precision against recall over within-class image pairs, one curve per
representation, swept over its threshold. The horizontal line is the base
rate (0.175): the precision attained by merging each class entirely, which
is also the precision a clustering scores when it has collapsed. Degenerate
operating points are drawn unfilled.

The figure makes three things visible at once, which is why it replaces the
accuracy--capacity scatter in the main text:

  * dHash sits in the bottom-right corner: high precision, no recall.
  * DINOv2 extends the curve leftward and upward but never far from the base
    rate at any usable recall.
  * No point is far from BOTH axes' failure modes; there is no operating
    point where a reader could trust a post hoc audit.

Values are those of Table 3 and are held in this file so the figure can be
regenerated without rerunning the calibration. If any number in Table 3
changes, change it here too -- the figure will not warn you.

    python make_fig3.py --out figures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_RATE = 0.175
TRUE_LEAK = 93.9

# (threshold label, recall, precision, degenerate)
SERIES = {
    "dHash": dict(
        colour="#444444", marker="s", ls="--",
        points=[("2", 0.0000, 0.333, False),
                ("6", 0.0007, 0.714, False),
                ("8", 0.0010, 0.800, False),
                ("10", 0.0020, 0.882, False),
                ("12", 0.0030, 0.913, False)],
        label="Difference hash (Hamming $t$)"),
    "EffNet": dict(
        colour="#1f77b4", marker="o", ls="-",
        points=[("0.50", 0.9999, 0.175, True),
                ("0.60", 0.9897, 0.176, True),
                ("0.65", 0.8954, 0.192, False),
                ("0.70", 0.5254, 0.203, False),
                ("0.75", 0.1401, 0.264, False),
                ("0.80", 0.0473, 0.386, False),
                ("0.85", 0.0216, 0.733, False),
                ("0.90", 0.0040, 0.933, False),
                ("0.95", 0.0006, 0.667, False)],
        label="EfficientNet-B0, ImageNet (cosine $\\tau$)"),
    "DINOcls": dict(
        colour="#d62728", marker="o", ls="-",
        points=[("0.50", 0.9973, 0.175, True),
                ("0.60", 0.9966, 0.175, True),
                ("0.70", 0.9890, 0.179, True),
                ("0.80", 0.8685, 0.216, False),
                ("0.85", 0.5992, 0.302, False),
                ("0.86", 0.5219, 0.322, False),
                ("0.87", 0.4044, 0.305, False),
                ("0.88", 0.1684, 0.411, False),
                ("0.89", 0.1148, 0.453, False),
                ("0.90", 0.0814, 0.488, False),
                ("0.95", 0.0056, 0.951, False)],
        label="DINOv2 ViT-B/14, class token (cosine $\\tau$)"),
    "DINOmean": dict(
        colour="#ff7f0e", marker="^", ls=":",
        points=[("0.50", 1.0000, 0.175, True),
                ("0.60", 0.9999, 0.175, True),
                ("0.70", 0.9973, 0.176, True),
                ("0.80", 0.9599, 0.177, True),
                ("0.85", 0.8019, 0.197, False),
                ("0.90", 0.2067, 0.560, False),
                ("0.95", 0.0178, 0.861, False)],
        label="DINOv2 ViT-B/14, mean patch tokens (cosine $\\tau$)"),
}

ANNOTATE = {("DINOcls", "0.86"), ("DINOcls", "0.95"), ("dHash", "12"),
            ("EffNet", "0.70"), ("DINOmean", "0.90")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    ax.axhline(BASE_RATE, color="#888888", lw=1.0, ls="-", zorder=1)
    ax.text(0.02, BASE_RATE - 0.022,
            "base rate 0.175: the precision of merging each class entirely",
            ha="left", va="top", fontsize=7.5, color="#555555")

    for key, spec in SERIES.items():
        xs = [p[1] for p in spec["points"]]
        ys = [p[2] for p in spec["points"]]
        ax.plot(xs, ys, spec["ls"], color=spec["colour"], lw=1.3, zorder=2,
                alpha=0.85)
        for lab, r, pr, degen in spec["points"]:
            ax.plot([r], [pr], spec["marker"], ms=6.5, zorder=3,
                    mec=spec["colour"], mew=1.4,
                    mfc="white" if degen else spec["colour"])
            if (key, lab) in ANNOTATE:
                ax.annotate(lab, (r, pr), textcoords="offset points",
                            xytext=(8, 7), fontsize=8, color=spec["colour"])

    ax.set_xlabel("Pair recall\n(share of genuine within-session pairs recovered)")
    ax.set_ylabel("Pair precision\n(share of same-cluster pairs that share a session)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0.09, 1.03)
    ax.grid(True, lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [Line2D([], [], color=s["colour"], ls=s["ls"], marker=s["marker"],
                      ms=6, label=s["label"]) for s in SERIES.values()]
    handles.append(Line2D([], [], color="#555555", ls="none", marker="o", ms=6,
                          mfc="white", mec="#555555",
                          label="degenerate (clustering collapsed)"))
    ax.legend(handles=handles, loc="upper center", fontsize=8, frameon=False,
              bbox_to_anchor=(0.5, -0.28), ncol=1)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        f = out / f"Fig3_proxy_calibration.{ext}"
        fig.savefig(f, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
