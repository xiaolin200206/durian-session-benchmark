#!/usr/bin/env python3
"""
make_paper_figures.py — Figs. 1-7 for the Ecological Informatics submission.

Every number plotted here is traced to a source, named in SOURCE below each
function. Figures whose source is the manuscript tables are reproducible from
the manuscript alone; Fig. 2 is recomputed from sessions.csv.

Fig. 5 is only partially determined by Table 8 (see note in the function).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "figure.dpi": 140,
})

# Greyscale-safe palette (distinguishable when printed in black and white)
C_DARK = "#1b1b1b"
C_MID = "#6e6e6e"
C_LIGHT = "#bdbdbd"
C_ACC = "#c1440e"      # accent, reads dark in greyscale
C_ACC2 = "#2f6690"
MM = 1 / 25.4


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------
# Fig. 1  Sampling hierarchy (schematic; no data)
# ----------------------------------------------------------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(180 * MM, 78 * MM))
    ax.set_xlim(0, 100)
    ax.set_ylim(2, 45)
    ax.axis("off")

    rungs = [
        ("Image", "one photograph", "frame", "one photo", True,
         "How well on another\nview of a specimen\nalready seen?"),
        ("Capture session", "one specimen,\none episode", "trigger burst", "one lesion,\none visit", True,
         "How well on an\nunseen session at a\nfamiliar site?"),
        ("Individual", "one organism", "one animal", "one tree", False,
         "The unit an\nintervention\napplies to."),
        ("Site", "one location", "one station", "one orchard", False,
         "How well at a\ndifferent site?"),
        ("Region", "one production\nregion", "one landscape", "one region", True,
         "Can the model\nbe exported?"),
    ]

    x0, w, gap = 2.5, 17.6, 2.2
    for i, (name, defn, ct, orch, evaluated, q) in enumerate(rungs):
        x = x0 + i * (w + gap)
        face = "#ebe8e3" if evaluated else "#ffffff"
        edge = C_DARK if evaluated else C_LIGHT
        lw = 1.3 if evaluated else 0.8
        ax.add_patch(FancyBboxPatch((x, 23.5), w, 15, boxstyle="round,pad=0.3,rounding_size=0.9",
                                    facecolor=face, edgecolor=edge, linewidth=lw))
        ax.text(x + w / 2, 36.6, name, ha="center", va="center", fontsize=7.4 if len(name) > 10 else 8.2,
                fontweight="bold", color=C_DARK if evaluated else C_MID)
        ax.text(x + w / 2, 33.0, defn, ha="center", va="center", fontsize=6.0, color=C_MID,
                style="italic", linespacing=1.3)
        ax.text(x + w / 2, 27.4, f"trap: {ct}\norchard: {orch}", ha="center", va="center",
                fontsize=5.4, color=C_MID, linespacing=1.4)
        ax.text(x + w / 2, 18.6, q, ha="center", va="top", fontsize=6.2,
                color=C_DARK if evaluated else C_LIGHT, linespacing=1.5)
        if evaluated:
            ax.plot([x + w / 2], [21.6], marker="v", ms=4.2, color=C_ACC)
        if i < len(rungs) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.25, 31), (x + w + gap - 0.25, 31),
                                         arrowstyle="-|>", mutation_scale=8,
                                         color=C_MID, linewidth=0.8))

    ax.text(50, 42.0, "image  \u2282  capture session  \u2282  individual  \u2282  site  \u2282  region",
            ha="center", va="center", fontsize=8.6, color=C_DARK)
    ax.plot([2.5, 97.5], [9.0, 9.0], color=C_LIGHT, lw=0.7)
    ax.text(2.5, 6.6, "Shaded rungs carrying a marker are the three scales evaluated here. The site "
                      "rung was not recoverable (Sect. 3.4) and is the scale a deployment decision "
                      "most needs.",
            ha="left", va="top", fontsize=6.2, color=C_MID)
    save(fig, "Fig1_sampling_hierarchy")


# ----------------------------------------------------------------------------
# Fig. 2  Session structure.  SOURCE: sessions.csv (recomputed)
# ----------------------------------------------------------------------------
def fig2(sessions_csv):
    df = pd.read_csv(sessions_csv)
    pretty = {"Algal": "Algal leaf spot", "Leaf_rot": "Leaf rot",
              "Phomopsis": "Phomopsis leaf spot", "Pink_disease": "Pink disease",
              "Root_disease": "Root and collar rot"}
    order = ["Algal", "Leaf_rot", "Phomopsis", "Pink_disease", "Root_disease"]
    df["type"] = df.session.str.split(":").str[0]

    fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 58 * MM),
                            gridspec_kw={"width_ratios": [1.18, 1, 0.80], "wspace": 0.60})

    # (a) session size distribution
    sizes = df.groupby("session").size().sort_values(ascending=False).values
    ax = axs[0]
    ax.bar(np.arange(1, len(sizes) + 1), sizes, width=0.85, color=C_MID, edgecolor="none")
    ax.axhline(np.median(sizes), color=C_ACC, lw=0.9, ls="--")
    ax.text(len(sizes) * 0.97, np.median(sizes) + 2.0, f"median {int(np.median(sizes))} images",
            color=C_ACC, fontsize=6.6, ha="right")
    ax.annotate(f"largest session\n{sizes.max()} images", xy=(1, sizes.max()),
                xytext=(9, sizes.max() - 6), fontsize=6.6, color=C_DARK,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C_DARK))
    ax.set_xlabel("Capture session (rank)")
    ax.set_ylabel("Images in session")
    ax.set_title("(a)  Session size distribution", loc="left", fontweight="bold", pad=9)
    ax.set_xlim(0, len(sizes) + 1)
    ten = sizes[:10].sum()
    ax.text(0.97, 0.72, f"ten largest sessions\nhold {ten} images ({ten/len(df)*100:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=C_DARK)

    # (b) images vs sessions per class
    ax = axs[1]
    y = np.arange(len(order))
    imgs = [int((df.cls == c).sum()) for c in order]
    sess = [int(df[df.cls == c].session.nunique()) for c in order]
    ax.barh(y + 0.2, imgs, height=0.36, color=C_LIGHT, label="images", edgecolor="none")
    ax.barh(y - 0.2, sess, height=0.36, color=C_DARK, label="capture sessions", edgecolor="none")
    for i, (a, b) in enumerate(zip(imgs, sess)):
        ax.text(a + 4, i + 0.2, str(a), va="center", fontsize=6.6, color=C_MID)
        ax.text(b + 4, i - 0.2, str(b), va="center", fontsize=6.6, color=C_DARK)
    ax.set_yticks(y)
    ax.set_yticklabels([pretty[c] for c in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_xlim(0, 215)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.03, 0.08), fontsize=6.8)
    ax.set_title("(b)  Nominal vs effective $n$", loc="left", fontweight="bold", pad=9)

    # (c) share of class held in its largest session
    ax = axs[2]
    share = []
    for c in order:
        sub = df[df.cls == c]
        share.append(sub.groupby("session").size().max() / len(sub) * 100)
    bars = ax.barh(y, share, height=0.55, color=[C_ACC if s > 45 else C_MID for s in share],
                   edgecolor="none")
    for i, s in enumerate(share):
        ax.text(s + 1.5, i, f"{s:.1f}%", va="center", fontsize=6.8, color=C_DARK)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlim(0, 82)
    ax.set_xlabel("Share of class in its\nlargest session (%)")
    ax.set_title("(c)  Concentration", loc="left", fontweight="bold", pad=9)

    save(fig, "Fig2_session_structure")


# ----------------------------------------------------------------------------
# Fig. 3  Calibration of content-based grouping.  SOURCE: Table 3
# ----------------------------------------------------------------------------
def fig3():
    # (recall, precision, threshold label, degenerate?)
    dhash = [(0.0007, 0.714, "t=6"), (0.0030, 0.913, "t=12")]
    inet = [(0.5254, 0.203, "0.70"), (0.0473, 0.386, "0.80"), (0.0040, 0.933, "0.90")]
    dino_cls = [(0.8685, 0.216, "0.80"), (0.5992, 0.302, "0.85"), (0.5219, 0.322, "0.86"),
                (0.1684, 0.411, "0.88"), (0.0814, 0.488, "0.90"), (0.0056, 0.951, "0.95")]
    dino_mean = [(0.8019, 0.197, "0.85"), (0.2067, 0.560, "0.90"), (0.0178, 0.861, "0.95")]
    BASE = 0.175

    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 72 * MM),
                            gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.42})

    ax = axs[0]
    ax.axhline(BASE, color=C_ACC, lw=0.9, ls="--")
    ax.text(0.985, BASE + 0.018, "base rate 0.175", ha="right", fontsize=6.6, color=C_ACC)

    series = [(dhash, "o", C_DARK, "Difference hash"),
              (inet, "s", C_MID, "EfficientNet-B0 (ImageNet)"),
              (dino_cls, "^", C_ACC2, "DINOv2 ViT-B/14, cls"),
              (dino_mean, "v", C_LIGHT, "DINOv2 ViT-B/14, mean")]
    for pts, mk, col, lab in series:
        r = [p[0] for p in pts]
        p = [p[1] for p in pts]
        ax.plot(r, p, marker=mk, ms=4.2, lw=0.9, color=col, label=lab,
                markerfacecolor="white" if mk == "v" else col, markeredgewidth=0.9)
    # trivial grouping
    ax.plot([1.0], [BASE], marker="*", ms=10, color=C_ACC, markerfacecolor="white",
            markeredgewidth=1.0, linestyle="none", label="Trivial (one cluster per class)")
    # selected operating point
    ax.annotate("selected operating point\n\u03c4 = 0.86, pair-F1 0.399",
                xy=(0.5219, 0.322), xytext=(0.60, 0.52), fontsize=6.6, color=C_DARK,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_DARK))
    ax.set_xlabel("Pair recall (share of same-session pairs placed in one cluster)")
    ax.set_ylabel("Pair precision")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper right", fontsize=6.8, handletextpad=0.5)
    ax.set_title("(a)  Precision\u2013recall vs the reference", loc="left", fontweight="bold", pad=9)

    ax = axs[1]
    labels = ["Reference grouping\n(capture sessions)", "DINOv2 cls\n\u03c4 = 0.86",
              "Difference hash\nt = 12", "Difference hash\nt = 6"]
    vals = [93.9, 66.4, 4.6, 0.8]
    cols = [C_ACC, C_ACC2, C_MID, C_LIGHT]
    bars = ax.barh(np.arange(len(vals)), vals, height=0.58, color=cols, edgecolor="none")
    for i, v in enumerate(vals):
        ax.text(v + 1.6, i, f"{v}%", va="center", fontsize=7.4, color=C_DARK)
    ax.set_yticks(np.arange(len(vals)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("Test images flagged (%)")
    ax.set_title("(b)  What each audit reports", loc="left", fontweight="bold", pad=9)
    ax.text(0.98, 0.02, "side by side, never a ratio",
            transform=ax.transAxes, ha="right", fontsize=6.4, color=C_MID, style="italic")
    save(fig, "Fig3_proxy_calibration")


# ----------------------------------------------------------------------------
# Fig. 4  Paired partition comparison.  SOURCE: Table 7
# ----------------------------------------------------------------------------
def fig4():
    models = ["EfficientNetV2-S", "ResNet-101", "ConvNeXt-Tiny", "EfficientNet-B0 + LFA",
              "VGG-16", "ResNet-50", "MobileNetV3-Large", "EfficientNet-B0", "MobileNetV2"]
    img = [93.4, 94.3, 97.3, 82.1, 92.9, 88.3, 92.4, 83.8, 87.9]
    imgsd = [1.5, 1.1, 2.3, 3.1, 6.0, 3.2, 1.2, 1.9, 2.0]
    ses = [88.5, 85.3, 81.8, 76.1, 76.0, 75.2, 74.0, 73.0, 72.9]
    sessd = [3.1, 5.2, 3.6, 3.7, 6.9, 2.3, 2.7, 2.4, 4.2]

    order = np.argsort(ses)[::-1]
    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 74 * MM),
                            gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.30})

    ax = axs[0]
    y = np.arange(len(models))
    for k, i in enumerate(order):
        ax.plot([ses[i], img[i]], [k, k], color=C_LIGHT, lw=1.4, zorder=1)
        ax.errorbar(ses[i], k, xerr=sessd[i], fmt="o", ms=4.6, color=C_DARK,
                    elinewidth=0.8, capsize=1.8, zorder=3)
        ax.errorbar(img[i], k, xerr=imgsd[i], fmt="o", ms=4.6, color="white",
                    markeredgecolor=C_ACC, markeredgewidth=1.2, ecolor=C_ACC,
                    elinewidth=0.8, capsize=1.8, zorder=3)
        ax.text(img[i] + imgsd[i] + 1.4, k, f"+{img[i]-ses[i]:.1f}", va="center",
                fontsize=6.8, color=C_ACC)
    ax.set_yticks(y)
    ax.set_yticklabels([models[i] for i in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(63, 108)
    ax.set_xlabel("Macro F1 (%), mean \u00b1 s.d. over four seeds")
    ax.plot([], [], "o", color=C_DARK, ms=4.6, label="session-level partition")
    ax.plot([], [], "o", color="white", markeredgecolor=C_ACC, markeredgewidth=1.2,
            ms=4.6, label="image-level partition")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=7)
    ax.set_title("(a)  Only the grouping rule differs", loc="left", fontweight="bold", pad=9)

    # rank reordering
    ax = axs[1]
    r_img = pd.Series(img).rank(ascending=False).values
    r_ses = pd.Series(ses).rank(ascending=False).values
    for i in range(len(models)):
        hl = abs(r_img[i] - r_ses[i]) >= 3
        ax.plot([0, 1], [r_img[i], r_ses[i]], color=C_ACC if hl else C_LIGHT,
                lw=1.4 if hl else 0.8, zorder=2 if hl else 1)
        ax.plot([0], [r_img[i]], "o", ms=3.4, color=C_ACC if hl else C_MID)
        ax.plot([1], [r_ses[i]], "o", ms=3.4, color=C_ACC if hl else C_MID)
        ax.text(-0.045, r_img[i], models[i], ha="right", va="center", fontsize=6.4,
                color=C_ACC if hl else C_MID)
        ax.text(1.045, r_ses[i], models[i], ha="left", va="center", fontsize=6.4,
                color=C_ACC if hl else C_MID)
    ax.set_xlim(-1.25, 2.25)
    ax.set_ylim(9.9, 0.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["image-\nlevel", "session-\nlevel"], fontsize=7.2)
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_title("(b)  Rank ordering ($\\rho$ = 0.65)", loc="left",
                 fontweight="bold", pad=9)
    save(fig, "Fig4_paired_partition")


# ----------------------------------------------------------------------------
# Fig. 5  Confusion matrix, session-level test set.  SOURCE: Table 8
# NOTE: Table 8 determines 23 of the 25 cells exactly. Two off-diagonal cells
# (one Phomopsis error and one pink-disease error, distributed between the
# algal and leaf-rot columns) are underdetermined by the per-class metrics
# alone. They are marked with an asterisk and MUST be replaced from
# export_confusion.py before submission.
# ----------------------------------------------------------------------------
def fig5(cm_csv=None):
    """Confusion matrix, session-level test set, seed 42, ABLATION instance.

    Table 8 of the manuscript reports the ablation instance (macro F1 76.0%),
    so this figure must be drawn from the ablation checkpoint (abl_*), NOT the
    comparison one (cmp_*, macro F1 72.0% at this seed).

        python export_confusion.py --ckpt ckpt/group_s42/abl_agri_efficientnet.pth \
            --split_dir clean_split --out results/group_s42
        python make_paper_figures.py sessions.csv results/group_s42/confusion_matrix.csv

    export_confusion.py orders classes as the sorted training sub-directories,
    i.e. Algal, Leaf_rot, Phomopsis, Pink_disease, Root_disease, which is the
    row order used here.

    Without the CSV the matrix falls back to the reconstruction from Table 8.
    That reconstruction fixes 23 of the 25 cells exactly; the two marked with an
    asterisk (one Phomopsis error and one pink-disease error, distributed between
    the algal and leaf-rot columns) are NOT determined by per-class metrics and
    must not go to a publisher.
    """
    classes = ["Algal leaf spot", "Leaf rot", "Phomopsis\nleaf spot", "Pink disease",
               "Root and\ncollar rot"]
    amb = set()
    if cm_csv and Path(cm_csv).is_file():
        M = np.loadtxt(cm_csv, delimiter=",", dtype=float)
        if M.shape != (5, 5):
            raise SystemExit(f"{cm_csv}: expected a 5x5 matrix, got {M.shape}")
        print(f"  Fig5: measured matrix from {cm_csv}  (total {int(M.sum())} images)")
        if int(M.sum()) != 56:
            print(f"  !! {int(M.sum())} images, expected 56 for the session-level test set")
        f1 = []
        for i in range(5):
            tp = M[i, i]
            prec = tp / M[:, i].sum() if M[:, i].sum() else 0
            rec = tp / M[i].sum() if M[i].sum() else 0
            f1.append(2 * prec * rec / (prec + rec) if prec + rec else 0)
        macro = 100 * sum(f1) / 5
        print(f"  Fig5: macro F1 {macro:.1f}%  "
              f"(Table 8, ablation instance, is 76.0%; the comparison instance is 72.0%)")
        if abs(macro - 76.0) > 1.0:
            print("  !! does not match Table 8 — wrong checkpoint? use abl_*, not cmp_*")
    else:
        print("  Fig5: NO CSV — using the matrix recovered from the archived "
              "ablation-instance figure (see comment)")
        # rows = true, cols = predicted.
        #
        # Fallback only. The figure as submitted is drawn from
        # results/group_s42/confusion_matrix.csv, produced by:
        #   python export_confusion.py \
        #       --ckpt ckpt/group_s42/abl_lfa.pth \
        #       --split_dir clean_split_512 \
        #       --out results/group_s42
        # which reproduces Table 8 exactly: precision 0.833 / 0.556 / 1.000 /
        # 1.000 / 1.000, recall 0.312 / 1.000 / 0.933 / 0.500 / 1.000, macro F1
        # 0.760, accuracy 0.768 over 56 images.
        #
        # Note the two arguments that matter and are easy to get wrong:
        #   abl_lfa.pth      not cmp_* (the comparison instance scores 72.0%)
        #   clean_split_512  not clean_split (Sect. 3.6: the same checkpoint
        #                    evaluated on the full-resolution originals moves
        #                    the held-out macro F1 from 72.0% to 77.6%)
        M = np.array([
            [5, 11, 0, 0, 0],   # algal     (16)
            [0, 15, 0, 0, 0],   # leaf rot  (15)
            [1, 0, 14, 0, 0],   # phomopsis (15): 1 -> algal
            [0, 1, 0, 1, 0],    # pink       (2): 1 -> leaf rot
            [0, 0, 0, 0, 8],    # root       (8)
        ], dtype=float)
    support = M.sum(axis=1, keepdims=True)
    N = M / support

    fig, ax = plt.subplots(figsize=(105 * MM, 92 * MM))
    im = ax.imshow(N, cmap="Greys", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            if M[i, j] == 0:
                continue
            txt = f"{int(M[i,j])}" + ("*" if (i, j) in amb else "")
            txt += f"\n{N[i,j]*100:.0f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if N[i, j] > 0.55 else C_DARK)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(classes, fontsize=6.8, rotation=30, ha="right")
    ax.set_yticklabels([c + f"\n(n={int(s)})" for c, s in zip(classes, support.ravel())],
                       fontsize=6.8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Session-level test set (n = 56), seed 42", loc="left", fontweight="bold")
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.7)
    if amb:
        ax.text(0, 5.95, "* cell not determined by the per-class metrics of Table 8; "
                         "replace from export_confusion.py",
                fontsize=6.0, color=C_ACC, ha="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
    cb.set_label("Share of true class", fontsize=7)
    cb.outline.set_linewidth(0.6)
    save(fig, "Fig5_confusion_session")


# ----------------------------------------------------------------------------
# Fig. 6  Cross-region transfer.  SOURCE: Table 10, Table S3
# ----------------------------------------------------------------------------
def fig6():
    models = ["EfficientNetV2-S", "ResNet-101", "EfficientNet-B0 + LFA", "MobileNetV2",
              "VGG-16", "ConvNeXt-Tiny", "EfficientNet-B0", "ResNet-50", "MobileNetV3-Large"]
    my = [80.8, 78.5, 73.3, 72.5, 71.8, 69.8, 67.2, 67.2, 66.2]
    mysd = [5.2, 5.1, 4.5, 8.4, 11.7, 6.0, 4.9, 3.2, 5.3]
    vn = [42.9, 42.3, 32.8, 30.2, 44.7, 48.9, 28.4, 34.5, 35.5]
    vnsd = [1.9, 1.3, 2.1, 5.2, 3.4, 2.0, 2.0, 5.8, 4.9]
    CHANCE = 32.5

    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 72 * MM),
                            gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.40})
    ax = axs[0]
    x = np.arange(len(models))
    ax.axhspan(0, CHANCE, color="#f0f0f0", zorder=0)
    ax.axhline(CHANCE, color=C_ACC, lw=1.0, ls="--", zorder=1)
    ax.text(len(models) - 0.35, CHANCE + 1.2, "chance, 3-class task (32.5%)",
            ha="right", fontsize=6.8, color=C_ACC)
    ax.bar(x - 0.19, my, 0.36, yerr=mysd, color=C_DARK, edgecolor="none",
           error_kw=dict(elinewidth=0.7, capsize=1.6, ecolor=C_MID), label="Malaysia (source)")
    ax.bar(x + 0.19, vn, 0.36, yerr=vnsd, color=C_LIGHT, edgecolor="none",
           error_kw=dict(elinewidth=0.7, capsize=1.6, ecolor=C_MID), label="Vietnam (zero-shot)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=32, ha="right", fontsize=6.6)
    ax.set_ylabel("Macro F1 (%), three foliar correspondences")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper right", ncol=1)
    ax.set_title("(a)  Source against zero-shot target", loc="left", fontweight="bold", pad=9)

    ax = axs[1]
    drops = [m - v for m, v in zip(my, vn)]
    ax.scatter(my, drops, s=26, color=C_ACC2, zorder=3)
    for m, d, lab in zip(my, drops, models):
        ax.annotate(lab.replace(" + LFA", "+LFA"), (m, d), textcoords="offset points",
                    xytext=(4, 4), fontsize=5.8, color=C_MID)
    ax.set_xlabel("Malaysia macro F1 (%)")
    ax.set_ylabel("Drop on transfer (pp)")
    ax.set_xlim(62, 86)
    ax.set_ylim(14, 50)
    ax.set_title("(b)  Loss against home score", loc="left", fontweight="bold", pad=9)
    r = np.corrcoef(my, drops)[0, 1]
    ax.text(0.97, 0.97, f"Pearson r = {r:.2f} (n = 9)", transform=ax.transAxes, ha="right",
            va="top", fontsize=7, color=C_DARK)
    # matched centre-crop control
    ax.annotate("matched centre-crop control narrows\nthe mean gap 40.5 \u2192 22.1 pp (single seed)",
                xy=(0.03, 0.13), xycoords="axes fraction", fontsize=6.2, color=C_ACC,
                va="top")
    save(fig, "Fig6_cross_region")


# ----------------------------------------------------------------------------
# Fig. 7  Zero-shot per-class behaviour.  SOURCE: Tables 12 and 13
# ----------------------------------------------------------------------------
def fig7():
    classes = ["Algal leaf spot", "Leaf rot", "Phomopsis leaf spot", "Root and collar rot"]
    zs_recall = [84.2, 43.3, 13.7, 1.5]
    indom_recall = [97.1, 88.9, 95.2, 98.4]
    pred_share = [49.8, 32.9, 10.5, 1.1]
    label_share = [21.9, 39.4, 19.7, 19.1]

    fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 68 * MM),
                            gridspec_kw={"wspace": 0.36})
    x = np.arange(len(classes))
    ax = axs[0]
    ax.bar(x - 0.19, indom_recall, 0.36, color=C_LIGHT, edgecolor="none",
           label="in-domain control (trained on target)")
    ax.bar(x + 0.19, zs_recall, 0.36, color=C_DARK, edgecolor="none",
           label="zero-shot from Malaysian weights")
    for xi, v in zip(x + 0.19, zs_recall):
        ax.text(xi, v + 2, f"{v}", ha="center", fontsize=6.8, color=C_DARK)
    ax.annotate("correspondence rejected\n(Sect. 4.7): organ mismatch,\nleaf \u2192 trunk and collar",
                xy=(3.19, 3), xytext=(2.15, 40), fontsize=6.4, color=C_ACC,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=C_ACC))
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n", 1) for c in classes], fontsize=6.8)
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 118)
    ax.legend(frameon=False, loc="upper center", fontsize=6.6, bbox_to_anchor=(0.5, 1.02))
    ax.set_title("(a)  Recall in domain and zero-shot", loc="left", fontweight="bold", pad=9)

    ax = axs[1]
    ax.bar(x - 0.19, label_share, 0.36, color=C_LIGHT, edgecolor="none", label="share of labels")
    ax.bar(x + 0.19, pred_share, 0.36, color=C_ACC2, edgecolor="none",
           label="share of predictions")
    ax.axhline(0, color=C_DARK, lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n", 1) for c in classes], fontsize=6.8)
    ax.set_ylabel("Share of the 320 target images (%)")
    ax.set_ylim(0, 62)
    ax.legend(frameon=False, loc="upper right", fontsize=6.8)
    ax.text(0.98, 0.62, "pink disease, absent from the\ntarget, takes 5.7% of predictions",
            transform=ax.transAxes, ha="right", fontsize=6.3, color=C_ACC)
    ax.set_title("(b)  Predictions against labels", loc="left", fontweight="bold", pad=9)
    save(fig, "Fig7_zeroshot_behaviour")


if __name__ == "__main__":
    import sys
    sessions = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "sessions.csv")
    cm_csv = sys.argv[2] if len(sys.argv) > 2 else "results/group_s42/confusion_matrix.csv"
    fig1()
    fig2(sessions)
    fig3()
    fig4()
    fig5(cm_csv)
    fig6()
    fig7()
    print("\nAll figures written to", OUT)
