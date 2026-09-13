#!/usr/bin/env python3
"""
make_figS5_similarity.py — why a calibrated threshold does not transfer
=======================================================================
Section 4.2 explains the collapse of the τ = 0.86 operating point on the
Vietnamese dataset by its preprocessing: background-removed, uniformly
resized images compress within-class cosine similarity into a narrow high
band, while raw field photographs spread it out. That is stated as a
mechanism visible in the data. This script measures it, so the sentence
rests on a figure rather than on plausibility.

It plots the distribution of within-class pairwise cosine similarity for two
or more datasets on one axis, with the chosen threshold marked. If the
explanation is right, the external dataset's mass sits above the threshold
and the source dataset's straddles it.

    python make_figS5_similarity.py \\
        --dataset "Malaysia=<zenodo>/images_512:<zenodo>/dinov2_cls.npz" \\
        --dataset "VN-A=Durian_Leaf_Diseases:vna_dino.npz" \\
        --threshold 0.86 --out figures

Each --dataset is NAME=IMAGE_ROOT:EMBEDDING_NPZ. The image root supplies the
class labels (similarity is computed within class only, as in the
clustering); the npz supplies the features, and must have been extracted on
that root with extract_features.py.

Pairs are subsampled when a class is large, since the point is the shape of
the distribution, not an exact count. The subsample size and seed are
printed and recorded in the caption line the script writes.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib missing.  pip install matplotlib")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TRAIN_NAMES = {"train", "training", "train_set"}
VAL_NAMES = {"val", "valid", "validation", "dev"}
TEST_NAMES = {"test", "testing", "test_set", "eval"}


def normalise_split(name):
    n = name.strip().lower().replace("-", "_").replace(" ", "_")
    if n in TRAIN_NAMES:
        return "train"
    if n in VAL_NAMES:
        return "val"
    if n in TEST_NAMES:
        return "test"
    return None


def scan(root):
    """Identical to extract_features.py, so npz row order matches."""
    root = Path(root)
    top = [d for d in sorted(root.iterdir()) if d.is_dir()]
    split_dirs = [d for d in top if normalise_split(d.name)]
    recs = []
    if len(split_dirs) >= 2:
        for d in split_dirs:
            for cls_dir in sorted(d.iterdir()):
                if cls_dir.is_dir():
                    for f in sorted(cls_dir.rglob("*")):
                        if f.is_file() and f.suffix.lower() in IMG_EXT:
                            recs.append((cls_dir.name.strip(), str(f)))
    else:
        for cls_dir in top:
            for f in sorted(cls_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in IMG_EXT:
                    recs.append((cls_dir.name.strip(), str(f)))
    return recs


def within_class_similarities(E, classes, max_pairs_per_class, rng):
    by_cls = defaultdict(list)
    for i, c in enumerate(classes):
        by_cls[c].append(i)
    out = []
    for cls, idxs in by_cls.items():
        idxs = np.asarray(idxs)
        m = len(idxs)
        if m < 2:
            continue
        total = m * (m - 1) // 2
        S = E[idxs] @ E[idxs].T
        iu = np.triu_indices(m, k=1)
        vals = S[iu]
        if total > max_pairs_per_class:
            sel = rng.choice(len(vals), size=max_pairs_per_class, replace=False)
            vals = vals[sel]
        out.append(vals)
    if not out:
        sys.exit("No class had two or more images.")
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True,
                    metavar="NAME=IMAGE_ROOT:EMBEDDING_NPZ")
    ap.add_argument("--threshold", type=float, default=0.86)
    ap.add_argument("--max-pairs-per-class", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--name", default="FigS5_similarity_distributions")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    series = []
    for spec in args.dataset:
        if "=" not in spec or ":" not in spec.split("=", 1)[1]:
            sys.exit(f"--dataset needs NAME=IMAGE_ROOT:EMBEDDING_NPZ, got {spec}")
        name, rest = spec.split("=", 1)
        # rsplit so Windows drive letters in the root do not break the split
        root, npz = rest.rsplit(":", 1)
        recs = scan(root)
        if not recs:
            sys.exit(f"No images under {root}")
        z = np.load(npz, allow_pickle=True)
        index = {p: i for i, p in enumerate(list(z["paths"]))}
        missing = [p for _, p in recs if p not in index]
        if missing:
            sys.exit(f"{npz}: {len(missing)} images absent, e.g. {missing[0]}. "
                     f"Re-extract against {root}.")
        rows = np.array([index[p] for _, p in recs])
        E = np.asarray(z["E"], dtype=np.float32)[rows]
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        classes = [c for c, _ in recs]
        vals = within_class_similarities(E, classes, args.max_pairs_per_class, rng)
        tag = str(z["weights"]) if "weights" in z else Path(npz).name
        series.append((name.strip(), vals, tag))
        above = 100.0 * float((vals >= args.threshold).mean())
        print(f"{name.strip():<12} {len(recs):>6} images, "
              f"{len(vals):>9} within-class pairs, "
              f"median {np.median(vals):.3f}, "
              f"{above:5.1f}% at or above {args.threshold}   [{tag}]")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colours = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    for i, (name, vals, _) in enumerate(series):
        above = 100.0 * float((vals >= args.threshold).mean())
        ax.hist(vals, bins=args.bins, density=True, histtype="stepfilled",
                alpha=0.35, color=colours[i % len(colours)])
        ax.hist(vals, bins=args.bins, density=True, histtype="step", lw=1.6,
                color=colours[i % len(colours)],
                label=f"{name} — {above:.1f}% of pairs at or above τ")

    ax.axvline(args.threshold, color="#333333", lw=1.4, ls="--")
    ax.text(args.threshold, ax.get_ylim()[1] * 0.96,
            f"  τ = {args.threshold} (calibrated on the source dataset)",
            ha="left", va="top", fontsize=8, color="#333333")

    ax.set_xlabel("Cosine similarity between within-class image pairs "
                  "(DINOv2 ViT-B/14, class token)")
    ax.set_ylabel("Density")
    ax.grid(True, lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        f = out / f"{args.name}.{ext}"
        fig.savefig(f, dpi=300, bbox_inches="tight")
        print(f"wrote {f}")

    print("\nCaption line to paste, with the bracketed values checked against "
          "the printout above:")
    print(f"  Fig. S5 Distribution of within-class pairwise cosine similarity "
          f"under DINOv2 ViT-B/14 class-token features, for "
          f"{' and '.join(n for n, _, _ in series)}. The dashed line marks the "
          f"operating point τ = {args.threshold} calibrated on the Malaysian "
          f"dataset (Sect. 4.2). Pairs are computed within class and "
          f"subsampled at most {args.max_pairs_per_class:,} per class "
          f"(seed {args.seed}).")


if __name__ == "__main__":
    main()
