#!/usr/bin/env python3
"""
plantvillage_leafmap.py — turn PlantVillage's recorded leaf identifiers into a
manifest this repository's tooling can read
=============================================================================
Mohanty et al. (2016) recorded which images depict the same physical leaf and
kept every image of a leaf on one side of each train/test split. Those
identifiers are public, in `leaf-map.json` at the root of

    https://github.com/spMohanty/PlantVillage-Dataset

This matters for the survey argument in a way the hash-based rows of
`leakage_survey.py` cannot. Everywhere else in this project the grouping is a
proxy: reconstructed from filenames on our own data, inferred from image
content on other people's. PlantVillage is the one large public dataset where
the grouping was *recorded at collection*. A leakage measurement there is not
a proxy measurement, and it is made on ~42,000 images rather than 560.

What this script does
---------------------
1. Reads `leaf-map.json` and the PlantVillage image tree.
2. Writes `plantvillage_sessions.csv` with the SAME three columns as
   `sessions.csv` — cls, file, session — so that `session_utils.load_sessions`,
   `grouping_sensitivity.py`, `leak_detection.py` and `train.py --cv_mode`
   all take it unchanged.
3. Reports what an image-level stratified split would leak against the
   recorded grouping, the same quantity Sect. 4.1 of the paper reports for the
   durian data.

Two facts about the leaf map that the naive reading gets wrong
--------------------------------------------------------------
* **The key is not a filename.** PlantVillage filenames look like
  `<uuid>___RS_Early.B 7557.JPG`; the leaf-map key is the lowercased part
  after `___`, without the extension: `rs_early.b 7557`.
* **The key is not unique across classes.** 2,172 keys appear under two
  different classes — `rs_hl 6251` is both `Soybean___healthy` leaf 398 and
  `Apple___healthy` leaf 101. Disambiguate by the class folder the file
  actually sits in, which is why this script needs `--root` and not only the
  JSON.
* **Leaf ids are reused across classes** (532 of 1,602 raw ids). The group key
  must therefore be (class, leaf id), not the leaf id alone.

Usage
-----
    # get the identifiers (2.3 MB, no images)
    curl -LO https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/leaf-map.json

    # images: either the GitHub repo's raw/color tree, or
    #   huggingface.co/datasets/mohanty/PlantVillage  (has a leaf_id field)

    python plantvillage_leafmap.py \
        --leafmap leaf-map.json \
        --root /path/to/PlantVillage/color \
        --out plantvillage_sessions.csv

    # then everything else in this repo works on it:
    python grouping_sensitivity.py --sessions plantvillage_sessions.csv --out pv_sensitivity

Without --root the script runs in `--map-only` mode: it measures leakage from
the leaf map alone, treating every (class, key) pair as one image. That is an
estimate, not a measurement on the files, and it is labelled as such in the
output. Use it to decide whether the full run is worth doing; do not put it in
a paper.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas missing.  pip install pandas")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".JPG", ".JPEG"}


def load_leafmap(path):
    """leaf-map.json: {image_key: ["Class:::leafid", ...]}  ->  {(cls, key): leafid}"""
    raw = json.loads(Path(path).read_text())
    out = {}
    for key, entries in raw.items():
        for e in entries:
            cls, leaf = e.rsplit(":::", 1)
            out[(cls, key.strip().lower())] = leaf
    return out


def key_from_filename(name):
    """`<uuid>___RS_Early.B 7557.JPG` -> `rs_early.b 7557`."""
    stem = Path(name).stem
    if "___" not in stem:
        return None
    return stem.split("___", 1)[1].strip().lower()


def build_from_tree(root, lm):
    root = Path(root)
    rows, unmatched = [], defaultdict(int)
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        cls = cdir.name
        for f in sorted(cdir.iterdir()):
            if not f.is_file() or f.suffix not in IMG_EXT:
                continue
            k = key_from_filename(f.name)
            leaf = lm.get((cls, k)) if k else None
            if leaf is None:
                unmatched[cls] += 1
                continue
            rows.append((cls, f.name, f"leaf:{cls}:{leaf}"))
    return pd.DataFrame(rows, columns=["cls", "file", "session"]), dict(unmatched)


def build_from_map(lm):
    rows = [(cls, key, f"leaf:{cls}:{leaf}") for (cls, key), leaf in lm.items()]
    return pd.DataFrame(rows, columns=["cls", "file", "session"]), {}


def leak(classes, groups, seeds, ratios=(0.8, 0.1, 0.1)):
    """Share of test images whose group also appears in training, under a
    stratified image-level split. Same definition as grouping_sensitivity.py."""
    classes = np.asarray(classes)
    g = np.asarray(groups)
    out = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        a = np.empty(len(classes), dtype=object)
        for c in np.unique(classes):
            idx = np.where(classes == c)[0]
            rng.shuffle(idx)
            n = len(idx)
            ntr = int(round(n * ratios[0]))
            nva = int(round(n * ratios[1]))
            a[idx[:ntr]] = "train"
            a[idx[ntr:ntr + nva]] = "val"
            a[idx[ntr + nva:]] = "test"
        tr = set(g[a == "train"])
        te = a == "test"
        out.append(100 * sum(1 for x in g[te] if x in tr) / te.sum())
    return float(np.mean(out)), float(np.std(out, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leafmap", required=True, help="leaf-map.json")
    ap.add_argument("--root", help="PlantVillage class-folder tree (color/). "
                                   "Omit for --map-only estimate.")
    ap.add_argument("--out", default="plantvillage_sessions.csv")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    lm = load_leafmap(args.leafmap)
    print(f"leaf map: {len(lm):,} (class, key) pairs, "
          f"{len({c for c, _ in lm}):,} classes")

    if args.root:
        df, unmatched = build_from_tree(args.root, lm)
        mode = "MEASURED on the image tree"
        if unmatched:
            tot = sum(unmatched.values())
            print(f"\n!! {tot:,} files carried no leaf identifier and were dropped.")
            for c, n in sorted(unmatched.items(), key=lambda x: -x[1])[:8]:
                print(f"     {n:>6}  {c}")
            print("   The leaf map does not cover the whole of PlantVillage. Report "
                  "the covered subset as the denominator, not 54,306.")
    else:
        df, _ = build_from_map(lm)
        mode = "ESTIMATED from the leaf map alone -- not a measurement on files"

    if df.empty:
        sys.exit("No rows built. Check --root points at the class-folder tree.")

    sz = df.groupby("session").size()
    print(f"\n{mode}")
    print(f"  images        {len(df):,}")
    print(f"  leaf groups   {df.session.nunique():,}")
    print(f"  images/leaf   {len(df) / df.session.nunique():.2f}")
    print(f"  median {int(sz.median())}   max {int(sz.max())}   "
          f"singletons {(sz == 1).sum():,} ({(sz == 1).mean() * 100:.1f}%)")

    m, s = leak(df.cls.values, df.session.values, range(args.seeds))
    print(f"\n  Under a stratified IMAGE-LEVEL split, {m:.1f}% +/- {s:.1f} of test "
          f"images\n  share a physical leaf with the training set "
          f"({args.seeds} seeds).")
    print("  (Mohanty et al. 2016 grouped by leaf and did not incur this. "
          "Work that\n   replicated their accuracies without stating a grouping "
          "rule may have.)")

    per = df.groupby("cls").agg(images=("file", "size"), leaves=("session", "nunique"))
    per["images_per_leaf"] = (per.images / per.leaves).round(2)
    per = per.sort_values("images", ascending=False)
    print("\n  Ten largest classes:")
    print(per.head(10).to_string(header=True))

    df.to_csv(args.out, index=False)
    per.to_csv(Path(args.out).with_name(Path(args.out).stem + "_per_class.csv"))
    print(f"\nmanifest -> {args.out}   (columns cls, file, session — feed it to "
          f"grouping_sensitivity.py)")


if __name__ == "__main__":
    main()
