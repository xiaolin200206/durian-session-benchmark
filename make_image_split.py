#!/usr/bin/env python3
"""
make_image_split.py — build the image-level control split
=========================================================
The session-level run trains on clean_split. Its control condition must be
the ORIGINAL image-level partition, restricted to the five classes the paper
reports, so that the two runs differ in the split rule and nothing else.

Classication_model_split holds eleven classes: the five disease classes plus
six pest classes added later. This copies out only the five, preserving the
train/val/test assignment exactly as it was.

    python make_image_split.py \
        --src /root/autodl-tmp/Classication_model_split \
        --dst /root/autodl-tmp/image_split
"""

import argparse
import shutil
import sys
from pathlib import Path

PAPER_CLASSES = ["Algal", "Leaf_rot", "Phomopsis", "Pink_disease", "Root_disease"]
SPLITS = ["train", "val", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic"}

# Table 1 of the manuscript, for verification.
EXPECTED = {
    "Algal":        (129, 16, 17),
    "Leaf_rot":     (122, 15, 16),
    "Phomopsis":    (125, 15, 17),
    "Pink_disease": (8, 1, 1),
    "Root_disease": (62, 7, 9),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        sys.exit(f"Not found: {src}")
    if dst.exists():
        sys.exit(f"{dst} exists. Delete it or choose another name.")

    # tolerate case differences in the split folder names
    on_disk = {p.name.lower(): p.name for p in src.iterdir() if p.is_dir()}
    total = 0
    counts = {}

    for split in SPLITS:
        real = on_disk.get(split)
        if real is None:
            sys.exit(f"No '{split}' folder under {src}. Found: {sorted(on_disk.values())}")
        for cls in PAPER_CLASSES:
            cdir = src / real / cls
            if not cdir.is_dir():
                sys.exit(f"Missing class folder: {cdir}")
            out = dst / split / cls
            out.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in sorted(cdir.iterdir()):
                if f.is_file() and f.suffix.lower() in IMG_EXT:
                    shutil.copy2(f, out / f.name)
                    n += 1
            counts[(cls, split)] = n
            total += n

    hdr = f"{'class':<16}{'train':>7}{'val':>6}{'test':>6}{'total':>7}   vs Table 1"
    print(hdr)
    print("-" * len(hdr))
    ok = True
    for cls in PAPER_CLASSES:
        row = tuple(counts[(cls, s)] for s in SPLITS)
        exp = EXPECTED[cls]
        verdict = "match" if row == exp else f"MISMATCH expected {exp}"
        if row != exp:
            ok = False
        print(f"{cls:<16}{row[0]:>7}{row[1]:>6}{row[2]:>6}{sum(row):>7}   {verdict}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<16}{'':>7}{'':>6}{'':>6}{total:>7}   paper says 560")

    if total != 560 or not ok:
        print("\n!! Counts do not reproduce Table 1. Check the source folder "
              "before using this as the control condition.")
        sys.exit(1)
    print(f"\nWritten to {dst.resolve()}")
    print("This is the image-level control. Do NOT deduplicate it — its "
          "contamination is the thing being measured.")


if __name__ == "__main__":
    main()
