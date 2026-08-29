#!/usr/bin/env python3
"""
Label conflict + paper-subset audit
===================================
Follow-up to audit_dataset.py.  Answers two questions:

  A. Is the 5-class dataset used in the paper (560 images) free of
     cross-split duplication?  Run with --paper-only.

  B. How bad is the multi-label contamination in the full 11-class tree?
     Which class pairs are entangled, and how many files are affected?

Only needs Pillow:   pip install pillow

Usage:
    python label_conflicts.py --paper-only --threshold 10
    python label_conflicts.py                      # full 11-class report
    python label_conflicts.py --export-clean clean_split   # write deduped copy
"""

import argparse
import csv
import hashlib
import os
import shutil
import sys
from collections import defaultdict, Counter

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow not installed.  Run:  pip install pillow")

DEFAULT_ROOT = r"C:\Users\Lim Ding Shan\Desktop\Durian project and paper\Classication_model_split"

SPLITS = ["train", "val", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic"}

# The five classes reported in Table 1 of the manuscript.
PAPER_CLASSES = {"algal", "leaf_rot", "phomopsis", "pink_disease", "root_disease"}


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dhash(img, size=8):
    small = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(small.getdata())
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits <<= 1
            if px[base + col] > px[base + col + 1]:
                bits |= 1
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


def scan(root, paper_only, exclude=()):
    excl = {c.lower() for c in exclude}
    recs = []
    for split in SPLITS:
        sdir = os.path.join(root, split)
        if not os.path.isdir(sdir):
            continue
        for cls in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, cls)
            if not os.path.isdir(cdir):
                continue
            if paper_only and cls.lower() not in PAPER_CLASSES:
                continue
            if cls.lower() in excl:
                continue
            for fn in sorted(os.listdir(cdir)):
                fp = os.path.join(cdir, fn)
                if not os.path.isfile(fp):
                    continue
                if os.path.splitext(fn)[1].lower() not in IMG_EXT:
                    continue
                try:
                    with Image.open(fp) as im:
                        im.load()
                        dh = dhash(im)
                except Exception:
                    continue
                recs.append({"split": split, "cls": cls, "file": fn,
                             "path": fp, "md5": md5_of(fp), "dhash": dh})
    return recs


BAR = "=" * 72


def sec(t):
    print(f"\n{BAR}\n{t}\n{BAR}")


def conflict_report(recs):
    """Identical file (same MD5) sitting under two or more different classes."""
    by_md5 = defaultdict(list)
    for r in recs:
        by_md5[r["md5"]].append(r)

    conflicts, split_dupes = [], []
    for md5, group in by_md5.items():
        if len(group) < 2:
            continue
        classes = {r["cls"] for r in group}
        splits = {r["split"] for r in group}
        if len(classes) > 1:
            conflicts.append((md5, group, classes, splits))
        elif len(splits) > 1:
            split_dupes.append((md5, group))

    sec("A. LABEL CONFLICTS  (same file, different class labels)")
    print("These are NOT burst shots. The identical bytes carry two or more")
    print("contradictory single-label class assignments.\n")
    n_files = sum(len(g) for _, g, _, _ in conflicts)
    print(f"conflicting groups          : {len(conflicts)}")
    print(f"file copies involved        : {n_files}")
    print(f"unique underlying images    : {len(conflicts)}")

    pairs = Counter()
    for _, _, classes, _ in conflicts:
        cl = sorted(classes)
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                pairs[(cl[i], cl[j])] += 1

    if pairs:
        print("\nmost entangled class pairs:")
        for (a, b), n in pairs.most_common(15):
            print(f"    {n:>4}   {a}  <->  {b}")

    per_class = Counter()
    for _, group, _, _ in conflicts:
        for r in group:
            per_class[r["cls"]] += 1
    if per_class:
        print("\nconflicting copies per class:")
        for c, n in per_class.most_common():
            print(f"    {n:>4}   {c}")

    sec("B. TRUE SPLIT DUPLICATES  (same file, SAME class, different split)")
    print("These are straightforward leakage: identical image in train and")
    print("in a held-out split under the same label.\n")
    if not split_dupes:
        print("None. Good.")
    else:
        print(f"!! {len(split_dupes)} groups\n")
        for _, g in split_dupes:
            for r in g:
                print(f"    {r['split']}/{r['cls']}/{r['file']}")
            print()
    return conflicts, split_dupes


def near_dup_report(recs, threshold):
    sec(f"C. NEAR DUPLICATES ACROSS SPLITS  (dHash <= {threshold})")
    train = [r for r in recs if r["split"] == "train"]
    held = [r for r in recs if r["split"] in ("val", "test")]
    hits = []
    for h in held:
        for t in train:
            if h["md5"] == t["md5"]:
                continue
            d = hamming(h["dhash"], t["dhash"])
            if d <= threshold:
                hits.append((d, h, t))
    hits.sort(key=lambda x: x[0])

    same_cls = [x for x in hits if x[1]["cls"] == x[2]["cls"]]
    print(f"pairs found                 : {len(hits)}")
    print(f"  ... same class (leakage)  : {len(same_cls)}")
    print(f"  ... different class       : {len(hits)-len(same_cls)}\n")
    for d, h, t in hits:
        flag = "  <-- SAME CLASS, LEAKAGE" if h["cls"] == t["cls"] else ""
        print(f"  dist {d:>2}  {h['split']}/{h['cls']}/{h['file']}")
        print(f"           train/{t['cls']}/{t['file']}{flag}")
    if hits:
        print("\n  Open these by eye. Distance 0-2 is almost always the same")
        print("  photograph; 3-10 needs a look.")
    return hits


def export_clean(recs, conflicts, split_dupes, near_hits, out_root):
    """Write a deduplicated copy: drop every file involved in a conflict,
    and for same-class duplicates keep only the training copy."""
    drop = set()
    for _, group, _, _ in conflicts:
        for r in group:
            drop.add(r["path"])
    for _, group in split_dupes:
        for r in group:
            if r["split"] != "train":
                drop.add(r["path"])
    for _, h, _ in near_hits:
        if h["cls"] == h["cls"]:
            drop.add(h["path"])

    kept = 0
    for r in recs:
        if r["path"] in drop:
            continue
        dst = os.path.join(out_root, r["split"], r["cls"])
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(r["path"], os.path.join(dst, r["file"]))
        kept += 1

    sec("D. CLEAN EXPORT")
    print(f"dropped : {len(drop)}")
    print(f"kept    : {kept}")
    print(f"written to: {os.path.abspath(out_root)}")
    print("\nRe-run audit_dataset.py against the clean copy to confirm.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--paper-only", action="store_true",
                    help="restrict to the 5 classes reported in Table 1")
    ap.add_argument("--threshold", type=int, default=10)
    ap.add_argument("--export-clean", default=None,
                    help="write a deduplicated copy to this folder")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="class folder names to drop entirely, e.g. Red_spider")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"Folder not found: {args.root}")

    recs = scan(args.root, args.paper_only, args.exclude)
    scope = "5 paper classes only" if args.paper_only else "all classes"
    if args.exclude:
        scope += f"  (excluding: {', '.join(args.exclude)})"
    print(f"Scanning: {args.root}")
    print(f"Scope   : {scope}")
    print(f"Images  : {len(recs)}")

    counts = defaultdict(lambda: defaultdict(int))
    for r in recs:
        counts[r["cls"]][r["split"]] += 1
    print()
    for c in sorted(counts):
        row = counts[c]
        print(f"  {c:<22}{row.get('train',0):>6}{row.get('val',0):>5}"
              f"{row.get('test',0):>5}{sum(row.values()):>7}")

    conflicts, split_dupes = conflict_report(recs)
    near_hits = near_dup_report(recs, args.threshold)

    if args.export_clean:
        export_clean(recs, conflicts, split_dupes, near_hits, args.export_clean)

    sec("VERDICT")
    if args.paper_only:
        if not conflicts and not split_dupes and not [
                x for x in near_hits if x[1]["cls"] == x[2]["cls"]]:
            print("The 560-image paper dataset is clean. You can state in")
            print("Section 3.1 that splits were verified free of exact and")
            print("perceptual duplicates, and report the check as a method step.")
        else:
            print("The paper dataset has contamination. Fix before submission:")
            print("  - remove the offending files")
            print("  - re-split by capture session, not by random image")
            print("  - re-run every experiment that touches the test set")
    else:
        print("Full-tree report. Label conflicts must be resolved before this")
        print("dataset is trained on or released. Decide per class pair whether")
        print("the overlap is genuine co-occurrence (-> multi-label task) or a")
        print("filing error (-> pick one label and delete the other copies).")


if __name__ == "__main__":
    main()
