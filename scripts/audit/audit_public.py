#!/usr/bin/env python3
"""
audit_public.py — audit a public dataset's own published split
==============================================================
Works on any root/<split>/<class>/*.img layout. Split folder names are
detected automatically, so Train/Test/Validation and train/val/test both
work.

Checks, in order of severity:

  1. Whole-split overlap   — are two splits the same files?
  2. Exact duplicates      — identical bytes across splits (MD5)
  3. Near duplicates       — same lesion, different frame (dHash)
  4. Numbering adjacency   — consecutive filename numbers across splits,
                             which indicates capture order was randomised
                             at image level rather than by session
  5. Preprocessing mix     — resolutions and formats that differ within a
                             class, e.g. some images pre-resized to 224x224

Needs Pillow:  pip install pillow

Usage:
    python audit_public.py --root "Durian_Leaf_Diseases"
    python audit_public.py --root "Ten_Classes_of_Durian_Leaf_Diseases"
    python audit_public.py --root ... --threshold 8 --sample 400
"""

import argparse
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow not installed.  Run:  pip install pillow")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RE_NUM = re.compile(r"(\d+)")

BAR = "=" * 72


def sec(t):
    print(f"\n{BAR}\n{t}\n{BAR}")


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
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


def find_splits(root):
    """Any immediate subfolder that itself contains class subfolders."""
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        if any(os.path.isdir(os.path.join(p, c)) for c in os.listdir(p)):
            out.append(d)
    return out


def scan(root, splits, want_hash):
    recs = []
    for sp in splits:
        sdir = os.path.join(root, sp)
        for cls in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, cls)
            if not os.path.isdir(cdir):
                continue
            for fn in sorted(os.listdir(cdir)):
                fp = os.path.join(cdir, fn)
                if not os.path.isfile(fp):
                    continue
                if os.path.splitext(fn)[1].lower() not in IMG_EXT:
                    continue
                r = {"split": sp, "cls": cls.strip(), "file": fn, "path": fp,
                     "md5": md5_of(fp)}
                if want_hash:
                    try:
                        with Image.open(fp) as im:
                            im.load()
                            r["dhash"] = dhash(im)
                            r["wh"] = im.size
                    except Exception:
                        r["dhash"] = None
                        r["wh"] = None
                recs.append(r)
    return recs


def check_split_overlap(recs, splits):
    sec("1. WHOLE-SPLIT OVERLAP")
    print("Do two splits contain the same files? Compared by content (MD5),")
    print("per class, so a shared filename with different bytes will not")
    print("register as an overlap.\n")

    by = defaultdict(lambda: defaultdict(set))
    for r in recs:
        by[r["cls"]][r["split"]].add(r["md5"])

    flagged = False
    for cls in sorted(by):
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                a, b = splits[i], splits[j]
                sa, sb = by[cls].get(a, set()), by[cls].get(b, set())
                if not sa or not sb:
                    continue
                inter = sa & sb
                if not inter:
                    continue
                flagged = True
                pct_a = 100 * len(inter) / len(sa)
                pct_b = 100 * len(inter) / len(sb)
                verdict = ""
                if pct_a > 95 and pct_b > 95:
                    verdict = "   <-- effectively the SAME SET"
                print(f"  {cls:<28} {a} n={len(sa):<5} {b} n={len(sb):<5}")
                print(f"    shared: {len(inter)}  "
                      f"({pct_a:.0f}% of {a}, {pct_b:.0f}% of {b}){verdict}")
    if not flagged:
        print("No content overlap between any pair of splits.")
    return flagged


def check_exact(recs):
    sec("2. EXACT DUPLICATES ACROSS SPLITS")
    by_md5 = defaultdict(list)
    for r in recs:
        by_md5[r["md5"]].append(r)
    groups = [g for g in by_md5.values()
              if len(g) > 1 and len({x["split"] for x in g}) > 1]
    total = sum(len(g) for g in groups)
    print(f"groups spanning splits : {len(groups)}")
    print(f"file copies involved   : {total} "
          f"({100*total/max(len(recs),1):.1f}% of the dataset)")
    same_cls = [g for g in groups if len({x['cls'] for x in g}) == 1]
    diff_cls = [g for g in groups if len({x['cls'] for x in g}) > 1]
    print(f"  same class (leakage) : {len(same_cls)}")
    print(f"  different class      : {len(diff_cls)}   <-- label conflict")
    for g in diff_cls[:15]:
        print()
        for r in g:
            print(f"    {r['split']}/{r['cls']}/{r['file']}")
    if len(diff_cls) > 15:
        print(f"\n    ... and {len(diff_cls)-15} more")
    return groups


def check_near(recs, threshold, sample):
    sec(f"3. NEAR DUPLICATES ACROSS SPLITS (dHash <= {threshold})")
    usable = [r for r in recs if r.get("dhash") is not None]
    by_cls = defaultdict(list)
    for r in usable:
        by_cls[r["cls"]].append(r)

    print("Compared within each class only. A hit means the same lesion")
    print("appears on both sides of a split boundary.\n")
    grand = 0
    for cls in sorted(by_cls):
        items = by_cls[cls]
        if sample and len(items) > sample:
            step = len(items) // sample
            items = items[::step]
        hits = 0
        seen = set()
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if a["split"] == b["split"] or a["md5"] == b["md5"]:
                    continue
                if hamming(a["dhash"], b["dhash"]) <= threshold:
                    hits += 1
                    seen.add(a["path"])
                    seen.add(b["path"])
        grand += hits
        note = "  <-- heavy" if len(seen) > 0.2 * len(items) else ""
        print(f"  {cls:<28} pairs={hits:<6} images involved={len(seen)}{note}")
    print(f"\n  total cross-split near-duplicate pairs: {grand}")
    return grand


def check_numbering(recs, splits):
    sec("4. NUMBERING ADJACENCY ACROSS SPLITS")
    print("If filenames carry the original capture order, consecutive")
    print("numbers landing in different splits means the split was made at")
    print("image level. Bursts of one lesion then straddle the boundary.\n")

    by_cls = defaultdict(dict)
    for r in recs:
        m = RE_NUM.findall(os.path.splitext(r["file"])[0])
        if not m:
            continue
        n = int(m[-1])
        by_cls[r["cls"]][n] = r["split"]

    for cls in sorted(by_cls):
        nums = sorted(by_cls[cls])
        if len(nums) < 2:
            continue
        adj = cross = 0
        for a, b in zip(nums, nums[1:]):
            if b - a == 1:
                adj += 1
                if by_cls[cls][a] != by_cls[cls][b]:
                    cross += 1
        if adj == 0:
            print(f"  {cls:<28} no consecutive pairs")
            continue
        pct = 100 * cross / adj
        note = "  <-- image-level split" if pct > 20 else ""
        print(f"  {cls:<28} consecutive pairs={adj:<5} "
              f"crossing a split={cross:<5} ({pct:.0f}%){note}")


def check_preproc(recs):
    sec("5. PREPROCESSING CONSISTENCY")
    by_cls = defaultdict(lambda: {"wh": Counter(), "ext": Counter()})
    for r in recs:
        c = by_cls[r["cls"]]
        c["ext"][os.path.splitext(r["file"])[1].lower()] += 1
        if r.get("wh"):
            c["wh"][r["wh"]] += 1

    for cls in sorted(by_cls):
        c = by_cls[cls]
        exts = dict(c["ext"])
        n_res = len(c["wh"])
        flag = ""
        if len(exts) > 1:
            flag += "  mixed formats"
        if (224, 224) in c["wh"] and n_res > 1:
            flag += "  contains pre-resized 224x224"
        print(f"  {cls:<28} formats={exts}  distinct resolutions={n_res}{flag}")
        for wh, n in c["wh"].most_common(3):
            print(f"      {wh[0]}x{wh[1]}: {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--threshold", type=int, default=6)
    ap.add_argument("--sample", type=int, default=500,
                    help="max images per class for the O(n^2) near-dup pass")
    ap.add_argument("--fast", action="store_true",
                    help="skip decoding: MD5 and filename checks only")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        sys.exit(f"Not found: {root}")

    splits = find_splits(root)
    if not splits:
        sys.exit("No split folders found under root.")

    print(f"Root   : {os.path.abspath(root)}")
    print(f"Splits : {splits}")
    recs = scan(root, splits, want_hash=not args.fast)
    print(f"Images : {len(recs)}")

    counts = defaultdict(lambda: defaultdict(int))
    for r in recs:
        counts[r["cls"]][r["split"]] += 1
    print()
    for c in sorted(counts):
        row = "  ".join(f"{s}:{counts[c].get(s,0)}" for s in splits)
        print(f"  {c:<28} {row}   total={sum(counts[c].values())}")

    overlap = check_split_overlap(recs, splits)
    check_exact(recs)
    if not args.fast:
        check_near(recs, args.threshold, args.sample)
    check_numbering(recs, splits)
    if not args.fast:
        check_preproc(recs)

    sec("VERDICT")
    if overlap:
        print("Two splits share content. The published partition cannot be")
        print("used as-is; deduplicate before any evaluation on it.")
    else:
        print("No whole-split overlap. Read sections 2-4 for finer problems.")


if __name__ == "__main__":
    main()
