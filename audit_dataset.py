#!/usr/bin/env python3
"""
Durian dataset audit
====================
Checks a train/val/test image folder tree for:
  1. per-class / per-split counts (vs the numbers reported in the paper)
  2. corrupt or unreadable files
  3. resolution + file-size distribution
  4. EXACT duplicates (MD5) within and across splits
  5. NEAR duplicates (perceptual dHash) across splits  <-- leakage check
  6. EXIF capture date / camera model / GPS presence

Only needs Pillow:   pip install pillow

Usage:
    python audit_dataset.py
    python audit_dataset.py --root "C:/path/to/Classication_model_split"
    python audit_dataset.py --threshold 8      # looser near-dup matching
"""

import argparse
import csv
import hashlib
import os
import sys
from collections import defaultdict, Counter

try:
    from PIL import Image, ExifTags
except ImportError:
    sys.exit("Pillow not installed.  Run:  pip install pillow")

# ---------------------------------------------------------------- config

DEFAULT_ROOT = r"C:\Users\Lim Ding Shan\Desktop\Durian project and paper\Classication_model_split"

SPLITS = ["train", "val", "test"]

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic"}

# Table 1 of the manuscript.  Keys are matched case-insensitively against
# the folder names actually found on disk; anything unmatched is reported.
EXPECTED = {
    "algal":        {"train": 129, "val": 16, "test": 17},
    "leaf_rot":     {"train": 122, "val": 15, "test": 16},
    "phomopsis":    {"train": 125, "val": 15, "test": 17},
    "pink_disease": {"train":   8, "val":  1, "test":  1},
    "root_disease": {"train":  62, "val":  7, "test":  9},
}

# ---------------------------------------------------------------- hashing


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dhash(img, size=8):
    """Difference hash - robust to resize/recompression, sensitive to content."""
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


# ---------------------------------------------------------------- exif

_EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}


def exif_bits(img):
    """Return (datetime, camera model, has_gps)."""
    try:
        ex = img.getexif()
    except Exception:
        return None, None, False
    if not ex:
        return None, None, False
    dt = ex.get(_EXIF_TAGS.get("DateTimeOriginal")) or ex.get(_EXIF_TAGS.get("DateTime"))
    model = ex.get(_EXIF_TAGS.get("Model"))
    has_gps = _EXIF_TAGS.get("GPSInfo") in ex
    clean = lambda v: str(v).strip() if v else None
    return clean(dt), clean(model), has_gps


# ---------------------------------------------------------------- scan


def scan(root):
    records, problems = [], []
    for split in SPLITS:
        sdir = os.path.join(root, split)
        if not os.path.isdir(sdir):
            problems.append(("MISSING_SPLIT", sdir, "split folder not found"))
            continue
        for cls in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, cls)
            if not os.path.isdir(cdir):
                continue
            for fname in sorted(os.listdir(cdir)):
                fpath = os.path.join(cdir, fname)
                ext = os.path.splitext(fname)[1].lower()
                if not os.path.isfile(fpath):
                    continue
                if ext not in IMG_EXT:
                    problems.append(("NON_IMAGE", fpath, f"unexpected extension {ext}"))
                    continue
                try:
                    with Image.open(fpath) as im:
                        im.load()
                        w, h = im.size
                        mode = im.mode
                        dh = dhash(im)
                        dt, model, gps = exif_bits(im)
                except Exception as exc:
                    problems.append(("UNREADABLE", fpath, repr(exc)))
                    continue
                records.append({
                    "split": split, "cls": cls, "file": fname, "path": fpath,
                    "w": w, "h": h, "mode": mode,
                    "bytes": os.path.getsize(fpath),
                    "md5": md5_of(fpath), "dhash": dh,
                    "exif_datetime": dt or "", "exif_model": model or "",
                    "exif_gps": "yes" if gps else "no",
                })
    return records, problems


# ---------------------------------------------------------------- report

BAR = "=" * 72


def section(title):
    print(f"\n{BAR}\n{title}\n{BAR}")


def report_counts(records):
    section("1. COUNTS PER CLASS AND SPLIT")
    table = defaultdict(lambda: defaultdict(int))
    for r in records:
        table[r["cls"]][r["split"]] += 1

    hdr = f"{'class':<20}{'train':>8}{'val':>7}{'test':>7}{'total':>8}   vs paper"
    print(hdr)
    print("-" * len(hdr))
    grand = 0
    for cls in sorted(table):
        row = table[cls]
        tot = sum(row.values())
        grand += tot
        exp = EXPECTED.get(cls.lower())
        if exp is None:
            verdict = "no Table 1 entry"
        else:
            diffs = [f"{s}{row.get(s,0)-exp[s]:+d}" for s in SPLITS if row.get(s, 0) != exp[s]]
            verdict = "match" if not diffs else "MISMATCH " + " ".join(diffs)
        print(f"{cls:<20}{row.get('train',0):>8}{row.get('val',0):>7}"
              f"{row.get('test',0):>7}{tot:>8}   {verdict}")
    print("-" * len(hdr))
    print(f"{'TOTAL':<20}{'':>8}{'':>7}{'':>7}{grand:>8}   paper says 560")

    missing = set(EXPECTED) - {c.lower() for c in table}
    if missing:
        print(f"\n!! classes in Table 1 but not on disk: {sorted(missing)}")


def report_integrity(records, problems):
    section("2. FILE INTEGRITY")
    if not problems:
        print("No unreadable or unexpected files.")
    for kind, path, msg in problems:
        print(f"[{kind}] {path}\n    {msg}")

    section("3. IMAGE PROPERTIES")
    sizes = Counter((r["w"], r["h"]) for r in records)
    modes = Counter(r["mode"] for r in records)
    mp = sorted(r["w"] * r["h"] / 1e6 for r in records)
    print(f"distinct resolutions : {len(sizes)}")
    for (w, h), n in sizes.most_common(8):
        print(f"    {w}x{h}  ->  {n} files")
    if len(sizes) > 8:
        print(f"    ... and {len(sizes)-8} more")
    print(f"colour modes         : {dict(modes)}")
    if mp:
        print(f"megapixels           : min {mp[0]:.2f} / median {mp[len(mp)//2]:.2f} / max {mp[-1]:.2f}")
        if mp[0] < 0.05:
            print("    !! some images are tiny - check they are not thumbnails")


def report_exif(records):
    section("4. EXIF")
    models = Counter(r["exif_model"] for r in records if r["exif_model"])
    dates = sorted(r["exif_datetime"][:7] for r in records if r["exif_datetime"])
    gps = sum(1 for r in records if r["exif_gps"] == "yes")
    n_dt = len(dates)

    print(f"images with capture date : {n_dt}/{len(records)}")
    if dates:
        print(f"date range               : {dates[0]} .. {dates[-1]}")
        print("  (paper claims Jul 2025 - Jun 2026; months present:)")
        for m, n in sorted(Counter(dates).items()):
            print(f"    {m}: {n}")
    print(f"distinct camera models   : {len(models)}")
    for m, n in models.most_common():
        print(f"    {m}: {n}")
    if not models:
        print("    (none - EXIF likely stripped by an earlier resize/export step)")
    print(f"\nimages carrying GPS      : {gps}")
    if gps:
        print("    !! STRIP GPS BEFORE PUBLIC RELEASE - orchard coordinates are")
        print("       covered by your confidentiality agreement with the growers.")


def report_duplicates(records, threshold):
    section("5. EXACT DUPLICATES (MD5)")
    by_md5 = defaultdict(list)
    for r in records:
        by_md5[r["md5"]].append(r)
    exact = [v for v in by_md5.values() if len(v) > 1]
    cross = [g for g in exact if len({r["split"] for r in g}) > 1]

    print(f"duplicate groups          : {len(exact)}")
    print(f"  ... spanning splits     : {len(cross)}   <-- these are leakage")
    for g in exact:
        tag = "LEAKAGE" if len({r["split"] for r in g}) > 1 else "within-split"
        print(f"\n  [{tag}]")
        for r in g:
            print(f"    {r['split']}/{r['cls']}/{r['file']}")

    section(f"6. NEAR DUPLICATES ACROSS SPLITS (dHash, hamming <= {threshold})")
    print("Near-duplicates are burst shots / small re-crops of the same lesion.")
    print("Any pair listed below means the test set is not independent.\n")

    train = [r for r in records if r["split"] == "train"]
    held = [r for r in records if r["split"] in ("val", "test")]
    hits = []
    for h in held:
        for t in train:
            d = hamming(h["dhash"], t["dhash"])
            if d <= threshold and h["md5"] != t["md5"]:
                hits.append((d, h, t))
    hits.sort(key=lambda x: x[0])

    if not hits:
        print("None found. Held-out splits look independent at this threshold.")
    else:
        print(f"!! {len(hits)} suspicious pairs\n")
        for d, h, t in hits:
            flag = "  <-- SAME CLASS" if h["cls"] == t["cls"] else ""
            print(f"  dist {d:>2}  {h['split']}/{h['cls']}/{h['file']}")
            print(f"           train/{t['cls']}/{t['file']}{flag}")
        print("\n  Open a few pairs by eye before acting - dHash can flag")
        print("  genuinely different leaves that share a layout. Re-split by")
        print("  capture session (group split), not by random image, for any")
        print("  that are real.")
    return hits


def write_manifest(records, out_path):
    cols = ["split", "cls", "file", "w", "h", "mode", "bytes",
            "md5", "exif_datetime", "exif_model", "exif_gps", "path"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(records)
    print(f"\nManifest written to: {out_path}")


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--threshold", type=int, default=5,
                    help="dHash hamming distance for near-duplicates (default 5)")
    ap.add_argument("--manifest", default="dataset_manifest.csv")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"Folder not found: {args.root}")

    print(f"Scanning: {args.root}")
    records, problems = scan(args.root)
    if not records:
        sys.exit("No readable images found. Check the folder layout: "
                 "root/<split>/<class>/*.jpg")
    print(f"Read {len(records)} images.")

    report_counts(records)
    report_integrity(records, problems)
    report_exif(records)
    hits = report_duplicates(records, args.threshold)
    write_manifest(records, args.manifest)

    section("VERDICT")
    if hits:
        print("Cross-split near-duplicates found. Verify them by eye, and if")
        print("real, re-split and re-run every experiment before submitting.")
    else:
        print("No cross-split leakage detected. You can state in the paper that")
        print("the splits were checked for exact and perceptual duplicates.")


if __name__ == "__main__":
    main()
