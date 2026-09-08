#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fill_table_s1.py
================
Computes Table S1 of Online Resource 1: how the leaked fraction responds to
the burst-recovery threshold.

This needs no training and no checkpoints. It re-derives capture sessions
from filenames at each gap value, using the same rules as session_split.py,
then measures how much of the dataset lands in sessions that cross a
boundary of the IMAGE-LEVEL partition.

Run it from the zenodo folder, or point --data at it:

    python fill_table_s1.py
    python fill_table_s1.py --data "C:\\path\\to\\zenodo"

Writes table_s1.md and prints the table, ready to paste into the supplement.

Requires: sessions.csv (for the class of each file) and splits/image_level/
with train/ val/ test/ subfolders.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.heic', '.heif'}

# Same patterns as session_split.py
VIDEO_RE = re.compile(r"^(?P<base>.+?)_frame\d+", re.I)
WHATSAPP_RE = re.compile(r"^WhatsApp Image (\d{4}-\d{2}-\d{2})", re.I)
NUM_RE = re.compile(r"^([A-Za-z_]*?)(\d{3,})")


def stem(fname):
    """Strip the extension, and a second one for names like IMG_1.HEIC.jpg."""
    p = Path(fname)
    s = p.stem
    if Path(s).suffix.lower() in IMG_EXT:
        s = Path(s).stem
    return s


def raw_key(fname):
    """Classify a filename: ('video'|'whatsapp'|'num'|'single', key, number)."""
    s = stem(fname)
    m = VIDEO_RE.match(s)
    if m:
        return "video", m.group("base"), None
    m = WHATSAPP_RE.match(s)
    if m:
        return "whatsapp", m.group(1), None
    m = NUM_RE.match(s)
    if m:
        return "num", m.group(1), int(m.group(2))
    return "single", s, None


def build_sessions(records, gap):
    """Assign a session id to every record, for a given burst gap."""
    runs = defaultdict(list)
    for r in records:
        kind, key, num = raw_key(r["file"])
        r["_kind"], r["_key"], r["_num"] = kind, key, num
        if kind == "video":
            r["session"] = f"video:{r['cls']}:{key}"
        elif kind == "whatsapp":
            r["session"] = f"whatsapp:{r['cls']}:{key}"
        elif kind == "single":
            r["session"] = f"single:{r['cls']}:{key}"
        else:
            # bursts are grouped within class and filename prefix
            runs[(r["cls"], key)].append(r)

    for (cls, prefix), group in runs.items():
        group.sort(key=lambda r: r["_num"])
        run = 0
        prev = None
        for r in group:
            if prev is not None and r["_num"] - prev > gap:
                run += 1
            r["session"] = f"burst:{cls}:{prefix}:{run}"
            prev = r["_num"]
    return records


def load_records(base):
    f = base / "sessions.csv"
    if not f.exists():
        sys.exit(f"sessions.csv not found in {base}")
    rows = list(csv.DictReader(open(f, newline='', encoding='utf-8-sig')))
    recs = [{"cls": r["cls"], "file": r["file"]} for r in rows]
    print(f"sessions.csv: {len(recs)} records")
    return recs


def load_image_level_split(base):
    """Return {(cls, file): split} for the image-level partition."""
    root = None
    for cand in (base / "splits" / "image_level", base / "splits"):
        if all((cand / s).is_dir() for s in ("train", "val", "test")):
            root = cand
            break
    if root is None:
        sys.exit("Could not find splits/image_level/{train,val,test}")
    print(f"image-level partition: {root}")
    out = {}
    for split in ("train", "val", "test"):
        for p in (root / split).rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                out[(p.parent.name, p.name)] = split
    print(f"  {len(out)} images assigned")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--gaps", default="3,4,5,6,8,10")
    args = ap.parse_args()
    base = Path(args.data)

    recs = load_records(base)
    split_of = load_image_level_split(base)

    missing = [k for k in ((r["cls"], r["file"]) for r in recs) if k not in split_of]
    if missing:
        print(f"  warning: {len(missing)} records not found in the split tree")

    lines = []
    header = ("| Burst gap threshold | Sessions recovered | "
              "Images in boundary-crossing sessions | Share of dataset (%) |")
    sep = "|---|---|---|---|"
    lines += [header, sep]
    print("\n" + header)
    print(sep)

    for gap in [int(g) for g in args.gaps.split(",")]:
        recs = build_sessions([dict(r) for r in recs], gap)
        members = defaultdict(set)      # session -> set of splits
        counts = defaultdict(int)       # session -> n images
        for r in recs:
            key = (r["cls"], r["file"])
            counts[r["session"]] += 1
            if key in split_of:
                members[r["session"]].add(split_of[key])
        n_sessions = len(counts)
        crossing = [s for s, sp in members.items() if len(sp) > 1]
        n_imgs = sum(counts[s] for s in crossing)
        total = sum(counts.values())
        share = n_imgs / total * 100
        label = f"{gap} (used throughout)" if gap == 3 else str(gap)
        row = f"| {label} | {n_sessions} | {n_imgs} | {share:.1f} |"
        lines.append(row)
        print(row)

    out = base / "table_s1.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWritten to {out}")
    print("\nCheck: the gap = 3 row should read 73 sessions, 446 images, 79.6%.")
    print("If it does, the other rows are computed the same way and can be")
    print("pasted straight into Table S1. If it does not, tell me what it says")
    print("before using any of it.")


if __name__ == "__main__":
    main()
