#!/usr/bin/env python3
"""
Capture-session grouping and group-aware re-split
=================================================
Perceptual hashing only catches frames that still LOOK alike.  Burst shots
and video frames from one session can drift past any sensible threshold
while still showing the same lesion on the same branch.

This script groups images by capture session using filename structure,
reports which sessions currently straddle train/val/test, and can write a
corrected split in which every session lands wholly in one split.

Session keys derived from:
  IMG_9954_frame00648.jpg     -> video   IMG_9954          (all frames)
  IMG_2155.HEIC.jpg           -> burst   IMG_21xx run      (consecutive nums)
  WhatsApp Image 2026-06-10.. -> whatsapp 2026-06-10
  <uuid>.jpg                  -> singleton

Only needs Pillow for the optional visual check:  pip install pillow

Usage:
    python session_split.py --report
    python session_split.py --report --gap 5
    python session_split.py --rebuild clean_split --seed 42
"""

import argparse
import os
import random
import re
import shutil
import sys
from collections import defaultdict

SPLITS = ["train", "val", "test"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic"}

DEFAULT_ROOT = r"C:\Users\Lim Ding Shan\Desktop\Durian project and paper\Classication_model_split"
PAPER_CLASSES = {"algal", "leaf_rot", "phomopsis", "pink_disease", "root_disease"}

RE_FRAME = re.compile(r"^(?P<base>.+?)_frame\d+", re.I)
RE_SEQ = re.compile(r"^(?P<prefix>[A-Za-z_]*?)(?P<num>\d{3,6})\b")
RE_WHATSAPP = re.compile(r"WhatsApp Image (\d{4}-\d{2}-\d{2})", re.I)


def stem(fname):
    """Strip the extension, including doubled ones like .HEIC.jpg"""
    s = fname
    while True:
        s2, ext = os.path.splitext(s)
        if ext.lower() in IMG_EXT:
            s = s2
        else:
            break
    return s


def raw_key(fname):
    """First-pass session key. Returns (kind, key, numeric_or_None)."""
    s = stem(fname)

    m = RE_FRAME.match(s)
    if m:
        return "video", m.group("base"), None

    m = RE_WHATSAPP.search(s)
    if m:
        return "whatsapp", m.group(1), None

    m = RE_SEQ.match(s)
    if m:
        return "seq", m.group("prefix") or "IMG", int(m.group("num"))

    return "single", s, None


def build_sessions(records, gap):
    """Assign every record a session id, merging consecutive sequence numbers."""
    # bucket the sequential ones by (class, prefix) so runs don't merge across
    # classes -- a burst belongs to one lesion, therefore one class
    seq_bucket = defaultdict(list)
    for r in records:
        kind, key, num = raw_key(r["file"])
        r["_kind"], r["_key"], r["_num"] = kind, key, num
        if kind == "seq":
            seq_bucket[(r["cls"], key)].append(r)
        elif kind == "video":
            r["session"] = f"video:{r['cls']}:{key}"
        elif kind == "whatsapp":
            r["session"] = f"whatsapp:{r['cls']}:{key}"
        else:
            r["session"] = f"single:{r['cls']}:{key}"

    for (cls, prefix), group in seq_bucket.items():
        group.sort(key=lambda r: r["_num"])
        run, prev = 0, None
        for r in group:
            if prev is not None and r["_num"] - prev > gap:
                run += 1
            r["session"] = f"burst:{cls}:{prefix}:{run}"
            prev = r["_num"]
    return records


def scan(root, paper_only):
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
            for fn in sorted(os.listdir(cdir)):
                fp = os.path.join(cdir, fn)
                if not os.path.isfile(fp):
                    continue
                s = fn
                while True:
                    s2, ext = os.path.splitext(s)
                    if ext.lower() in IMG_EXT:
                        s = s2
                    else:
                        break
                if s == fn:          # no image extension at all
                    continue
                recs.append({"split": split, "cls": cls, "file": fn, "path": fp})
    return recs


BAR = "=" * 72


def sec(t):
    print(f"\n{BAR}\n{t}\n{BAR}")


def report(records):
    by_session = defaultdict(list)
    for r in records:
        by_session[r["session"]].append(r)

    straddling = {s: g for s, g in by_session.items()
                  if len({r["split"] for r in g}) > 1}

    sec("SESSION SUMMARY")
    kinds = defaultdict(int)
    for s in by_session:
        kinds[s.split(":", 1)[0]] += 1
    print(f"images                  : {len(records)}")
    print(f"capture sessions        : {len(by_session)}")
    for k, n in sorted(kinds.items()):
        print(f"    {k:<10}: {n}")
    print(f"\nsessions straddling splits : {len(straddling)}   <-- LEAKAGE")

    affected = sum(len(g) for g in straddling.values())
    print(f"images involved            : {affected} "
          f"({100*affected/max(len(records),1):.1f}% of the dataset)")

    if straddling:
        sec("SESSIONS THAT STRADDLE SPLITS")
        for s in sorted(straddling, key=lambda x: -len(by_session[x])):
            g = by_session[s]
            dist = defaultdict(int)
            for r in g:
                dist[r["split"]] += 1
            layout = "  ".join(f"{k}:{dist.get(k,0)}" for k in SPLITS)
            print(f"\n  {s}")
            print(f"    {len(g)} images   {layout}")
            for r in sorted(g, key=lambda r: (r["split"], r["file"]))[:12]:
                print(f"      {r['split']:<6} {r['file']}")
            if len(g) > 12:
                print(f"      ... and {len(g)-12} more")

    sec("PER-CLASS EXPOSURE")
    hdr = f"{'class':<20}{'images':>8}{'sessions':>10}{'leaky imgs':>12}"
    print(hdr)
    print("-" * len(hdr))
    per_cls = defaultdict(lambda: {"n": 0, "sess": set(), "leak": 0})
    for r in records:
        d = per_cls[r["cls"]]
        d["n"] += 1
        d["sess"].add(r["session"])
        if r["session"] in straddling:
            d["leak"] += 1
    for c in sorted(per_cls):
        d = per_cls[c]
        print(f"{c:<20}{d['n']:>8}{len(d['sess']):>10}{d['leak']:>12}")

    return by_session, straddling


def rebuild(records, by_session, out_root, seed, ratios):
    """Greedy group-stratified split: whole sessions, per class, hit ratios."""
    rng = random.Random(seed)
    per_class = defaultdict(list)
    for s, g in by_session.items():
        per_class[g[0]["cls"]].append((s, g))

    assignment = {}
    sec("REBUILT SPLIT")
    hdr = f"{'class':<20}{'train':>8}{'val':>6}{'test':>6}{'total':>8}{'sessions':>10}"
    print(hdr)
    print("-" * len(hdr))

    for cls in sorted(per_class):
        sessions = per_class[cls]
        rng.shuffle(sessions)
        sessions.sort(key=lambda x: -len(x[1]))     # big sessions placed first
        total = sum(len(g) for _, g in sessions)
        target = {sp: total * ratios[sp] for sp in SPLITS}
        cur = {sp: 0 for sp in SPLITS}

        for s, g in sessions:
            # place where the shortfall relative to target is largest
            sp = max(SPLITS, key=lambda k: target[k] - cur[k])
            # never leave val/test empty if any session remains
            assignment[s] = sp
            cur[sp] += len(g)

        # guarantee at least one session in val and test where possible
        for need in ("test", "val"):
            if cur[need] == 0 and len(sessions) >= 3:
                donor = max((s for s, _ in sessions
                             if assignment[s] == "train"),
                            key=lambda s: -len(by_session[s]), default=None)
                if donor:
                    cur["train"] -= len(by_session[donor])
                    cur[need] += len(by_session[donor])
                    assignment[donor] = need

        print(f"{cls:<20}{cur['train']:>8}{cur['val']:>6}{cur['test']:>6}"
              f"{total:>8}{len(sessions):>10}")

    if not out_root:
        return

    for r in records:
        sp = assignment[r["session"]]
        dst = os.path.join(out_root, sp, r["cls"])
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(r["path"], os.path.join(dst, r["file"]))
    print(f"\nWritten to: {os.path.abspath(out_root)}")
    print("Re-run label_conflicts.py against it to confirm it is clean,")
    print("then retrain every model from scratch on the new split.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--paper-only", action="store_true", default=True)
    ap.add_argument("--all-classes", dest="paper_only", action="store_false")
    ap.add_argument("--gap", type=int, default=3,
                    help="max gap in image numbers still counted as one burst")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rebuild", default=None, metavar="OUTDIR")
    ap.add_argument("--manifest", default=None, metavar="CSV",
                    help="write file -> session mapping for GroupKFold")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"Folder not found: {args.root}")

    recs = scan(args.root, args.paper_only)
    if not recs:
        sys.exit("No images found.")
    recs = build_sessions(recs, args.gap)

    print(f"Scanning : {args.root}")
    print(f"Scope    : {'5 paper classes' if args.paper_only else 'all classes'}")
    print(f"Burst gap: {args.gap}")

    by_session, straddling = report(recs)

    if args.manifest:
        import csv as _csv
        with open(args.manifest, "w", newline="", encoding="utf-8") as fh:
            wr = _csv.writer(fh)
            wr.writerow(["path", "file", "cls", "split", "session"])
            for r in sorted(recs, key=lambda r: (r["cls"], r["session"], r["file"])):
                wr.writerow([r["path"], r["file"], r["cls"], r["split"], r["session"]])
        print(f"\nManifest written to: {os.path.abspath(args.manifest)}")
        print("Use the 'session' column as the groups argument of")
        print("sklearn.model_selection.StratifiedGroupKFold.")

    if args.rebuild:
        rebuild(recs, by_session, args.rebuild, args.seed,
                {"train": 0.8, "val": 0.1, "test": 0.1})

    sec("VERDICT")
    if not straddling:
        print("No session straddles a split. The held-out sets are independent.")
    else:
        print(f"{len(straddling)} sessions straddle splits. Rebuild with:")
        print(f"    python session_split.py --rebuild clean_split")
        print("Then retrain everything. Expect the Malaysian scores to fall;")
        print("the post-rebuild numbers are the defensible ones.")


if __name__ == "__main__":
    main()
