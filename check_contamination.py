#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_contamination2.py  -- corrected version
=============================================
The first version keyed images by filename alone. That was wrong: 34
filenames occur in more than one class folder (e.g. IMG_9890.jpg exists
under both Algal and Phomopsis), so distinct images were merged, images
were lost from the counts, and spurious cross-split overlaps appeared.

This version keys every image by (class, filename), which is unique.

It reports:
  A. train<->test / train<->val / val<->test session overlap, per partition
  B. the "any boundary" figure for comparison with the manuscript's 79.6%
  C. cross-class camera-sequence adjacency -- images with consecutive
     camera numbers that were filed under different disease classes.
     Sessions are built within class, so such images fall into different
     sessions and can land on opposite sides of a split even under
     session-grouped partitioning. This is residual dependence the
     session rule does not remove, and it should be quantified.

Writes contamination_report2.txt into the folder below.
"""

import csv
import re
import collections
from pathlib import Path
from datetime import datetime

import argparse as _ap
_p=_ap.ArgumentParser(add_help=True)
_p.add_argument("--data", default=".",
                help="folder holding sessions.csv, splits/ and the image dirs")
BASE = Path(_p.parse_known_args()[0].data)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp"}
_lines = []


def say(m=""):
    print(m)
    _lines.append(str(m))


# ------------------------------------------------------------------ inputs
def load_sessions():
    f = BASE / "sessions.csv"
    if not f.exists():
        say("!! sessions.csv not found")
        return None
    rows = list(csv.DictReader(open(f, newline="", encoding="utf-8-sig")))
    say(f"sessions.csv: {len(rows)} rows, columns {list(rows[0].keys())}")

    key_to_session = {}
    collisions = collections.Counter(r["file"] for r in rows)
    n_dup = sum(1 for v in collisions.values() if v > 1)
    say(f"  distinct filenames: {len(collisions)}  "
        f"(filenames used in >1 class: {n_dup})")
    say(f"  -> keying by (class, filename), which is unique")

    for r in rows:
        key_to_session[(r["cls"], r["file"])] = r["session"]
    say(f"  built {len(key_to_session)} (class, file) -> session entries")
    return rows, key_to_session


def splits_from_folder(root):
    """root/{train,val,test}/{class}/img.jpg  ->  split -> set of (class, file)"""
    out = {}
    for split_name in ("train", "val", "test"):
        d = root / split_name
        if not d.is_dir():
            continue
        keys = set()
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                cls = p.parent.name          # class is the immediate parent dir
                keys.add((cls, p.name))
        out[split_name] = keys
    return out


def find_split_roots():
    base_splits = BASE / "splits"
    if not base_splits.is_dir():
        return []
    cands = [base_splits] + [p for p in base_splits.rglob("*") if p.is_dir()]
    roots = [c for c in cands
             if all((c / s).is_dir() for s in ("train", "val", "test"))]
    return roots


# ----------------------------------------------------------------- analysis
def analyse(split_map, key_to_session, label):
    say("\n" + "=" * 72)
    say(f"PARTITION: {label}")
    say("=" * 72)

    sizes = {k: len(v) for k, v in split_map.items()}
    total = sum(sizes.values())
    say(f"Images per split: {sizes}   total {total}")
    if total != 560:
        say(f"  NOTE: total is {total}, not 560 -- see unmatched count below")

    sess = {}
    unmatched_total = 0
    for k, keys in split_map.items():
        c = collections.Counter()
        unmatched = 0
        for key in keys:
            sid = key_to_session.get(key)
            if sid is None:
                unmatched += 1
            else:
                c[sid] += 1
        sess[k] = c
        unmatched_total += unmatched
        if unmatched:
            say(f"  '{k}': {unmatched} images not found in sessions.csv")
    say(f"Sessions per split: { {k: len(v) for k, v in sess.items()} }")
    if unmatched_total == 0:
        say("  all images matched to a session")

    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        if a not in sess or b not in sess:
            continue
        shared = set(sess[a]) & set(sess[b])
        ia = sum(sess[a][s] for s in shared)
        ib = sum(sess[b][s] for s in shared)
        tot_b = sum(sess[b].values())
        tot_a = sum(sess[a].values())
        say(f"\n  {a.upper()} <-> {b.upper()}")
        say(f"    sessions in both:               {len(shared)}")
        say(f"    {b} images in a shared session:  {ib} / {tot_b}"
            f"  = {ib / max(1, tot_b) * 100:.1f}%")
        say(f"    {a} images in a shared session:  {ia} / {tot_a}"
            f"  = {ia / max(1, tot_a) * 100:.1f}%")
        if (a, b) == ("train", "test"):
            say(f"    >>> the {ib / max(1, tot_b) * 100:.1f}% figure is the one to quote:")
            say(f"        test images whose session also appears in training")

    all_sess = collections.defaultdict(set)
    for k, c in sess.items():
        for sid in c:
            all_sess[sid].add(k)
    strad = {s for s, ks in all_sess.items() if len(ks) > 1}
    imgs = sum(sum(sess[k][s] for k in sess if s in sess[k]) for s in strad)
    say(f"\n  ANY boundary (compare with the manuscript's 79.6%)")
    say(f"    sessions touching >1 split:     {len(strad)} / {len(all_sess)}")
    say(f"    images in those sessions:       {imgs} / {total}"
        f"  = {imgs / max(1, total) * 100:.1f}%")


# ------------------------------------------- C. cross-class sequence check
def cross_class_sequences(rows, split_map, label):
    say("\n" + "=" * 72)
    say(f"CROSS-CLASS CAMERA SEQUENCES -- {label}")
    say("=" * 72)
    say("Sessions are built within class, so consecutive camera frames filed")
    say("under two different disease classes fall into two different sessions")
    say("and are not kept together by the session rule.")

    # parse a numeric camera sequence out of the filename
    seq = {}   # (prefix, number) -> list of (cls, file)
    pat = re.compile(r"^([A-Za-z_]*?)(\d{3,})")
    for r in rows:
        m = pat.match(r["file"])
        if not m:
            continue
        seq.setdefault((m.group(1), int(m.group(2))), []).append((r["cls"], r["file"]))

    # same camera number, different class
    same_num_diff_class = {k: v for k, v in seq.items()
                           if len({c for c, _ in v}) > 1}
    say(f"\nIdentical camera numbers filed under >1 class: {len(same_num_diff_class)}")
    for k, v in list(same_num_diff_class.items())[:10]:
        say(f"    {k[0]}{k[1]}: {[c for c, _ in v]}")

    # adjacent camera numbers (n, n+1) in different classes
    bykey = {}
    for (pfx, num), v in seq.items():
        for cls, fn in v:
            bykey[(pfx, num, cls)] = fn
    adjacent = []
    for (pfx, num, cls), fn in bykey.items():
        for other_cls in {c for (p, n, c) in bykey if p == pfx and n == num + 1}:
            if other_cls != cls:
                adjacent.append(((pfx, num, cls, fn),
                                 (pfx, num + 1, other_cls, bykey[(pfx, num + 1, other_cls)])))
    say(f"Adjacent camera numbers in different classes: {len(adjacent)} pair(s)")
    for a, b in adjacent[:10]:
        say(f"    {a[3]} ({a[2]})  <->  {b[3]} ({b[2]})")

    # do any such pairs straddle train/test?
    if "train" in split_map and "test" in split_map:
        tr, te = split_map["train"], split_map["test"]
        straddle = []
        for a, b in adjacent:
            ka, kb = (a[2], a[3]), (b[2], b[3])
            if (ka in tr and kb in te) or (kb in tr and ka in te):
                straddle.append((a, b))
        say(f"\n  >>> of those, pairs split across TRAIN and TEST: {len(straddle)}")
        for a, b in straddle[:10]:
            say(f"      {a[3]} ({a[2]}) and {b[3]} ({b[2]})")
        if straddle:
            say("      These are residual dependence the session rule does not remove.")
        else:
            say("      None -- the session rule happens to keep these together here.")


# --------------------------------------------------------------------- main
def main():
    say(f"CONTAMINATION AUDIT v2 (corrected keying)   {datetime.now():%Y-%m-%d %H:%M:%S}")
    say(f"Folder: {BASE}\n")

    loaded = load_sessions()
    if not loaded:
        return
    rows, key_to_session = loaded

    roots = find_split_roots()
    if not roots:
        say("\nNo train/val/test trees found under splits/")
        return

    for r in roots:
        sm = splits_from_folder(r)
        if not sm:
            continue
        label = str(r.relative_to(BASE))
        analyse(sm, key_to_session, label)
        if "image_level" in label.lower() or "session_level" in label.lower():
            cross_class_sequences(rows, sm, label)

    say("\n" + "=" * 72)
    say("SUMMARY TO READ")
    say("=" * 72)
    say("1. image_level TRAIN<->TEST percentage = the true contamination the")
    say("   conventional protocol produces. This replaces 79.6% in the abstract.")
    say("2. session_level TRAIN<->TEST should now be 0.0%. If it is not, the")
    say("   session-grouped partition genuinely leaks and must be reported.")
    say("3. The cross-class sequence section quantifies dependence that the")
    say("   session rule cannot remove, because sessions are built within class.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        say(traceback.format_exc())
    finally:
        try:
            out = BASE / "contamination_report2.txt"
            out.write_text("\n".join(_lines), encoding="utf-8")
            print(f"\n\nSaved to: {out}")
        except Exception as e:
            print(f"(could not save: {e})")
