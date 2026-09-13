#!/usr/bin/env python3
"""
phash_sessions.py — recover capture sessions when filenames carry no structure
==============================================================================
session_split.py reads sessions off filename structure: IMG_9954_frame00648,
consecutive camera sequence numbers, messaging batch names. Most published
datasets have none of that, because they were renumbered after collection. The
Vietnamese datasets used here are renumbered, so their sessions have to be
recovered from the images themselves.

The method is single-linkage clustering on a perceptual hash. Two images join
the same cluster when their dHash Hamming distance is at most `--threshold`,
and clustering is transitive, so a burst photographed while walking around one
lesion chains together even though its first and last frames are far apart.
Clustering runs within a class, since two images of different diseases are not
the same capture episode whatever they look like.

This is a proxy, not ground truth, and the paper must say so. Three things
follow from that and are built in here:

  * The threshold is swept, not assumed. `--sweep` reports cluster counts and
    the leaked fraction across a range so the reader can see how sensitive the
    conclusion is. If the answer moves a lot, report the range.
  * Singletons are reported separately. A dataset where every image is its own
    cluster has either no redundancy or a threshold that is too tight, and the
    two look identical in the summary line.
  * The largest clusters are printed with their filenames, so you can open a
    few and check the method is finding what you think it is.

    pip install pillow numpy
    python phash_sessions.py --root Durian_Leaf_Diseases --sweep
    python phash_sessions.py --root Durian_Leaf_Diseases --threshold 6 \\
        --manifest vn_a_sessions.csv

Output manifest has the same columns session_split.py produces, so the
downstream tooling takes it unchanged.
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow missing.  pip install pillow")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def dhash(img, size=8):
    small = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = np.asarray(small, dtype=np.int16)
    bits = px[:, :-1] > px[:, 1:]
    return np.packbits(bits.flatten())


def hamming_matrix(hashes):
    """Pairwise Hamming distance over packed uint8 hashes."""
    h = np.stack(hashes)                       # (n, 8) uint8
    x = np.bitwise_xor(h[:, None, :], h[None, :, :])
    return np.unpackbits(x, axis=-1).sum(-1)


class Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def find_splits(root):
    out = []
    for d in sorted(Path(root).iterdir()):
        if d.is_dir() and any(p.is_dir() for p in d.iterdir()):
            out.append(d.name)
    return out


def scan(root, splits):
    recs = []
    for sp in splits:
        for cls_dir in sorted((Path(root) / sp).iterdir()):
            if not cls_dir.is_dir():
                continue
            for f in sorted(cls_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in IMG_EXT:
                    recs.append({"split": sp, "cls": cls_dir.name.strip(),
                                 "file": f.name, "path": str(f)})
    return recs


def hash_all(recs):
    keep = []
    for r in recs:
        try:
            with Image.open(r["path"]) as im:
                im.load()
                r["h"] = dhash(im)
            keep.append(r)
        except Exception as exc:
            print(f"  unreadable, skipped: {r['path']}  {exc!r}")
    return keep


def cluster(recs, threshold):
    """Single-linkage within class. Returns {index: session_id}."""
    by_cls = defaultdict(list)
    for i, r in enumerate(recs):
        by_cls[r["cls"]].append(i)
    session = {}
    for cls, idxs in by_cls.items():
        if len(idxs) == 1:
            session[idxs[0]] = f"{cls}:s0"
            continue
        D = hamming_matrix([recs[i]["h"] for i in idxs])
        uf = Union(len(idxs))
        ii, jj = np.where(np.triu(D <= threshold, k=1))
        for a, b in zip(ii, jj):
            uf.union(int(a), int(b))
        roots = {}
        for k, i in enumerate(idxs):
            r = uf.find(k)
            roots.setdefault(r, len(roots))
            session[i] = f"{cls}:s{roots[r]}"
    return session


def report(recs, session, threshold, show=0):
    groups = defaultdict(list)
    for i, s in session.items():
        groups[s].append(i)
    sizes = [len(v) for v in groups.values()]
    singles = sum(1 for s in sizes if s == 1)

    straddle, leaked = 0, 0
    for s, idxs in groups.items():
        splits = {recs[i]["split"] for i in idxs}
        if len(splits) > 1:
            straddle += 1
            leaked += len(idxs)

    print(f"\n  threshold {threshold:>2}  clusters {len(groups):>5}  "
          f"singletons {singles:>5} ({100*singles/len(groups):4.1f}%)  "
          f"largest {max(sizes):>4}  "
          f"straddling {straddle:>4}  leaked {leaked:>5} "
          f"({100*leaked/len(recs):4.1f}%)")

    if show:
        big = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:show]
        for s, idxs in big:
            spl = Counter(recs[i]["split"] for i in idxs)
            print(f"\n    {s}: {len(idxs)} images  {dict(spl)}")
            for i in idxs[:6]:
                print(f"      {recs[i]['split']}/{recs[i]['file']}")
            if len(idxs) > 6:
                print(f"      ... and {len(idxs)-6} more")
    return dict(clusters=len(groups), singletons=singles, largest=max(sizes),
                straddling=straddle, leaked=leaked,
                leaked_pct=100 * leaked / len(recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--threshold", type=int, default=6)
    ap.add_argument("--sweep", action="store_true",
                    help="report 2,4,6,8,10,12 instead of clustering once")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--show", type=int, default=5,
                    help="print this many of the largest clusters, to eyeball")
    args = ap.parse_args()

    if not Path(args.root).is_dir():
        sys.exit(f"Not found: {args.root}")
    splits = find_splits(args.root)
    if not splits:
        sys.exit("No split folders under root.")
    print(f"root   : {Path(args.root).resolve()}")
    print(f"splits : {splits}")

    recs = scan(args.root, splits)
    print(f"images : {len(recs)}")
    print("hashing...")
    recs = hash_all(recs)
    print(f"hashed : {len(recs)}")
    per_cls = Counter(r["cls"] for r in recs)
    print(f"classes: {dict(per_cls)}")

    if args.sweep:
        print("\nthreshold sweep — read this before fixing a value")
        rows = []
        for t in (2, 4, 6, 8, 10, 12):
            rows.append((t, report(recs, cluster(recs, t), t)))
        print("\n  A threshold is too tight if almost every cluster is a")
        print("  singleton, and too loose if one cluster swallows a class.")
        print("  Pick the widest range over which the leaked fraction is")
        print("  stable, and report the range rather than a point estimate.")
        lp = [r[1]["leaked_pct"] for r in rows]
        print(f"\n  leaked fraction across the sweep: "
              f"{min(lp):.1f}% to {max(lp):.1f}%")
        if max(lp) - min(lp) > 15:
            print("  That is a wide spread. The conclusion depends on the")
            print("  threshold, and the paper has to say so.")
        return

    session = cluster(recs, args.threshold)
    stats = report(recs, session, args.threshold, show=args.show)

    if stats["singletons"] == stats["clusters"]:
        print("\n  Every cluster is a singleton. Either this dataset has no")
        print("  near-duplicate redundancy, or the threshold is too tight to")
        print("  find it. Run --sweep before concluding the former.")
    if stats["largest"] > 0.4 * len(recs) / max(len(per_cls), 1):
        print("\n  One cluster holds a large share of its class. Single-linkage")
        print("  chains, so a gradual sequence can merge unrelated images.")
        print("  Inspect the printed filenames before using this manifest.")

    if args.manifest:
        with open(args.manifest, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["cls", "file", "split", "session", "path"])
            for i, r in enumerate(recs):
                w.writerow([r["cls"], r["file"], r["split"], session[i], r["path"]])
        print(f"\n  manifest written to {args.manifest}")
        print("  Columns match session_split.py, so session_utils.load_sessions")
        print("  and StratifiedGroupKFold take it unchanged.")
        print("\n  State in the paper that these sessions are inferred by")
        print("  perceptual hashing rather than read from capture metadata, and")
        print("  give the threshold sweep. They are a proxy and a reader should")
        print("  be able to see how much the result depends on it.")


if __name__ == "__main__":
    main()
