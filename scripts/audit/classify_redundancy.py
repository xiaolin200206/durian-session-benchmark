#!/usr/bin/env python3
"""
classify_redundancy.py — is the redundancy a capture burst or a copied file?
============================================================================
phash_sessions.py finds clusters of visually near-identical images. It cannot
tell you why they are near-identical, and the reason decides whether they mean
anything.

  BURST     several photographs of one lesion, seconds apart. Different bytes,
            different pixels, same specimen. This is redundancy produced by the
            act of collecting field data, and it is the thing a session-level
            partition exists to handle.

  COPY      the same file present more than once, or re-encoded. Different
            filename, identical or near-identical bytes. This is a mistake made
            while assembling the dataset. It also inflates scores, but it is a
            different phenomenon with a different fix, and lumping the two
            together would compare quantities that are not the same quantity.

The distinction matters here because the Vietnamese ten-class dataset shows
filename patterns (136 / 1363 / 363 / 3632, and identical names across splits)
that suggest copying rather than bursts.

    python classify_redundancy.py --manifest vn_b_sessions.csv --label VN-B
    python classify_redundancy.py --manifest vn_a_sessions.csv --label VN-A
"""

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas missing.  pip install pandas")

try:
    import numpy as np
    from PIL import Image
except ImportError:
    sys.exit("Pillow/numpy missing.  pip install pillow numpy")


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def pixel_hash(p):
    """Hash of the decoded pixels: catches re-encoding, which MD5 misses."""
    try:
        with Image.open(p) as im:
            im.load()
            a = np.asarray(im.convert("RGB").resize((64, 64), Image.LANCZOS))
        return hashlib.md5(a.tobytes()).hexdigest()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--label", default="dataset")
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    f = Path(args.manifest)
    if not f.is_file():
        sys.exit(f"Not found: {f}")
    d = pd.read_csv(f)
    need = {"cls", "file", "split", "session", "path"}
    if not need.issubset(d.columns):
        sys.exit(f"{f} is missing columns: {need - set(d.columns)}")

    n_total = len(d)
    sizes = d.groupby("session").size()
    multi = d[d.session.isin(sizes[sizes > 1].index)].copy()
    print(f"{args.label}: {n_total} images, {len(sizes)} clusters, "
          f"{len(multi)} images in clusters of two or more "
          f"({100*len(multi)/n_total:.1f}%)\n")
    if multi.empty:
        print("No multi-image clusters. Nothing to classify.")
        return

    print("hashing bytes and pixels...")
    multi["md5"] = [md5(p) for p in multi["path"]]
    multi["px"] = [pixel_hash(p) for p in multi["path"]]

    byte_copies = pixel_copies = burst = 0
    per_cluster = []
    for s, g in multi.groupby("session"):
        n = len(g)
        b = n - g["md5"].nunique()                  # copies beyond the first
        px = n - g["px"].nunique()                  # includes byte copies
        px_only = max(px - b, 0)                    # re-encodes, not byte-equal
        rest = n - 1 - b - px_only                  # near, but genuinely distinct
        byte_copies += b
        pixel_copies += px_only
        burst += max(rest, 0)
        per_cluster.append(dict(session=s, n=n, byte=b, reenc=px_only,
                                distinct=max(rest, 0),
                                splits=g["split"].nunique(),
                                cls=g["cls"].iloc[0]))

    redundant = byte_copies + pixel_copies + burst
    print(f"\nredundant images beyond one representative per cluster: {redundant}")
    if redundant:
        print(f"  byte-identical copies          {byte_copies:>6}  "
              f"({100*byte_copies/redundant:5.1f}%)")
        print(f"  re-encoded copies              {pixel_copies:>6}  "
              f"({100*pixel_copies/redundant:5.1f}%)")
        print(f"  visually near but distinct     {burst:>6}  "
              f"({100*burst/redundant:5.1f}%)   <- capture-burst redundancy")

    frac_copy = (byte_copies + pixel_copies) / redundant if redundant else 0
    print(f"\ncopies as a share of all redundancy: {100*frac_copy:.1f}%")
    if frac_copy > 0.7:
        print("  Most of the redundancy here is duplicated files, not capture")
        print("  bursts. Do not use this dataset's redundancy figure as a")
        print("  measure of collection-process redundancy; report the two")
        print("  separately or exclude it from that comparison.")
    elif frac_copy < 0.3:
        print("  Most of the redundancy is genuinely distinct images of the")
        print("  same subject. This is capture-burst redundancy and is")
        print("  comparable with a session-level analysis.")
    else:
        print("  Mixed. Report both components; a single redundancy number")
        print("  would conflate two different problems.")

    pc = pd.DataFrame(per_cluster)
    cross = pc[pc.splits > 1]
    print(f"\nclusters straddling a split: {len(cross)} of {len(pc)}")
    if len(cross):
        cb = cross.byte.sum() + cross.reenc.sum()
        cd = cross.distinct.sum()
        print(f"  of the images they contribute: {cb} copies, {cd} distinct")
        print("  Copies crossing a split are a dataset-assembly fault;")
        print("  distinct images crossing a split are a partitioning fault.")

    print(f"\nper class, redundancy beyond one per cluster:")
    byc = pc.groupby("cls")[["byte", "reenc", "distinct"]].sum()
    byc["total"] = byc.sum(axis=1)
    byc = byc.sort_values("total", ascending=False)
    print(byc.to_string())

    if args.show:
        print(f"\nlargest clusters, with composition:")
        for r in pc.sort_values("n", ascending=False).head(args.show).itertuples():
            print(f"  {r.session:<28} n={r.n:<3} byte={r.byte:<3} "
                  f"re-enc={r.reenc:<3} distinct={r.distinct:<3} "
                  f"splits={r.splits}")

    out = f.with_name(f.stem + "_redundancy.csv")
    pc.to_csv(out, index=False)
    print(f"\nper-cluster breakdown written to {out}")


if __name__ == "__main__":
    main()
