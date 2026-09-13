#!/usr/bin/env python3
"""
make_centrecrop.py — deterministic stand-in for the Vietnamese preprocessing
===========================================================================
Why not background removal: u2net is a *saliency* model. In hand-held field
photographs it segments the hand, not the leaf, and discards the lesion
entirely. Verified on a 20-image sample before abandoning that approach.

What the Vietnamese pipeline actually achieves, from the diagnostic point of
view, is three things: peripheral clutter is removed, the subject is centred,
and every image arrives at the network at one fixed size. All three can be
had with a centre crop, which has no model in the loop and therefore cannot
silently delete the thing being classified.

    centre crop to `frac` of the shorter edge  ->  square  ->  resize

Filenames and folder structure are preserved, so sessions.csv still applies
and no split logic changes.

    python make_centrecrop.py --src clean_split --dst clean_split_cc
    python make_centrecrop.py --src clean_split --dst _cc50 --frac 0.5 --limit 20

This is an approximation and the paper should say so. It removes clutter
without claiming to isolate the leaf.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow missing.  pip install pillow")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def centre_square(img, frac, size):
    w, h = img.size
    side = int(round(min(w, h) * frac))
    side = max(side, 32)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize(
        (size, size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--frac", type=float, default=0.7,
                    help="fraction of the shorter edge to keep (default 0.7)")
    ap.add_argument("--size", type=int, default=400,
                    help="output edge, matching the Vietnamese dataset")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not 0.1 <= args.frac <= 1.0:
        sys.exit("--frac must be between 0.1 and 1.0")

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        sys.exit(f"Not found: {src}")
    if dst.exists():
        sys.exit(f"{dst} exists. Delete it or pick another name.")

    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not files:
        sys.exit(f"No images under {src}")
    if args.limit:
        files = files[:args.limit]

    print(f"{len(files)} images | centre {args.frac:.0%} of shorter edge "
          f"-> {args.size}x{args.size}")

    failed = []
    small = []
    for i, p in enumerate(files, 1):
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(p) as im:
                im.load()
                if im.mode != "RGB":
                    im = im.convert("RGB")
                w, h = im.size
                if min(w, h) * args.frac < args.size:
                    small.append(f"{p.relative_to(src)} ({w}x{h})")
                centre_square(im, args.frac, args.size).save(
                    out, "JPEG", quality=args.quality, optimize=True)
        except Exception as exc:
            failed.append((str(p.relative_to(src)), repr(exc)))
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", end="\r", flush=True)
    print()

    n_out = sum(1 for p in dst.rglob("*") if p.is_file())
    print(f"written : {n_out} / {len(files)}")

    by_class = Counter(p.parent.name for p in dst.rglob("*") if p.is_file())
    print("per class:", dict(sorted(by_class.items())))

    if small:
        print(f"\n{len(small)} images were upsampled to reach {args.size}px "
              f"({100*len(small)/len(files):.1f}%):")
        for s in small[:10]:
            print(f"   {s}")
        if len(small) > 10:
            print(f"   ... and {len(small)-10} more")
        print("  Upsampling adds no detail. If this is a large fraction, use a "
              "smaller --size or a larger --frac.")

    if failed:
        print(f"\n{len(failed)} failed:")
        for r, e in failed[:10]:
            print(f"   {r}  {e}")

    if n_out != len(files):
        print("\n!! count mismatch — do not train on this until resolved")
    else:
        print("\nCounts match. Filenames and structure preserved.")
        print("\nEyeball a dozen before training: the lesion must survive the "
              "crop. If a class is routinely cut off, raise --frac for "
              "everything rather than per class — a per-class crop would make "
              "the crop itself a class cue.")


if __name__ == "__main__":
    main()
