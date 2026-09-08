#!/usr/bin/env python3
"""
shrink.py — downscale a dataset folder before uploading to a remote GPU

Training resizes to 224x224, so full-resolution phone images waste ~40x the
bandwidth for pixels that are discarded. This makes a 512px-max-side copy,
preserving the folder structure and filenames exactly, so every downstream
script (sessions.csv, split logic, class folders) still works unchanged.

Keeps the originals untouched. Release the originals on Zenodo; train on
these.

    pip install pillow
    python shrink.py --src clean_split --dst clean_split_512
    python shrink.py --src Classication_model_split --dst Classication_model_split_512
    python shrink.py --src vietnam --dst vietnam_512
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow not installed.  Run:  pip install pillow")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic"}


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--max-side", type=int, default=512,
                    help="longest edge in pixels (default 512; training uses 224)")
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        sys.exit(f"Not found: {src}")
    if dst.exists():
        sys.exit(f"{dst} already exists. Delete it or pick another name.")

    files = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT]
    if not files:
        sys.exit(f"No images under {src}")

    print(f"{len(files)} images  ->  max side {args.max_side}px")
    before = after = 0
    skipped = []

    for i, p in enumerate(files, 1):
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        before += p.stat().st_size

        try:
            with Image.open(p) as im:
                im.load()
                if im.mode in ("RGBA", "P", "LA"):
                    im = im.convert("RGB")
                w, h = im.size
                scale = args.max_side / max(w, h)
                if scale < 1:
                    im = im.resize((max(1, round(w * scale)),
                                    max(1, round(h * scale))),
                                   Image.LANCZOS)
                # keep the original filename, including doubled extensions
                # like .HEIC.jpg, so sessions.csv keys still match
                im.save(out, "JPEG", quality=args.quality, optimize=True)
        except Exception as exc:
            skipped.append((str(rel), repr(exc)))
            shutil.copy2(p, out)

        after += out.stat().st_size
        if i % 100 == 0 or i == len(files):
            print(f"  {i}/{len(files)}", end="\r", flush=True)

    print()
    print(f"before : {human(before)}")
    print(f"after  : {human(after)}   ({before/max(after,1):.1f}x smaller)")
    if skipped:
        print(f"\n{len(skipped)} files copied unchanged (could not re-encode):")
        for rel, err in skipped[:10]:
            print(f"   {rel}  {err}")

    n_out = sum(1 for p in dst.rglob("*") if p.is_file())
    print(f"\nfiles in  {src}: {len(files)}")
    print(f"files in  {dst}: {n_out}")
    if n_out != len(files):
        print("!! counts differ — check before uploading")
    else:
        print("Counts match. Filenames and folder structure preserved.")


if __name__ == "__main__":
    main()
