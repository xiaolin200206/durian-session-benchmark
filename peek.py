#!/usr/bin/env python3
"""
Quick inventory of the Vietnam dataset folders.
Just tells you what is actually on disk. Changes nothing.

Usage:
    python peek.py
    python peek.py --root "C:\\path\\to\\folder"
"""

import argparse
import os
from collections import Counter

DEFAULT_ROOT = r"C:\Users\Lim Ding Shan\Desktop\Vietnam dataset"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def walk(root, max_depth=4):
    """Print the folder tree with image counts at each leaf."""
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)

    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.count(os.sep) - base_depth
        if depth > max_depth:
            dirnames[:] = []
            continue

        dirnames.sort()
        imgs = [f for f in filenames
                if os.path.splitext(f)[1].lower() in IMG_EXT]
        others = len(filenames) - len(imgs)

        name = os.path.basename(dirpath) or dirpath
        indent = "  " * depth

        if depth == 0:
            print(f"\n{name}/")
        else:
            line = f"{indent}{name}/"
            if imgs:
                line += f"   {len(imgs)} images"
            if others:
                line += f"   (+{others} other files)"
            print(line)

        # show a couple of example filenames so we can see the naming scheme
        if imgs:
            for f in sorted(imgs)[:3]:
                print(f"{indent}    e.g. {f}")
            exts = Counter(os.path.splitext(f)[1].lower() for f in imgs)
            print(f"{indent}    ext: {dict(exts)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--depth", type=int, default=4)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f"Not found: {args.root}")

    walk(args.root, args.depth)

    total = 0
    for dirpath, _, filenames in os.walk(args.root):
        total += sum(1 for f in filenames
                     if os.path.splitext(f)[1].lower() in IMG_EXT)
    print(f"\nTOTAL IMAGES UNDER ROOT: {total}")


if __name__ == "__main__":
    main()
