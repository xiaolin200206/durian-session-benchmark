#!/usr/bin/env python3
"""
extract_features.py — feature extractors for the capture-session calibration
============================================================================
`embed_proxy.py` showed that ImageNet-pretrained EfficientNet-B0 features do
not recover capture sessions: no operating point reaches usable recall and
precision together, and the best pair-F1 (0.316) barely exceeds the trivial
grouping that puts every image of a class in one cluster (0.298).

The obvious objection is that ImageNet classification features are trained to
be invariant to exactly what this task needs — which instance it is. Models
trained for instance-level correspondence should do better. If they do not,
the claim stops being about one backbone and becomes a claim about post hoc
recovery as such, which is what the reporting protocol needs.

This script produces embeddings from those models in the `.npz` format
`embed_proxy.py` reads, so the calibration is re-run without changing the
evaluation code:

    python extract_features.py --root images_512 --backend hub \\
        --model dinov2_vitb14 --out dinov2.npz

    python embed_proxy.py --root images_512 --sessions sessions.csv \\
        --embeddings dinov2.npz --sims 0.5,0.6,0.7,0.8,0.9,0.95 \\
        --out proxy_dinov2

Backends
    hub          torch.hub, for DINOv2 (needs one-off internet access)
                   dinov2_vits14 / dinov2_vitb14 / dinov2_vitl14
    timm         any timm model, e.g. vit_base_patch14_dinov2.lvd142m,
                   vit_base_patch16_clip_224.openai
    torchvision  resnet50, convnext_tiny, ... (the ImageNet baseline)

Pooling
    cls    the class token (DINOv2 default, good for image-level identity)
    mean   mean of patch tokens; often stronger for instance matching, and
           worth reporting alongside cls so the negative result does not
           rest on one pooling choice

The file ordering is produced by the same scan used by embed_proxy.py and
validate_proxy.py, so the three agree on which row is which image. Do not
reorder the tree between extraction and calibration.

Choose the SAME image tree you calibrate on — the 512 px copies, not the
full-resolution originals. Sect. 3.5 of the manuscript shows the resampling
path changes results.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow missing.  pip install pillow")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TRAIN_NAMES = {"train", "training", "train_set"}
VAL_NAMES = {"val", "valid", "validation", "dev"}
TEST_NAMES = {"test", "testing", "test_set", "eval"}


def normalise_split(name):
    n = name.strip().lower().replace("-", "_").replace(" ", "_")
    if n in TRAIN_NAMES:
        return "train"
    if n in VAL_NAMES:
        return "val"
    if n in TEST_NAMES:
        return "test"
    return None


def scan(root):
    """Byte-for-byte the scan of embed_proxy.py and validate_proxy.py."""
    root = Path(root)
    top = [d for d in sorted(root.iterdir()) if d.is_dir()]
    split_dirs = [d for d in top if normalise_split(d.name)]
    recs = []
    if len(split_dirs) >= 2:
        for d in split_dirs:
            sp = normalise_split(d.name)
            for cls_dir in sorted(d.iterdir()):
                if cls_dir.is_dir():
                    for f in sorted(cls_dir.rglob("*")):
                        if f.is_file() and f.suffix.lower() in IMG_EXT:
                            recs.append(dict(split=sp, cls=cls_dir.name.strip(),
                                             file=f.name, path=str(f)))
    else:
        for cls_dir in top:
            for f in sorted(cls_dir.rglob("*")):
                if f.is_file() and f.suffix.lower() in IMG_EXT:
                    recs.append(dict(split="__none__", cls=cls_dir.name.strip(),
                                     file=f.name, path=str(f)))
    return recs


def save_npz(path, E, paths, tag):
    """Write in the format embed_proxy.py --embeddings expects."""
    E = np.asarray(E, dtype=np.float32)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    if np.any(norms == 0):
        print("  !! some embeddings are all-zero; check for unreadable images")
    E = E / (norms + 1e-9)
    np.savez_compressed(path, E=E, paths=np.array(list(paths), dtype=object),
                        weights=tag)
    return E


def build_model(backend, model_name, device):
    import torch
    if backend == "hub":
        print(f"  torch.hub: facebookresearch/dinov2 -> {model_name}")
        print("  (first run downloads weights; needs internet once)")
        m = torch.hub.load("facebookresearch/dinov2", model_name)
        default_size = 224
    elif backend == "timm":
        try:
            import timm
        except ImportError:
            sys.exit("timm missing.  pip install timm")
        m = timm.create_model(model_name, pretrained=True, num_classes=0)
        cfg = getattr(m, "default_cfg", {}) or {}
        default_size = (cfg.get("input_size") or (3, 224, 224))[-1]
        print(f"  timm: {model_name}, native input {default_size}")
    elif backend == "torchvision":
        from torchvision import models
        builder = getattr(models, model_name, None)
        if builder is None:
            sys.exit(f"torchvision has no model {model_name!r}")
        try:
            m = builder(weights="IMAGENET1K_V1")
        except TypeError:
            m = builder(pretrained=True)
        for attr in ("classifier", "fc", "head"):
            if hasattr(m, attr):
                setattr(m, attr, torch.nn.Identity())
                break
        default_size = 224
    else:
        sys.exit(f"unknown backend {backend!r}")
    return m.eval().to(device), default_size


def forward_features(model, x, pool):
    """CLS token or mean of patch tokens, whichever the model exposes."""
    if pool == "mean" and hasattr(model, "forward_features"):
        out = model.forward_features(x)
        if isinstance(out, dict):                     # DINOv2 hub models
            if "x_norm_patchtokens" in out:
                return out["x_norm_patchtokens"].mean(1)
            if "x_norm_clstoken" in out:
                return out["x_norm_clstoken"]
            out = next(iter(out.values()))
        if out.ndim == 3:                             # (B, tokens, dim)
            return out[:, 1:, :].mean(1) if out.shape[1] > 1 else out[:, 0, :]
        if out.ndim == 4:                             # (B, C, H, W)
            return out.mean((2, 3))
        return out
    out = model(x)
    if isinstance(out, dict):
        out = out.get("x_norm_clstoken", next(iter(out.values())))
    return out.flatten(1) if out.ndim > 2 else out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="image tree (use the 512 px copies)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--backend", default="hub", choices=["hub", "timm", "torchvision"])
    ap.add_argument("--model", default="dinov2_vitb14")
    ap.add_argument("--pool", default="cls", choices=["cls", "mean"])
    ap.add_argument("--size", type=int, default=None,
                    help="input size; defaults to the model's native size. "
                         "ViT/14 models need a multiple of 14 (224, 518).")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    recs = scan(args.root)
    if not recs:
        sys.exit(f"No images under {args.root}")
    paths = [r["path"] for r in recs]
    print(f"images: {len(paths)}")

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
    except ImportError:
        sys.exit("torch/torchvision missing.")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"device: {device}")

    model, native = build_model(args.backend, args.model, device)
    size = args.size or native
    patch = 14 if "14" in args.model else (16 if "16" in args.model else None)
    if patch and size % patch:
        adj = (size // patch) * patch
        print(f"  !! {size} is not a multiple of {patch}; using {adj}")
        size = adj
    print(f"  input size {size}, pooling '{args.pool}'")

    tf = transforms.Compose([
        transforms.Resize(int(round(size * 256 / 224))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class DS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            with Image.open(paths[i]) as im:
                im.load()
                return tf(im.convert("RGB"))

    loader = DataLoader(DS(), batch_size=args.batch, shuffle=False, num_workers=0)
    feats = []
    with torch.no_grad():
        for i, x in enumerate(loader):
            feats.append(forward_features(model, x.to(device), args.pool).cpu().numpy())
            if i and i % 5 == 0:
                print(f"  {i * args.batch}/{len(paths)}")
    E = np.concatenate(feats, 0)
    tag = f"{args.backend}:{args.model} pool={args.pool} size={size}"
    E = save_npz(args.out, E, paths, tag)
    print(f"\nembeddings {E.shape} -> {args.out}")
    print(f"weights tag: {tag}")
    print("\nNext:")
    print(f'  python embed_proxy.py --root {args.root} --sessions sessions.csv \\')
    print(f'      --embeddings {args.out} --sims 0.5,0.6,0.7,0.8,0.85,0.9,0.95 \\')
    print(f'      --out proxy_{Path(args.out).stem}')


if __name__ == "__main__":
    main()
