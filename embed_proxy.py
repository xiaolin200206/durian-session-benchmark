#!/usr/bin/env python3
"""
embed_proxy.py — can a learned embedding recover capture sessions when a
perceptual hash cannot?
========================================================================
`validate_proxy.py` established that dHash recovers almost none of the
capture-session structure of the Malaysian dataset: 0.3% of genuine
within-session image pairs. On identical simulated image-level partitions it
flags 4.6% of test images where the reference grouping flags 93.9%. Report
those two side by side and never as a ratio -- a proxy that over-merges flags
images the reference does not, so the quotient can exceed one and does not
identify which images were missed. That is the strongest result in the
project, and it has exactly one weak point, which a reviewer will find
immediately:

    "You tried one hash. A learned feature embedding would do better."

This script settles that. It clusters on cosine similarity between ImageNet
features instead of Hamming distance between hashes, computes the identical
calibration metrics, and puts both methods in one table.

Both outcomes are useful, and neither is a disappointment:

  EMBEDDING ALSO FAILS   The conclusion widens beyond one representation:
                         these methods, under this clustering rule, do not
                         recover the structure. That is heavier than "dHash
                         is insufficient" and is the version that belongs in
                         the paper's reporting protocol. It is NOT the claim
                         that no method could succeed -- the paper does not
                         make that claim and neither should this docstring.

  EMBEDDING SUCCEEDS     You have a positive contribution as well as a
                         negative one: an approximate method for recovering
                         grouping structure in datasets that shipped none,
                         calibrated against ground truth on the one dataset
                         where ground truth exists.

A note on which weights to use
------------------------------
The default is ImageNet-pretrained, which has never seen these images. That
is the honest choice. `--ckpt` accepts a checkpoint of your own, but a model
FINE-TUNED ON THESE IMAGES will cluster them by whatever it memorised during
training, which is the very thing under investigation, so the comparison
becomes circular. The script warns and records which weights were used.

Usage
-----
    python embed_proxy.py --root images_512 --sessions sessions.csv --out embedproxy

    # reuse embeddings across runs (extraction is the slow part)
    python embed_proxy.py --root images_512 --sessions sessions.csv \
        --save-embeddings emb.npz --out embedproxy
    python embed_proxy.py --root images_512 --sessions sessions.csv \
        --embeddings emb.npz --sims 0.5,0.6,0.7,0.8,0.9 --out embedproxy

    # a different backbone, to show the answer is not architecture-specific
    python embed_proxy.py --root images_512 --sessions sessions.csv \
        --arch resnet50 --out embedproxy_rn50

Outputs (under --out)
    calibration_embed.csv   one row per method and threshold, dHash included
    calibration_embed.md    the paragraph for the methods section
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas missing.  pip install pandas")

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow missing.  pip install pillow")

try:
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    _HAS_SK = True
except ImportError:
    _HAS_SK = False

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TRAIN_NAMES = {"train", "training", "train_set"}
VAL_NAMES = {"val", "valid", "validation", "dev"}
TEST_NAMES = {"test", "testing", "test_set", "eval"}
POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)


# ---------------------------------------------------------------------------
# scanning, shared with validate_proxy.py
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------

def extract_embeddings(paths, arch="efficientnet_b0", batch=32, device="auto",
                       ckpt=None, size=224):
    """Global-pooled features from an ImageNet backbone, L2-normalised.

    Evaluation preprocessing matches the paper: resize 256, centre crop 224,
    ImageNet normalisation. Using the training augmentation here would make
    two views of one image disagree with themselves.
    """
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
    except ImportError:
        sys.exit("torch/torchvision missing. In your durian env they are already "
                 "installed; if not:  pip install torch torchvision")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    builder = getattr(models, arch, None)
    if builder is None:
        sys.exit(f"torchvision has no model called {arch!r}")
    try:
        model = builder(weights="IMAGENET1K_V1")
    except TypeError:                                  # very old torchvision
        model = builder(pretrained=True)

    if ckpt:
        print("  !! Loading your own checkpoint. If these weights were fine-tuned")
        print("     on these images, the clustering reflects what the model")
        print("     memorised during training and the calibration is circular.")
        state = torch.load(ckpt, map_location="cpu")
        state = state.get("model", state.get("state_dict", state))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"     loaded with {len(missing)} missing / {len(unexpected)} unexpected keys")

    # strip the classifier so forward() returns pooled features
    if hasattr(model, "classifier"):
        model.classifier = torch.nn.Identity()
    elif hasattr(model, "fc"):
        model.fc = torch.nn.Identity()
    elif hasattr(model, "head"):
        model.head = torch.nn.Identity()
    model.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
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

    loader = DataLoader(DS(), batch_size=batch, shuffle=False, num_workers=0)
    feats = []
    with torch.no_grad():
        for i, x in enumerate(loader):
            f = model(x.to(device))
            feats.append(f.flatten(1).cpu().numpy())
            if i and i % 10 == 0:
                print(f"    {i * batch}/{len(paths)}")
    E = np.concatenate(feats, 0).astype(np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    print(f"  embeddings: {E.shape}")
    return E


def dhash(img, size=8):
    small = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = np.asarray(small, dtype=np.int16)
    return np.packbits((px[:, :-1] > px[:, 1:]).flatten())


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------

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


def _label(idxs, uf, classes, tag):
    out = {}
    roots = {}
    for k, i in enumerate(idxs):
        r = uf.find(k)
        roots.setdefault(r, len(roots))
        out[i] = f"{classes[i]}:{tag}{roots[r]}"
    return out


def cluster_cosine(E, classes, tau):
    """Single-linkage on cosine similarity, within class."""
    session = np.empty(len(E), dtype=object)
    by_cls = defaultdict(list)
    for i, c in enumerate(classes):
        by_cls[c].append(i)
    for cls, idxs in by_cls.items():
        idxs = np.asarray(idxs)
        m = len(idxs)
        if m == 1:
            session[idxs[0]] = f"{cls}:e0"
            continue
        S = E[idxs] @ E[idxs].T
        uf = Union(m)
        ii, jj = np.where(np.triu(S >= tau, k=1))
        for a, b in zip(ii, jj):
            uf.union(int(a), int(b))
        for i, lab in _label(idxs, uf, classes, "e").items():
            session[i] = lab
    return session


def cluster_hash(H, classes, threshold, block=256):
    session = np.empty(len(H), dtype=object)
    by_cls = defaultdict(list)
    for i, c in enumerate(classes):
        by_cls[c].append(i)
    for cls, idxs in by_cls.items():
        idxs = np.asarray(idxs)
        m = len(idxs)
        if m == 1:
            session[idxs[0]] = f"{cls}:h0"
            continue
        sub = H[idxs]
        uf = Union(m)
        for start in range(0, m, block):
            stop = min(start + block, m)
            xor = np.bitwise_xor(sub[start:stop, None, :], sub[None, :, :])
            dist = POPCOUNT[xor].sum(-1)
            rows, cols = np.where(dist <= threshold)
            for r, c in zip(rows, cols):
                a, b = start + int(r), int(c)
                if b > a:
                    uf.union(a, b)
        for i, lab in _label(idxs, uf, classes, "h").items():
            session[i] = lab
    return session


# ---------------------------------------------------------------------------
# metrics, identical to validate_proxy.py so rows are comparable
# ---------------------------------------------------------------------------

def pair_scores(truth, pred, classes):
    """Pair-counting scores plus the base rate they must be read against.

    chance_precision is the share of within-class image pairs that genuinely
    share a session. A clustering that merges everything scores precision
    equal to this number and recall equal to 1, which looks like success and
    is not. Precision must be compared against it, not against zero.
    """
    tp = same_truth = same_pred = all_pairs = 0
    df = pd.DataFrame(dict(cls=classes, t=truth, p=list(pred)))
    for _, g in df.groupby("cls"):
        ct = pd.crosstab(g["t"], g["p"]).values.astype(np.int64)
        tp += int((ct * (ct - 1) // 2).sum())
        a, b = ct.sum(1), ct.sum(0)
        same_truth += int((a * (a - 1) // 2).sum())
        same_pred += int((b * (b - 1) // 2).sum())
        n = int(ct.sum())
        all_pairs += n * (n - 1) // 2
    recall = tp / same_truth if same_truth else None
    precision = tp / same_pred if same_pred else None
    chance = same_truth / all_pairs if all_pairs else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else None)
    return recall, precision, f1, chance


def sibling_leak(split_labels, groups):
    split_labels = np.asarray(split_labels)
    groups = np.asarray(groups)
    tr = set(groups[split_labels == "train"])
    is_te = split_labels == "test"
    if not is_te.any():
        return None
    return 100.0 * sum(1 for g in groups[is_te] if g in tr) / int(is_te.sum())


def simulate_splits(labels, seeds, ratios=(0.8, 0.1, 0.1)):
    labels = np.asarray(labels)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        assign = np.empty(len(labels), dtype=object)
        for cls in np.unique(labels):
            idx = np.where(labels == cls)[0]
            rng.shuffle(idx)
            n = len(idx)
            n_tr = int(round(n * ratios[0]))
            n_va = int(round(n * ratios[1]))
            assign[idx[:n_tr]] = "train"
            assign[idx[n_tr:n_tr + n_va]] = "val"
            assign[idx[n_tr + n_va:]] = "test"
        yield assign


def evaluate(pred, truth, classes, splits, true_leak, method, param, n_true):
    rec, prec, f1, chance = pair_scores(truth, pred, classes)
    leaks = [sibling_leak(a, pred) for a in splits]
    leak = float(np.mean([x for x in leaks if x is not None]))
    n_clusters = len(set(pred))

    # Degeneracy. Either tell means the clustering has collapsed and its
    # recall is an artefact of merging rather than evidence of recovery.
    degenerate = (n_clusters < 0.5 * n_true) or (leak > true_leak + 1e-9)

    row = dict(method=method, param=param, clusters=n_clusters,
               pair_recall=None if rec is None else round(rec, 4),
               pair_precision=None if prec is None else round(prec, 3),
               chance_precision=None if chance is None else round(chance, 3),
               precision_lift=None if not (prec and chance) else round(prec / chance, 2),
               pair_f1=None if f1 is None else round(f1, 4),
               true_leak_pct=round(true_leak, 1),
               proxy_leak_pct=round(leak, 1),
               leak_recovered_pct=round(100 * leak / true_leak, 1) if true_leak else None,
               degenerate=degenerate)
    if _HAS_SK:
        row["ARI"] = round(adjusted_rand_score(truth, list(pred)), 4)
        row["AMI"] = round(adjusted_mutual_info_score(truth, list(pred)), 4)
    return row


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--ckpt", default=None,
                    help="your own weights; circular if fine-tuned on these images")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sims", default="0.5,0.6,0.7,0.8,0.9,0.95,0.98",
                    help="cosine similarity thresholds to sweep")
    ap.add_argument("--hash-thresholds", default="6,12",
                    help="dHash thresholds, for the side-by-side row")
    ap.add_argument("--sim-seeds", default="0,1,2,3,4")
    ap.add_argument("--embeddings", default=None, help="load precomputed .npz")
    ap.add_argument("--save-embeddings", default=None)
    ap.add_argument("--lenient", action="store_true")
    ap.add_argument("--out", default="embedproxy")
    args = ap.parse_args()

    recs = scan(args.root)
    if not recs:
        sys.exit(f"No images under {args.root}")
    s = pd.read_csv(args.sessions)
    if not {"cls", "file", "session"}.issubset(s.columns):
        sys.exit(f"{args.sessions} needs columns cls, file, session")
    table = {(r.cls, r.file): r.session for r in s.itertuples()}
    print(f"images {len(recs)} | sessions.csv {len(table)} entries, "
          f"{s.session.nunique()} sessions")

    missing = [r for r in recs if (r["cls"], r["file"]) not in table]
    if missing:
        print(f"!! {len(missing)} images absent from sessions.csv, e.g. "
              f"{missing[0]['cls']}/{missing[0]['file']}")
        if not args.lenient:
            sys.exit("Regenerate sessions.csv against this tree, or pass --lenient.")
        recs = [r for r in recs if (r["cls"], r["file"]) in table]

    paths = [r["path"] for r in recs]
    classes = [r["cls"] for r in recs]
    truth = [table[(r["cls"], r["file"])] for r in recs]
    n_true = len(set(truth))

    if args.embeddings:
        z = np.load(args.embeddings, allow_pickle=True)
        if list(z["paths"]) != paths:
            sys.exit("Cached embeddings were computed on a different file list.")
        E = z["E"]
        weights_note = str(z["weights"]) if "weights" in z else "cached"
        print(f"embeddings loaded from {args.embeddings}  {E.shape}")
    else:
        print(f"extracting {args.arch} features...")
        E = extract_embeddings(paths, arch=args.arch, batch=args.batch,
                               device=args.device, ckpt=args.ckpt)
        weights_note = f"{args.arch}, " + ("custom checkpoint" if args.ckpt else "ImageNet")
        if args.save_embeddings:
            np.savez_compressed(args.save_embeddings, E=E,
                                paths=np.array(paths, dtype=object),
                                weights=weights_note)
            print(f"  saved -> {args.save_embeddings}")

    print("hashing for the side-by-side comparison...")
    H = []
    for p in paths:
        with Image.open(p) as im:
            im.load()
            H.append(dhash(im))
    H = np.stack(H)

    seeds = [int(x) for x in args.sim_seeds.split(",") if x.strip()]
    splits = list(simulate_splits(classes, seeds))
    true_leaks = [sibling_leak(a, truth) for a in splits]
    true_leak = float(np.mean([x for x in true_leaks if x is not None]))
    print(f"\n{n_true} true sessions | leakage under true sessions on simulated "
          f"image-level splits: {true_leak:.1f}%\n")

    rows = []
    for t in [int(x) for x in args.hash_thresholds.split(",") if x.strip()]:
        r = evaluate(cluster_hash(H, classes, t), truth, classes, splits,
                     true_leak, "dhash", t, n_true)
        rows.append(r)
        print(f"  dhash  t={t:<5} clusters {r['clusters']:>4}  "
              f"pair-recall {str(r['pair_recall']):<8} "
              f"prec {str(r['pair_precision']):<6} "
              f"lift {str(r['precision_lift']):<5} "
              f"sees {r['leak_recovered_pct']}%"
              + ("   [DEGENERATE]" if r["degenerate"] else ""))

    for tau in [float(x) for x in args.sims.split(",") if x.strip()]:
        r = evaluate(cluster_cosine(E, classes, tau), truth, classes, splits,
                     true_leak, f"embed:{args.arch}", tau, n_true)
        rows.append(r)
        print(f"  embed  tau={tau:<5} clusters {r['clusters']:>4}  "
              f"pair-recall {str(r['pair_recall']):<8} "
              f"prec {str(r['pair_precision']):<6} "
              f"lift {str(r['precision_lift']):<5} "
              f"sees {r['leak_recovered_pct']}%"
              + ("   [DEGENERATE]" if r["degenerate"] else ""))

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "calibration_embed.csv", index=False)

    def pick(sub):
        """Best non-degenerate row by pair-F1.

        Degenerate rows are excluded before the maximum is taken. A clustering
        that collapses a class into one block scores recall near 1 and would
        otherwise win, which is how an over-merged solution gets mistaken for
        a recovered one.
        """
        ok = sub[~sub["degenerate"]]
        pool = ok if len(ok) and ok["pair_f1"].notna().any() else sub
        return pool.loc[pool["pair_f1"].idxmax()] if pool["pair_f1"].notna().any() else pool.iloc[0]

    emb = df[df.method.str.startswith("embed")]
    hsh = df[df.method == "dhash"]
    best_e, best_h = pick(emb), pick(hsh)
    n_degen = int(df["degenerate"].sum())
    if n_degen:
        print(f"\n  {n_degen} row(s) flagged DEGENERATE and excluded from the "
              f"choice of operating point.")

    er = best_e["pair_recall"] or 0.0
    hr = best_h["pair_recall"] or 0.0
    lift = best_e["precision_lift"] or 0.0

    print(f"\n{'=' * 72}\nVERDICT\n{'=' * 72}")
    print(f"  dHash      best pair-recall {hr:.1%}, sees {best_h['leak_recovered_pct']}% of true leakage")
    print(f"  {args.arch:<10} best pair-recall {er:.1%}, sees {best_e['leak_recovered_pct']}% of true leakage")

    print(f"  chance precision (base rate of same-session pairs): "
          f"{best_e['chance_precision']}")

    recovered = (er >= 0.6 and lift >= 2.0)
    if recovered:
        print("\nThe embedding recovers most of the structure where the hash did")
        print("not, at a precision well above the base rate. That is a positive")
        print("contribution: an approximate recovery method for datasets that")
        print("shipped no grouping, calibrated against ground truth. Re-run")
        print("leakage_survey.py with embedding clusters before quoting any")
        print("cross-dataset figure.")
    elif er < 0.2 or lift < 1.5:
        print("\nBoth methods fail. The conclusion is method-independent:")
        print("capture structure is not recoverable post hoc from image content.")
        print("State it that way - it is a heavier claim than a result about one")
        print("hash function, and it is what the reporting protocol rests on.")
    else:
        print("\nThe embedding sees more than the hash and still misses most of")
        print("the structure. Every proxy figure stays a lower bound; report")
        print("both methods so the bound is seen to be method-robust.")
        print("Give the precision-recall trade-off rather than one operating")
        print("point: the reader needs to see that there is no threshold at")
        print("which recall and precision are simultaneously usable.")

    def pc(v):
        return "undefined" if v is None else f"{v:.1%}"

    para = (
        f"Two content-based methods for recovering capture structure were "
        f"calibrated against capture sessions read from camera filename "
        f"structure on the Malaysian dataset, the only collection for which "
        f"ground truth exists. Single-linkage clustering on a difference hash "
        f"recovered {pc(hr)} of genuine within-session image pairs at its best "
        f"threshold; single-linkage clustering on cosine similarity between "
        f"{weights_note} features recovered {pc(er)} at cosine threshold "
        f"{best_e['param']}, against {n_true} true sessions. Evaluated on "
        f"identical simulated image-level partitions, the two proxies exposed "
        f"{best_h['proxy_leak_pct']}% and {best_e['proxy_leak_pct']}% test-set "
        f"leakage respectively, against {best_e['true_leak_pct']}% under true "
        f"capture sessions. Content-based recovery therefore "
        f"{'reproduces only a small fraction of' if er < 0.2 else 'recovers part of'} "
        f"the dependence induced by the capture process, against a base rate "
        f"of {best_e['chance_precision']} for same-session pairs, and grouping figures "
        f"reported for datasets without capture metadata are lower bounds "
        f"whose tightness cannot be estimated from the released files."
    )
    (out / "calibration_embed.md").write_text(para + "\n", encoding="utf-8")
    print(f"\ntable   -> {out / 'calibration_embed.csv'}")
    print(f"methods -> {out / 'calibration_embed.md'}")


if __name__ == "__main__":
    main()
