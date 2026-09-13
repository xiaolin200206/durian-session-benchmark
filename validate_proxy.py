#!/usr/bin/env python3
"""
validate_proxy.py — is a perceptual hash a usable stand-in for a capture session?
================================================================================
`leakage_survey.py` infers clusters from image content, because most published
datasets have been renumbered and their capture structure is gone. That makes
every cross-dataset number a proxy measurement, and a proxy is worth nothing
until somebody calibrates it.

The Malaysian dataset is the only one where both are available: sessions read
off camera filename structure (bursts, video frames, messaging batches) and
clusters inferred from pixels. So it is the only place the proxy can be
checked. This script does that check and reports the one quantity the survey
argument depends on:

    RECALL  — of all pairs of images that genuinely belong to one capture
              session, what fraction does the hash put in one cluster?

If recall is high, a hash-derived figure on a renumbered dataset is close to
the truth. If recall is low, every such figure is a LOWER BOUND and the survey
must say so in those words. Either answer is publishable; not knowing which
is not.

Precision is reported alongside but matters less here. A hash that merges two
different sessions is conservative for this argument: it enlarges groups and
therefore removes more data from training under grouped partitioning.

The calibration figure
----------------------
The abstract statistics (ARI, AMI) are reported because reviewers expect them,
but the number to quote in the paper is the last one: on identical simulated
image-level splits, the leakage measured with true sessions against the
leakage measured with hash clusters. Their ratio is how much of the real
leakage a hash-only audit can see, expressed in the units the paper actually
uses.

Usage
-----
    python validate_proxy.py --root <512px image tree> --sessions sessions.csv

    # if sessions.csv covers only part of the tree, or paths have moved
    python validate_proxy.py --root ... --sessions ... --lenient

Outputs (under --out, default `proxy/`)
    calibration.csv    one row per threshold
    calibration.md     the paragraph to paste into the methods section
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


def dhash(img, size=8):
    small = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = np.asarray(small, dtype=np.int16)
    return np.packbits((px[:, :-1] > px[:, 1:]).flatten())


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


def cluster(H, classes, threshold, block=256):
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
        roots = {}
        for k in range(m):
            r = uf.find(k)
            roots.setdefault(r, len(roots))
            session[idxs[k]] = f"{cls}:h{roots[r]}"
    return session


def pair_scores(truth, pred, classes):
    """Pair-counting precision and recall, computed within class.

    Both groupings are constructed within class, so cross-class pairs are
    trivially agreed and would inflate every score if included.
    """
    tp = same_truth = same_pred = 0
    df = pd.DataFrame(dict(cls=classes, t=truth, p=pred))
    for _, g in df.groupby("cls"):
        ct = pd.crosstab(g["t"], g["p"]).values
        n_ij = ct.astype(np.int64)
        tp += int((n_ij * (n_ij - 1) // 2).sum())
        a = n_ij.sum(1)
        b = n_ij.sum(0)
        same_truth += int((a * (a - 1) // 2).sum())
        same_pred += int((b * (b - 1) // 2).sum())
    recall = tp / same_truth if same_truth else None
    precision = tp / same_pred if same_pred else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else None)
    return dict(pair_recall=recall, pair_precision=precision, pair_f1=f1,
                true_pairs=same_truth, pred_pairs=same_pred, tp_pairs=tp)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="image tree (use the 512 px copies)")
    ap.add_argument("--sessions", required=True, help="sessions.csv from session_split.py")
    ap.add_argument("--thresholds", default="2,4,6,8,10,12")
    ap.add_argument("--sim-seeds", default="0,1,2,3,4")
    ap.add_argument("--lenient", action="store_true",
                    help="drop images absent from sessions.csv instead of failing")
    ap.add_argument("--out", default="proxy")
    args = ap.parse_args()

    recs = scan(args.root)
    if not recs:
        sys.exit(f"No images under {args.root}")
    print(f"images found : {len(recs)}")

    s = pd.read_csv(args.sessions)
    need = {"cls", "file", "session"}
    if not need.issubset(s.columns):
        sys.exit(f"{args.sessions} missing columns: {need - set(s.columns)}")
    table = {(r.cls, r.file): r.session for r in s.itertuples()}
    print(f"sessions.csv : {len(table)} entries, {s.session.nunique()} sessions")

    missing = [r for r in recs if (r["cls"], r["file"]) not in table]
    if missing:
        print(f"\n!! {len(missing)} images have no entry in sessions.csv, e.g.")
        for r in missing[:5]:
            print(f"     {r['cls']}/{r['file']}")
        if not args.lenient:
            sys.exit("\nRegenerate sessions.csv against this image tree, or pass "
                     "--lenient to drop these. Do not calibrate on a partial "
                     "overlap without knowing which images are absent.")
        recs = [r for r in recs if (r["cls"], r["file"]) in table]
        print(f"   dropped; {len(recs)} images remain")

    print("hashing...")
    H, keep = [], []
    for i, r in enumerate(recs):
        if i and i % 500 == 0:
            print(f"  {i}/{len(recs)}")
        try:
            with Image.open(r["path"]) as im:
                im.load()
                H.append(dhash(im))
            keep.append(r)
        except Exception as exc:
            print(f"  unreadable, skipped: {r['path']}  {exc!r}")
    H = np.stack(H)
    recs = keep

    classes = [r["cls"] for r in recs]
    truth = [table[(r["cls"], r["file"])] for r in recs]
    n_true = len(set(truth))
    print(f"\nhashed {len(recs)} images, {n_true} filename-derived sessions, "
          f"{len(set(classes))} classes")

    seeds = [int(x) for x in args.sim_seeds.split(",") if x.strip()]
    splits = list(simulate_splits(classes, seeds))
    true_leaks = [sibling_leak(a, truth) for a in splits]
    true_leak = float(np.mean([x for x in true_leaks if x is not None]))
    print(f"leakage under TRUE sessions, simulated image-level splits: "
          f"{true_leak:.1f}%")

    rows = []
    for t in [int(x) for x in args.thresholds.split(",")]:
        pred = cluster(H, classes, t)
        sc = pair_scores(truth, pred, classes)
        hash_leaks = [sibling_leak(a, pred) for a in splits]
        hash_leak = float(np.mean([x for x in hash_leaks if x is not None]))
        row = dict(threshold=t, hash_clusters=len(set(pred)), true_sessions=n_true,
                   pair_recall=None if sc["pair_recall"] is None else round(sc["pair_recall"], 3),
                   pair_precision=None if sc["pair_precision"] is None else round(sc["pair_precision"], 3),
                   pair_f1=None if sc["pair_f1"] is None else round(sc["pair_f1"], 3),
                   true_leak_pct=round(true_leak, 1),
                   hash_leak_pct=round(hash_leak, 1),
                   leak_recovered_pct=round(100 * hash_leak / true_leak, 1) if true_leak else None)
        if _HAS_SK:
            row["ARI"] = round(adjusted_rand_score(truth, list(pred)), 3)
            row["AMI"] = round(adjusted_mutual_info_score(truth, list(pred)), 3)
        rows.append(row)
        print(f"  t={t:>2}  clusters {row['hash_clusters']:>5} (vs {n_true} true)  "
              f"pair-recall {row['pair_recall']}  pair-prec {row['pair_precision']}  "
              f"hash-leak {row['hash_leak_pct']}%  "
              f"= {row['leak_recovered_pct']}% of true")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "calibration.csv", index=False)

    best = df.loc[df["pair_f1"].idxmax()] if df["pair_f1"].notna().any() else df.iloc[0]
    rec = best["pair_recall"]
    got = best["leak_recovered_pct"]

    print(f"\n{'=' * 72}\nVERDICT\n{'=' * 72}")
    print(f"Best threshold by pair-F1: {int(best['threshold'])}")
    if rec is not None:
        print(f"  recovers {rec:.1%} of genuine within-session pairs")
    print(f"  recovers {got}% of the leakage that true sessions reveal")
    if rec is not None and rec >= 0.8:
        print("\nThe hash is a close stand-in. Hash-derived figures on renumbered")
        print("datasets can be reported as estimates, with this calibration cited.")
    elif rec is not None and rec >= 0.4:
        print("\nThe hash sees much but not most of the structure. Report every")
        print("hash-derived figure as a LOWER BOUND and give this calibration")
        print("as the reason. That is a defensible claim and a stronger one")
        print("than an unqualified estimate.")
    else:
        print("\nThe hash sees only a fraction of the real dependence. Every")
        print("hash-derived figure in the survey is a lower bound, and the")
        print("survey's conclusion is not that other datasets are clean but")
        print("that their true structure is unrecoverable from what was")
        print("released. Say exactly that.")

    def pct(v):
        return "not defined (no multi-image clusters)" if v is None else f"{v:.1%}"

    para = (
        f"Cluster membership inferred by perceptual hashing was calibrated "
        f"against capture sessions recovered from camera filename structure on "
        f"the Malaysian dataset, the only collection for which both are "
        f"available. At a Hamming threshold of {int(best['threshold'])} the hash "
        f"recovered {best['hash_clusters']} clusters against {n_true} "
        f"filename-derived sessions, capturing "
        f"{pct(rec)} of genuine within-session image pairs at a pair-level "
        f"precision of {pct(best['pair_precision'])}. Measured on identical "
        f"simulated image-level partitions, hash-derived grouping revealed "
        f"{best['hash_leak_pct']}% test-set leakage against {best['true_leak_pct']}% "
        f"under true sessions, i.e. {got}% of the leakage that the capture "
        f"metadata exposes. Hash-derived figures reported for datasets without "
        f"recoverable capture metadata are therefore "
        f"{'estimates' if (rec or 0) >= 0.8 else 'lower bounds'}."
    )
    (out / "calibration.md").write_text(para + "\n", encoding="utf-8")
    print(f"\nmethods paragraph -> {out / 'calibration.md'}")
    print(f"table             -> {out / 'calibration.csv'}")


if __name__ == "__main__":
    main()
