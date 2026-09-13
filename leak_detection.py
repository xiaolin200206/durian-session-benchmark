#!/usr/bin/env python3
"""
leak_detection.py — what does a content-based audit actually detect?
====================================================================
An earlier version of this analysis divided the leakage rate measured under a
proxy grouping by the leakage rate measured under the reference grouping and
called the result "the share of real leakage the audit sees". That quantity is
not a detection rate. Two aggregate rates can coincide while flagging
different images, and a proxy that merges unrelated images can flag test
images the reference grouping does not — which is why the trivial grouping
scored 106.5% and no true detection rate can exceed 100%.

This script answers the question the ratio was standing in for, at the level
at which it is defined: for each test image, on one partition, does the
reference grouping say it shares a group with training, and does the proxy say
so too?

                          proxy: leaked   proxy: clean
    reference: leaked          TP              FN        <- missed by the audit
    reference: clean           FP              TN        <- invented by the audit

and reports, over repeated simulated image-level partitions:

    recall      TP / (TP + FN)   of the test images that genuinely share a
                                 specimen with training, how many does the
                                 audit flag?
    precision   TP / (TP + FP)   of the ones it flags, how many genuinely do?

Both are needed. A proxy can reach high recall by over-merging, in which case
precision collapses; the pair of numbers distinguishes an audit that works
from one that flags everything. This is the quantity to quote when describing
what a duplicate check establishes about a released dataset.

The reference grouping is not ground truth. It is the grouping recovered from
filename structure, with the uncertainty characterised in Sect. 4.1 of the
paper. Everything below is agreement with that reference, not with the truth.

Usage
-----
    # hash proxy against the reference grouping
    python leak_detection.py --root images_512 --sessions sessions.csv \\
        --method dhash --thresholds 2,4,6,8,10,12 --out detect

    # learned-representation proxy, features from extract_features.py
    python leak_detection.py --root images_512 --sessions sessions.csv \\
        --method embed --embeddings dinov2_cls.npz \\
        --thresholds 0.80,0.85,0.86,0.88,0.90,0.95 \\
        --linkage single --out detect_dinov2

    # does chaining explain the degenerate groupings? compare linkages
    python leak_detection.py ... --linkage complete
    python leak_detection.py ... --linkage average

Outputs
    detection.csv   one row per threshold: TP/FP/FN/TN, recall, precision, F1,
                    plus the two marginal leak rates so the old ratio can still
                    be computed by anyone who wants it, clearly labelled as a
                    ratio of marginals and not a detection rate
    detection.md    the same as a table for the supplement
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

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
TRAIN = {"train", "training", "train_set"}
VAL = {"val", "valid", "validation", "dev"}
TEST = {"test", "testing", "test_set", "eval"}
POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)


def split_name(n):
    n = n.strip().lower().replace("-", "_").replace(" ", "_")
    return "train" if n in TRAIN else "val" if n in VAL else "test" if n in TEST else None


def scan(root):
    root = Path(root)
    top = [d for d in sorted(root.iterdir()) if d.is_dir()]
    sd = [d for d in top if split_name(d.name)]
    recs = []
    if len(sd) >= 2:
        for d in sd:
            for c in sorted(d.iterdir()):
                if c.is_dir():
                    for f in sorted(c.rglob("*")):
                        if f.is_file() and f.suffix.lower() in IMG_EXT:
                            recs.append((c.name.strip(), f.name, str(f)))
    else:
        for c in top:
            for f in sorted(c.rglob("*")):
                if f.is_file() and f.suffix.lower() in IMG_EXT:
                    recs.append((c.name.strip(), f.name, str(f)))
    return recs


def dhash(img, size=8):
    px = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    return np.packbits((px[:, :-1] > px[:, 1:]).flatten())


class UF:
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


def cluster(sim_or_dist, classes, threshold, mode, linkage):
    """Cluster within class.

    single    join whenever any cross-cluster pair passes the threshold
    complete  join only when every cross-cluster pair passes it
    average   join when the mean cross-cluster similarity passes it

    Single linkage is permissive and chains; the other two are included so a
    degenerate grouping can be attributed to the linkage rule or acquitted of it.
    """
    n = len(classes)
    labels = np.empty(n, dtype=object)
    by_cls = defaultdict(list)
    for i, c in enumerate(classes):
        by_cls[c].append(i)

    for cls, idxs in by_cls.items():
        idxs = np.asarray(idxs)
        m = len(idxs)
        if m == 1:
            labels[idxs[0]] = f"{cls}:c0"
            continue
        if mode == "embed":
            S = sim_or_dist[idxs] @ sim_or_dist[idxs].T
            passes = S >= threshold
        else:
            H = sim_or_dist[idxs]
            xor = np.bitwise_xor(H[:, None, :], H[None, :, :])
            D = POPCOUNT[xor].sum(-1)
            passes = D <= threshold
            S = -D.astype(float)
        np.fill_diagonal(passes, False)

        if linkage == "single":
            uf = UF(m)
            for a, b in zip(*np.where(np.triu(passes, 1))):
                uf.union(int(a), int(b))
            groups = defaultdict(list)
            for k in range(m):
                groups[uf.find(k)].append(k)
            members = list(groups.values())
        else:
            # agglomerative with a hard stop, small m so O(m^3) is acceptable
            members = [[k] for k in range(m)]
            merged = True
            while merged:
                merged = False
                for a in range(len(members)):
                    for b in range(a + 1, len(members)):
                        ia, ib = np.ix_(members[a], members[b])
                        block = passes[ia, ib] if linkage == "complete" else S[ia, ib]
                        ok = block.all() if linkage == "complete" else block.mean() >= threshold
                        if ok:
                            members[a] = members[a] + members[b]
                            members.pop(b)
                            merged = True
                            break
                    if merged:
                        break
        for gi, mem in enumerate(members):
            for k in mem:
                labels[idxs[k]] = f"{cls}:c{gi}"
    return labels


def simulate(classes, seed, ratios=(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(seed)
    a = np.empty(len(classes), dtype=object)
    classes = np.asarray(classes)
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        ntr = int(round(n * ratios[0]))
        nva = int(round(n * ratios[1]))
        a[idx[:ntr]] = "train"
        a[idx[ntr:ntr + nva]] = "val"
        a[idx[ntr + nva:]] = "test"
    return a


def flags(assign, groups):
    """Per-test-image boolean: does this image share a group with training?"""
    groups = np.asarray(groups)
    tr = set(groups[assign == "train"])
    is_te = assign == "test"
    return is_te, np.array([g in tr for g in groups[is_te]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--sessions", required=True, help="reference grouping (sessions.csv)")
    ap.add_argument("--method", choices=["dhash", "embed"], default="dhash")
    ap.add_argument("--embeddings", help="required for --method embed")
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--linkage", choices=["single", "complete", "average"], default="single")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default="detect")
    args = ap.parse_args()

    recs = scan(args.root)
    if not recs:
        sys.exit(f"No images under {args.root}")
    ref = pd.read_csv(args.sessions)
    if not {"cls", "file", "session"}.issubset(ref.columns):
        sys.exit("sessions.csv needs columns cls, file, session")
    table = {(r.cls, r.file): r.session for r in ref.itertuples()}
    missing = [f"{c}/{f}" for c, f, _ in recs if (c, f) not in table]
    if missing:
        sys.exit(f"{len(missing)} images absent from the reference grouping, "
                 f"e.g. {missing[0]}. Detection cannot be scored on a partial overlap.")

    classes = [c for c, _, _ in recs]
    paths = [p for _, _, p in recs]
    reference = [table[(c, f)] for c, f, _ in recs]
    print(f"{len(recs)} images, {len(set(reference))} reference groups, "
          f"{len(set(classes))} classes, linkage={args.linkage}")

    if args.method == "embed":
        if not args.embeddings:
            sys.exit("--method embed needs --embeddings")
        z = np.load(args.embeddings, allow_pickle=True)
        index = {p: i for i, p in enumerate(list(z["paths"]))}
        if any(p not in index for p in paths):
            sys.exit("Embedding file does not cover this image tree; re-extract.")
        E = np.asarray(z["E"], dtype=np.float32)[[index[p] for p in paths]]
        data = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        ths = [float(t) for t in args.thresholds.split(",")]
        tag = str(z["weights"]) if "weights" in z else Path(args.embeddings).name
    else:
        print("hashing...")
        data = np.stack([dhash(Image.open(p).convert("RGB")) for p in paths])
        ths = [int(t) for t in args.thresholds.split(",")]
        tag = "dhash"

    seeds = [int(x) for x in args.seeds.split(",")]
    partitions = [simulate(classes, s) for s in seeds]

    rows = []
    for t in ths:
        proxy = cluster(data, classes, t, args.method, args.linkage)
        TP = FP = FN = TN = 0
        ref_rate, proxy_rate = [], []
        for a in partitions:
            is_te, rf = flags(a, reference)
            _, pf = flags(a, proxy)
            TP += int((rf & pf).sum())
            FN += int((rf & ~pf).sum())
            FP += int((~rf & pf).sum())
            TN += int((~rf & ~pf).sum())
            ref_rate.append(100 * rf.mean())
            proxy_rate.append(100 * pf.mean())
        rec = TP / (TP + FN) if TP + FN else None
        pre = TP / (TP + FP) if TP + FP else None
        f1 = 2 * pre * rec / (pre + rec) if (pre and rec) else None
        row = dict(method=tag, linkage=args.linkage, threshold=t,
                   clusters=len(set(proxy)),
                   TP=TP, FP=FP, FN=FN, TN=TN,
                   recall=None if rec is None else round(100 * rec, 1),
                   precision=None if pre is None else round(100 * pre, 1),
                   f1=None if f1 is None else round(100 * f1, 1),
                   ref_leak_pct=round(float(np.mean(ref_rate)), 1),
                   proxy_leak_pct=round(float(np.mean(proxy_rate)), 1))
        row["ratio_of_marginals_NOT_a_detection_rate"] = (
            round(100 * row["proxy_leak_pct"] / row["ref_leak_pct"], 1)
            if row["ref_leak_pct"] else None)
        rows.append(row)
        print(f"  t={t:<6} clusters {row['clusters']:>5}  "
              f"recall {str(row['recall']):>6}%  precision {str(row['precision']):>6}%  "
              f"missed {FN:>5}  invented {FP:>5}")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "detection.csv", index=False)

    cols = ["threshold", "clusters", "recall", "precision", "f1", "FN", "FP",
            "ref_leak_pct", "proxy_leak_pct"]
    head = ["Threshold", "Clusters", "Detection recall (%)", "Detection precision (%)",
            "F1 (%)", "Missed", "Invented", "Reference leak (%)", "Proxy leak (%)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join("—" if r[c] is None else str(r[c]) for c in cols) + " |")
    lines += ["", f"*Proxy: {tag}, {args.linkage} linkage, clustering within class. "
                  f"Detection recall and precision are computed per test image over "
                  f"{len(seeds)} simulated stratified image-level partitions, against the "
                  f"grouping recovered from filename structure. The last two columns are "
                  f"marginal rates; their ratio is not a detection rate, because a proxy "
                  f"that over-merges flags test images the reference grouping does not.*"]
    (out / "detection.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\ntable -> {out / 'detection.csv'}\n         {out / 'detection.md'}")
    best = df.loc[df["f1"].idxmax()] if df["f1"].notna().any() else None
    if best is not None:
        print(f"\nBest detection F1 at threshold {best['threshold']}: "
              f"recall {best['recall']}%, precision {best['precision']}%. "
              f"Quote this pair, not a ratio.")


if __name__ == "__main__":
    main()
