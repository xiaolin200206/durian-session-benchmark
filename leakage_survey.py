#!/usr/bin/env python3
"""
leakage_survey.py — one comparable leakage row per public dataset
================================================================
`audit_public.py` and `phash_sessions.py` audit one dataset at a time and
print prose. That is right for the dataset you collected and wrong for the
claim this survey has to support, which is about a literature rather than a
dataset. P1_REMAINING.md already states the condition: five or six datasets,
each classified by `classify_redundancy.py` first, so that the numbers
measure the same quantity. This script is that condition, implemented once.

It takes several dataset roots, applies an identical pipeline to each, and
writes ONE tidy row per dataset. Rows concatenate into the survey table.

What it measures, and why each column exists
--------------------------------------------
The headline number in the paper is not "images in straddling clusters"
(79.6%). It is "test images having a same-cluster sibling in train" (96.7%),
because that is the quantity that bears on a reported test score. Both are
computed here, kept separate, and named separately.

Redundancy is split into two kinds before anything is compared, because they
are different phenomena with different fixes:

  BURST  near-identical, genuinely distinct bytes and pixels — the capture
         process. This is what a session-level partition exists to handle.
  COPY   byte-identical or re-encoded duplicates — a dataset-assembly fault.

A dataset whose redundancy is three-quarters COPY (VN-B: 75.8%) must not sit
in the same column as one whose redundancy is BURST. `redundancy_kind` flags
this so the survey table can exclude or footnote it, rather than averaging
two unlike things.

Most published datasets ship no split at all. For those, `--sim-seeds`
simulates the stratified image-level split the literature uses, at several
seeds, and reports the leakage that protocol WOULD produce. That is the
counterfactual the survey argument needs, and it is what makes datasets
without an official partition usable as evidence.

Class filtering is first-class, not an afterthought. VN-A looks redundant at
10.9% overall and is 0.7% across the four classes actually mapped, because
123 of its 141 redundant images are in `Leaf_Healthy`. A survey that does not
record which classes were counted is not comparable across datasets, so
`--only-classes` is stored in the output row.

Usage
-----
    pip install pillow numpy pandas

    # one dataset, shipped split, restricted to the mapped classes
    python leakage_survey.py \
        --dataset "VN-A=/path/Durian_Leaf_Diseases" \
        --only-classes "VN-A=Leaf_Algal,Leaf_Colletotrichum,Leaf_Blight,Leaf_Phomopsis,Leaf_Rhizoctonia"

    # several at once, including ones with no shipped split
    python leakage_survey.py \
        --dataset "MY-durian=/path/Classication_model_split_512" \
        --dataset "VN-A=/path/Durian_Leaf_Diseases" \
        --dataset "VN-B=/path/Ten_Classes_of_Durian_Leaf_Diseases" \
        --dataset "PlantVillage=/path/plantvillage" \
        --out survey

    # sensitivity of every row to the hash threshold
    python leakage_survey.py --dataset "VN-A=/path/..." --sweep

Outputs (under --out, default `survey/`)
    survey.csv                 one row per dataset, the table
    survey.md                  the same, formatted for the supplement
    <name>_sessions.csv        per-dataset manifest, columns match
                               session_split.py, so session_utils.load_sessions
                               and StratifiedGroupKFold take it unchanged
    .cache/<name>.npz          hashes, so re-runs and sweeps are cheap

Caveats this script prints rather than hides
    * Clusters are inferred by perceptual hash, not read from capture
      metadata. They are a proxy. Report the threshold and the sweep.
    * Single-linkage chains: a gradual sequence can merge unrelated images.
      `--show` prints the largest clusters so you can open a few and look.
    * A dataset that is uniformly preprocessed (background-removed, resized)
      hashes differently from raw field imagery. Cross-dataset comparison of
      the ABSOLUTE cluster count is weak; comparison of the split-leakage
      columns is what the argument rests on.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
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

# Split folder names seen in the wild, lowercased.
TRAIN_NAMES = {"train", "training", "train_set"}
VAL_NAMES = {"val", "valid", "validation", "dev"}
TEST_NAMES = {"test", "testing", "test_set", "eval"}
SPLIT_NAMES = TRAIN_NAMES | VAL_NAMES | TEST_NAMES

POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.uint8)


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------

def dhash(img, size=8):
    """64-bit difference hash, packed to 8 bytes. Same definition as
    phash_sessions.py, so manifests from the two agree."""
    small = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = np.asarray(small, dtype=np.int16)
    bits = px[:, :-1] > px[:, 1:]
    return np.packbits(bits.flatten())


def pixel_hash_bytes(img):
    """Hash of decoded pixels at 64x64. Catches re-encoding, which MD5 misses."""
    a = np.asarray(img.convert("RGB").resize((64, 64), Image.LANCZOS))
    return hashlib.md5(a.tobytes()).hexdigest()


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# layout discovery
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


def scan_dataset(root):
    """Return (records, has_shipped_split).

    Handles root/<split>/<class>/*.img and root/<class>/*.img. A split folder
    is recognised by name, not by position, so Train/Test/Validation and
    train/val/test both work and a class called 'test_leaf' does not.
    """
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    top = [d for d in sorted(root.iterdir()) if d.is_dir()]
    split_dirs = [d for d in top if normalise_split(d.name)]
    recs = []

    if split_dirs and len(split_dirs) >= 2:
        for d in split_dirs:
            sp = normalise_split(d.name)
            for cls_dir in sorted(d.iterdir()):
                if not cls_dir.is_dir():
                    continue
                for f in sorted(cls_dir.rglob("*")):
                    if f.is_file() and f.suffix.lower() in IMG_EXT:
                        recs.append(dict(split=sp, cls=cls_dir.name.strip(),
                                         file=f.name, path=str(f)))
        return recs, True

    for cls_dir in top:
        for f in sorted(cls_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMG_EXT:
                recs.append(dict(split="__none__", cls=cls_dir.name.strip(),
                                 file=f.name, path=str(f)))
    return recs, False


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


def load_embeddings(npz_path, wanted_paths):
    """Load an .npz from extract_features.py and select the rows we need.

    The npz is extracted over a whole tree; a class filter here reduces the
    record list, so rows are selected by path rather than assumed aligned.
    A path present in the scan but absent from the npz is fatal: silently
    dropping it would change which images the survey covers.
    """
    z = np.load(npz_path, allow_pickle=True)
    E = np.asarray(z["E"], dtype=np.float32)
    index = {p: i for i, p in enumerate(list(z["paths"]))}
    missing = [p for p in wanted_paths if p not in index]
    if missing:
        raise SystemExit(
            f"{npz_path}: {len(missing)} scanned images are absent from the "
            f"embedding file, e.g. {missing[0]}. Re-extract against this tree.")
    rows = np.array([index[p] for p in wanted_paths])
    E = E[rows]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    tag = str(z["weights"]) if "weights" in z else Path(npz_path).name
    return E, tag


def cluster_cosine(E, classes, tau):
    """Single-linkage on cosine similarity, within class.

    Matches embed_proxy.py exactly, so a threshold calibrated there means the
    same thing here.
    """
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
        roots = {}
        for k in range(m):
            r = uf.find(k)
            roots.setdefault(r, len(roots))
            session[idxs[k]] = f"{cls}:e{roots[r]}"
    return session


def is_degenerate(sessions, classes):
    """Structural collapse check, usable without ground truth.

    embed_proxy.py can flag an over-merged clustering because it has true
    sessions to compare against. Here there are none, so the check is
    structural: a grouping whose clusters are enormous, or one of whose
    clusters swallows a quarter of its class, reports high leakage by
    merging rather than by detecting. Such a row must not be quoted.
    """
    df = pd.DataFrame(dict(cls=classes, s=list(sessions)))
    n = len(df)
    n_clusters = df["s"].nunique()
    if n_clusters == 0:
        return True, "no clusters"
    if n / n_clusters > 10:
        return True, f"mean cluster size {n / n_clusters:.1f}"
    for cls, g in df.groupby("cls"):
        largest = g["s"].value_counts().iloc[0]
        if largest > 0.25 * len(g):
            return True, f"largest cluster is {100 * largest / len(g):.0f}% of {cls}"
    return False, ""


def cluster_within_class(hashes, classes, threshold, block=256):
    """Single-linkage on dHash within each class.

    Blocked so a class with tens of thousands of images does not allocate an
    n x n matrix. Memory is block x n x 8 bytes, ~20 MB at the default for a
    class of 10,000.
    """
    H = np.stack(hashes)                       # (n, 8) uint8
    session = np.empty(len(H), dtype=object)
    by_cls = defaultdict(list)
    for i, c in enumerate(classes):
        by_cls[c].append(i)

    for cls, idxs in by_cls.items():
        idxs = np.asarray(idxs)
        m = len(idxs)
        if m == 1:
            session[idxs[0]] = f"{cls}:s0"
            continue
        sub = H[idxs]
        uf = Union(m)
        for start in range(0, m, block):
            stop = min(start + block, m)
            xor = np.bitwise_xor(sub[start:stop, None, :], sub[None, :, :])
            dist = POPCOUNT[xor].sum(-1)       # (block, m)
            # upper triangle only, relative to the block offset
            rows, cols = np.where(dist <= threshold)
            for r, c in zip(rows, cols):
                a = start + int(r)
                b = int(c)
                if b > a:
                    uf.union(a, b)
        roots = {}
        for k in range(m):
            r = uf.find(k)
            roots.setdefault(r, len(roots))
            session[idxs[k]] = f"{cls}:s{roots[r]}"
    return session


# ---------------------------------------------------------------------------
# leakage metrics
# ---------------------------------------------------------------------------

def sibling_leakage(split_labels, sessions):
    """Fraction of test images whose cluster also appears in train.

    This is the 96.7% quantity, not the 79.6% one. It is what bears on a
    reported test score: an image whose specimen the model trained on is not
    a held-out observation whatever the partition claims.
    """
    split_labels = np.asarray(split_labels)
    sessions = np.asarray(sessions)
    train_sessions = set(sessions[split_labels == "train"])
    is_test = split_labels == "test"
    n_test = int(is_test.sum())
    if n_test == 0:
        return None, 0, 0
    hit = int(sum(1 for s in sessions[is_test] if s in train_sessions))
    shared = len(set(sessions[is_test]) & train_sessions)
    return 100.0 * hit / n_test, hit, shared


def straddle_stats(split_labels, sessions):
    """Clusters appearing in more than one split, and images in them.

    The looser measure. Kept because it is the one the manuscript reports as
    79.6% and a reader will look for it, but it is not the headline.
    """
    by_session = defaultdict(list)
    for sp, s in zip(split_labels, sessions):
        by_session[s].append(sp)
    straddling = [s for s, sps in by_session.items() if len(set(sps)) > 1]
    n_imgs = sum(len(by_session[s]) for s in straddling)
    return len(straddling), n_imgs, 100.0 * n_imgs / len(sessions)


def simulate_image_split(labels, sessions, seeds, ratios=(0.8, 0.1, 0.1)):
    """Stratified image-level split, the prevailing protocol, at several seeds.

    Returns mean and s.d. of the sibling-leakage percentage. This is the
    counterfactual for datasets shipping no partition of their own: not what
    their authors did, but what the protocol in general use would produce on
    these images.
    """
    labels = np.asarray(labels)
    out = []
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
        pct, _, _ = sibling_leakage(assign, sessions)
        if pct is not None:
            out.append(pct)
    if not out:
        return None, None
    return float(np.mean(out)), float(np.std(out))


def classify_redundancy(df, sessions):
    """Split redundancy into COPY and BURST, following classify_redundancy.py.

    Only images in multi-image clusters are byte- and pixel-hashed, which is
    what keeps this affordable on a large dataset.
    """
    df = df.copy()
    df["session"] = sessions
    sizes = df.groupby("session").size()
    multi_ids = set(sizes[sizes > 1].index)
    multi = df[df.session.isin(multi_ids)]

    if multi.empty:
        return dict(images_in_clusters=0, byte_copies=0, reencoded=0,
                    burst_distinct=0, copy_share=None)

    md5s, pxs = [], []
    for p in multi["path"]:
        try:
            md5s.append(md5_of(p))
            with Image.open(p) as im:
                im.load()
                pxs.append(pixel_hash_bytes(im))
        except Exception:
            md5s.append(None)
            pxs.append(None)
    multi = multi.assign(md5=md5s, px=pxs)

    byte_copies = reencoded = burst = 0
    for _, g in multi.groupby("session"):
        n = len(g)
        b = n - g["md5"].nunique(dropna=False)
        px = n - g["px"].nunique(dropna=False)
        px_only = max(px - b, 0)
        byte_copies += b
        reencoded += px_only
        burst += max(n - 1 - b - px_only, 0)

    redundant = byte_copies + reencoded + burst
    copy_share = (byte_copies + reencoded) / redundant if redundant else None
    return dict(images_in_clusters=len(multi), byte_copies=byte_copies,
                reencoded=reencoded, burst_distinct=burst,
                copy_share=copy_share)


def redundancy_kind(copy_share):
    """The gate that decides whether a dataset may enter the survey column.

    VN-B is 75.8% copies. Averaging it with a burst-dominated dataset would
    compare two quantities that are not the same quantity, which is the error
    this whole paper is about, committed one level up.
    """
    if copy_share is None:
        return "none"
    if copy_share > 0.7:
        return "copy-dominated"
    if copy_share < 0.3:
        return "burst-dominated"
    return "mixed"


# ---------------------------------------------------------------------------
# per-dataset driver
# ---------------------------------------------------------------------------

def hash_dataset(name, recs, cache_dir, rehash=False):
    """dHash every image, cached, so sweeps and re-runs are cheap."""
    cache = Path(cache_dir) / f"{name}.npz"
    paths = [r["path"] for r in recs]
    if cache.is_file() and not rehash:
        z = np.load(cache, allow_pickle=True)
        if list(z["paths"]) == paths:
            print(f"  hashes from cache ({len(paths)} images)")
            return z["hashes"], list(z["ok"])
    print(f"  hashing {len(paths)} images...")
    hashes, ok = [], []
    for i, p in enumerate(paths):
        if i and i % 2000 == 0:
            print(f"    {i}/{len(paths)}")
        try:
            with Image.open(p) as im:
                im.load()
                hashes.append(dhash(im))
            ok.append(True)
        except Exception as exc:
            print(f"    unreadable, skipped: {p}  {exc!r}")
            hashes.append(np.zeros(8, dtype=np.uint8))
            ok.append(False)
    H = np.stack(hashes)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, hashes=H, paths=np.array(paths, dtype=object),
                        ok=np.array(ok))
    return H, ok


def audit_one(name, root, threshold, only_classes, sim_seeds, cache_dir,
              out_dir, show, rehash, emb_path=None, tau=0.85):
    print(f"\n{'=' * 72}\n{name}  <-  {root}\n{'=' * 72}")
    recs, shipped = scan_dataset(root)
    if not recs:
        print("  no images found, skipped")
        return None
    print(f"  layout : {'shipped split' if shipped else 'no split folders'}")

    if only_classes:
        keep = set(only_classes)
        present = {r["cls"] for r in recs}
        missing = keep - present
        if missing:
            print(f"  !! requested classes absent: {sorted(missing)}")
        recs = [r for r in recs if r["cls"] in keep]
        if not recs:
            print("  no images left after class filter, skipped")
            return None
        print(f"  classes: restricted to {len(keep)} of {len(present)}")

    if emb_path:
        E, tag = load_embeddings(emb_path, [r["path"] for r in recs])
        df = pd.DataFrame(recs)
        print(f"  images : {len(df)} in {df.cls.nunique()} classes")
        print(f"  method : cosine >= {tau} on {tag}")
        sessions = cluster_cosine(E, df["cls"].tolist(), tau)
        method, param = f"embed[{tag}]", tau
    else:
        H, ok = hash_dataset(name, recs, cache_dir, rehash=rehash)
        recs = [r for r, good in zip(recs, ok) if good]
        H = H[np.array(ok, dtype=bool)]
        df = pd.DataFrame(recs)
        print(f"  images : {len(df)} in {df.cls.nunique()} classes")
        print(f"  method : dHash <= {threshold}")
        sessions = cluster_within_class(list(H), df["cls"].tolist(), threshold)
        method, param = "dhash", threshold
    df = df.assign(session=sessions)
    sizes = df.groupby("session").size()
    n_clusters = len(sizes)
    singletons = int((sizes == 1).sum())

    degenerate, why = is_degenerate(sessions, df["cls"].tolist())
    row = dict(
        dataset=name,
        root=str(root),
        method=method,
        param=param,
        images=len(df),
        classes=df.cls.nunique(),
        classes_counted=",".join(sorted(df.cls.unique())) if only_classes else "all",
        threshold=threshold,
        clusters=n_clusters,
        images_per_cluster=round(len(df) / n_clusters, 2),
        singleton_pct=round(100 * singletons / n_clusters, 1),
        largest_cluster=int(sizes.max()),
        shipped_split=shipped,
        degenerate=degenerate,
        degenerate_reason=why,
    )

    red = classify_redundancy(df[["path", "cls"]], df["session"].values)
    redundant = red["byte_copies"] + red["reencoded"] + red["burst_distinct"]
    row.update(
        redundant_images=redundant,
        redundant_pct=round(100 * redundant / len(df), 1),
        byte_copies=red["byte_copies"],
        reencoded=red["reencoded"],
        burst_distinct=red["burst_distinct"],
        burst_pct=round(100 * red["burst_distinct"] / len(df), 2),
        copy_share_pct=None if red["copy_share"] is None else round(100 * red["copy_share"], 1),
        redundancy_kind=redundancy_kind(red["copy_share"]),
    )

    if shipped:
        pct, hit, shared = sibling_leakage(df["split"].values, df["session"].values)
        n_str, n_str_imgs, str_pct = straddle_stats(df["split"].values, df["session"].values)
        row.update(
            shipped_test_leak_pct=None if pct is None else round(pct, 1),
            shipped_test_leaked_images=hit,
            shipped_shared_clusters=shared,
            shipped_straddling_clusters=n_str,
            shipped_straddle_pct=round(str_pct, 1),
        )
    else:
        row.update(shipped_test_leak_pct=None, shipped_test_leaked_images=None,
                   shipped_shared_clusters=None, shipped_straddling_clusters=None,
                   shipped_straddle_pct=None)

    mean, sd = simulate_image_split(df["cls"].values, df["session"].values, sim_seeds)
    row.update(
        sim_test_leak_pct=None if mean is None else round(mean, 1),
        sim_test_leak_sd=None if sd is None else round(sd, 1),
        sim_seeds=len(sim_seeds),
    )

    print(f"  clusters               {n_clusters}  "
          f"({row['images_per_cluster']} images each, "
          f"{row['singleton_pct']}% singletons, largest {row['largest_cluster']})")
    print(f"  redundant images       {redundant} ({row['redundant_pct']}%)  "
          f"-> {row['redundancy_kind']}"
          + (f", copies {row['copy_share_pct']}% of redundancy"
             if row["copy_share_pct"] is not None else ""))
    if shipped:
        print(f"  SHIPPED split: {row['shipped_test_leak_pct']}% of test images "
              f"have a same-cluster sibling in train "
              f"({row['shipped_straddle_pct']}% of images in straddling clusters)")
    print(f"  SIMULATED image-level split: {row['sim_test_leak_pct']}% "
          f"± {row['sim_test_leak_sd']} over {len(sim_seeds)} seeds")

    if degenerate:
        print(f"  !! DEGENERATE ({why}). This grouping reports high leakage by")
        print("     merging rather than by detecting. Do not quote this row;")
        print("     tighten the threshold.")
    if row["redundancy_kind"] == "copy-dominated":
        print("  !! Copy-dominated. This is a dataset-assembly fault, not")
        print("     collection-process redundancy. Footnote it or exclude it;")
        print("     do not average it with burst-dominated datasets.")
    if row["singleton_pct"] > 98:
        print("  !! Almost every cluster is a singleton. Either there is no")
        print("     redundancy or the threshold is too tight. Run --sweep.")
    if row["largest_cluster"] > 0.4 * len(df) / max(df.cls.nunique(), 1):
        print("  !! One cluster holds a large share of its class; single-linkage")
        print("     chains. Inspect it before trusting this row.")

    if show:
        print(f"\n  largest {show} clusters:")
        for s, g in sorted(df.groupby("session"), key=lambda kv: -len(kv[1]))[:show]:
            spl = Counter(g["split"])
            print(f"    {s:<34} n={len(g):<4} splits={dict(spl)}")
            for f in list(g["file"])[:4]:
                print(f"        {f}")
            if len(g) > 4:
                print(f"        ... and {len(g) - 4} more")

    man = Path(out_dir) / f"{name}_sessions.csv"
    df[["cls", "file", "split", "session", "path"]].to_csv(man, index=False)
    print(f"\n  manifest -> {man}")
    return row


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def sweep_one(name, root, only_classes, sim_seeds, cache_dir, thresholds, rehash,
              emb_path=None):
    print(f"\n{'=' * 72}\n{name} — threshold sweep\n{'=' * 72}")
    recs, shipped = scan_dataset(root)
    if only_classes:
        recs = [r for r in recs if r["cls"] in set(only_classes)]
    if not recs:
        print("  nothing to sweep")
        return []
    if emb_path:
        E, tag = load_embeddings(emb_path, [r["path"] for r in recs])
        H = None
        df = pd.DataFrame(recs)
        print(f"  cosine sweep on {tag}")
    else:
        H, ok = hash_dataset(name, recs, cache_dir, rehash=rehash)
        recs = [r for r, good in zip(recs, ok) if good]
        H = H[np.array(ok, dtype=bool)]
        df = pd.DataFrame(recs)

    rows = []
    for t in thresholds:
        sessions = (cluster_cosine(E, df["cls"].tolist(), float(t)) if emb_path
                    else cluster_within_class(list(H), df["cls"].tolist(), int(t)))
        degen, why = is_degenerate(sessions, df["cls"].tolist())
        sizes = pd.Series(sessions).value_counts()
        shipped_pct = None
        if shipped:
            shipped_pct, _, _ = sibling_leakage(df["split"].values, sessions)
        sim_mean, _ = simulate_image_split(df["cls"].values, sessions, sim_seeds)
        rows.append(dict(dataset=name, threshold=t, clusters=len(sizes),
                         degenerate=degen,
                         singleton_pct=round(100 * (sizes == 1).sum() / len(sizes), 1),
                         largest=int(sizes.max()),
                         shipped_test_leak_pct=None if shipped_pct is None else round(shipped_pct, 1),
                         sim_test_leak_pct=None if sim_mean is None else round(sim_mean, 1)))
        r = rows[-1]
        print(f"  t={t:>2}  clusters {r['clusters']:>6}  "
              f"singletons {r['singleton_pct']:>5}%  largest {r['largest']:>5}  "
              f"shipped-leak {r['shipped_test_leak_pct']}  "
              f"sim-leak {r['sim_test_leak_pct']}"
              + ("   [DEGENERATE]" if r["degenerate"] else ""))

    vals = [r["sim_test_leak_pct"] for r in rows
            if r["sim_test_leak_pct"] is not None and not r["degenerate"]]
    if vals:
        spread = max(vals) - min(vals)
        print(f"\n  simulated leakage across the sweep: "
              f"{min(vals):.1f}% to {max(vals):.1f}% (spread {spread:.1f} pp)")
        if spread > 15:
            print("  Wide. The conclusion depends on the threshold and the")
            print("  paper must report the range, not a point estimate.")
        else:
            print("  Stable. Report the point estimate with the range in a footnote.")
    return rows


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

MD_COLUMNS = [
    ("dataset", "Dataset"),
    ("method", "Grouping method"),
    ("param", "Threshold"),
    ("images", "Images"),
    ("clusters", "Clusters"),
    ("images_per_cluster", "Img/cluster"),
    ("burst_pct", "Burst redundancy (%)"),
    ("copy_share_pct", "Copies as % of redundancy"),
    ("redundancy_kind", "Kind"),
    ("shipped_test_leak_pct", "Shipped split: test leak (%)"),
    ("sim_test_leak_pct", "Image-level split: test leak (%)"),
    ("degenerate", "Degenerate"),
]


def write_markdown(rows, path):
    heads = [h for _, h in MD_COLUMNS]
    lines = ["| " + " | ".join(heads) + " |",
             "|" + "|".join(["---"] * len(heads)) + "|"]
    for r in rows:
        cells = []
        for key, _ in MD_COLUMNS:
            v = r.get(key)
            cells.append("—" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "*Clusters are inferred by perceptual hash (dHash, single-linkage within "
        "class) rather than read from capture metadata, and are a proxy for the "
        "capture session. Burst redundancy counts near-identical images that are "
        "genuinely distinct in bytes and pixels; copies are byte-identical or "
        "re-encoded duplicates and are a dataset-assembly fault rather than a "
        "property of the collection process. Test leak is the percentage of test "
        "images having a same-cluster sibling in train. The image-level column is "
        "simulated: it reports what the stratified random split prevailing in this "
        "literature would produce on these images, and is defined for datasets "
        "shipping no partition of their own.*",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_kv(items, what):
    out = {}
    for s in items or []:
        if "=" not in s:
            raise SystemExit(f"--{what} needs NAME=VALUE, got: {s}")
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", required=True,
                    metavar="NAME=PATH", help="repeatable")
    ap.add_argument("--only-classes", action="append", metavar="NAME=a,b,c",
                    help="restrict one dataset to these classes; repeatable")
    ap.add_argument("--threshold", type=int, default=6,
                    help="dHash Hamming distance for the same cluster (default 6)")
    ap.add_argument("--embeddings", action="append", metavar="NAME=FILE.npz",
                    help="use cosine clustering on these features instead of "
                         "dHash for this dataset; repeatable")
    ap.add_argument("--sim-threshold", type=float, default=0.85,
                    help="cosine threshold when --embeddings is given. Use the "
                         "value calibrated by embed_proxy.py on the dataset "
                         "where ground truth exists (default 0.85)")
    ap.add_argument("--sweep-sims", default="0.75,0.80,0.85,0.88,0.90,0.95",
                    help="cosine thresholds for --sweep with --embeddings")
    ap.add_argument("--sweep", action="store_true",
                    help="report across thresholds instead of writing the survey")
    ap.add_argument("--sweep-thresholds", default="2,4,6,8,10,12")
    ap.add_argument("--sim-seeds", default="0,1,2,3,4",
                    help="seeds for the simulated image-level split")
    ap.add_argument("--out", default="survey")
    ap.add_argument("--show", type=int, default=3,
                    help="print this many of the largest clusters per dataset")
    ap.add_argument("--rehash", action="store_true", help="ignore the hash cache")
    args = ap.parse_args()

    datasets = parse_kv(args.dataset, "dataset")
    classes = {k: [c.strip() for c in v.split(",") if c.strip()]
               for k, v in parse_kv(args.only_classes, "only-classes").items()}
    unknown = set(classes) - set(datasets)
    if unknown:
        raise SystemExit(f"--only-classes names no such dataset: {sorted(unknown)}")

    embeddings = parse_kv(args.embeddings, "embeddings")
    unknown_e = set(embeddings) - set(datasets)
    if unknown_e:
        raise SystemExit(f"--embeddings names no such dataset: {sorted(unknown_e)}")
    if embeddings and set(embeddings) != set(datasets):
        print("!! Only some datasets have embeddings. Rows produced by "
              "different grouping methods are not comparable; the method "
              "column records which is which, but do not read them as one "
              "column of the same quantity.\n")

    sim_seeds = [int(s) for s in args.sim_seeds.split(",") if s.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(exist_ok=True)

    if args.sweep:
        allrows = []
        for name, root in datasets.items():
            emb = embeddings.get(name)
            ths = ([float(t) for t in args.sweep_sims.split(",")] if emb
                   else [int(t) for t in args.sweep_thresholds.split(",")])
            allrows += sweep_one(name, root, classes.get(name), sim_seeds,
                                 cache_dir, ths, args.rehash, emb_path=emb)
        if allrows:
            p = out_dir / "sweep.csv"
            pd.DataFrame(allrows).to_csv(p, index=False)
            print(f"\nsweep -> {p}")
        return

    rows = []
    for name, root in datasets.items():
        r = audit_one(name, root, args.threshold, classes.get(name), sim_seeds,
                      cache_dir, out_dir, args.show, args.rehash,
                      emb_path=embeddings.get(name), tau=args.sim_threshold)
        if r:
            rows.append(r)

    if not rows:
        raise SystemExit("No dataset produced a row.")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "survey.csv"
    md_path = out_dir / "survey.md"
    df.to_csv(csv_path, index=False)
    write_markdown(rows, md_path)

    print(f"\n{'=' * 72}\nSURVEY\n{'=' * 72}")
    print(df[[c for c, _ in MD_COLUMNS if c in df.columns]].to_string(index=False))
    print(f"\ntable -> {csv_path}\n         {md_path}")

    if df["degenerate"].any():
        bad = ", ".join(df[df.degenerate].dataset)
        print(f"\n!! Degenerate rows: {bad}. Tighten the threshold before "
              f"quoting them.")
    usable = df[df.redundancy_kind.isin(["burst-dominated", "mixed"])
                & ~df.degenerate]
    print(f"\n{len(usable)} of {len(df)} datasets are burst-dominated or mixed, "
          f"non-degenerate, and belong in the same column.")
    if len(usable) < 5:
        print("P1_REMAINING.md sets the bar at five or six comparable datasets.")
        print("Below that, report the rows individually and do not fit a")
        print("relationship across them.")


if __name__ == "__main__":
    main()
