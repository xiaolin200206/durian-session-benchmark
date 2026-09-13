#!/usr/bin/env python3
"""
grouping_sensitivity.py — does the leakage measurement survive the grouping rule?
================================================================================
The capture sessions in this dataset are reconstructed from filename structure,
not recorded at collection. That makes every figure derived from them
conditional on a heuristic, and an earlier version of the paper defended the
heuristic by arguing that its errors were conservative. They are not: merging
two specimens into one group inflates a leakage estimate, splitting one
specimen across two groups deflates it, and no argument from the rule's
construction settles which dominates.

What can be settled is how much the measurement moves when the rule moves.
This script varies the reconstruction in both directions and reports the
leakage measurement under each:

    conservative split   strictly consecutive camera counters (gap 1), within
                         class — the finest defensible grouping
    as published         gap 3, within class
    cross-class          gap 3, ignoring the disease label, so one continuous
                         camera run is one group even where its frames were
                         filed under different diseases
    conservative merge   gap 10, ignoring the disease label — the coarsest

Leakage is measured over repeated simulated stratified image-level partitions:
the share of test images whose group also appears in training. If that share
is stable across a large change in the number of groups, it is a property of
the data; if it is not, the paper must report the range rather than a point.

Inputs are `sessions.csv` alone (columns cls, file, session) — the images are
not needed, because the rule operates on filenames.

    python grouping_sensitivity.py --sessions sessions.csv --out sensitivity
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Matches the sequence-number convention of session_split.py.
RE_SEQ = re.compile(r"^(?P<prefix>[A-Za-z_]*?)(?P<num>\d{3,6})\b")

RULES = [
    (1, False, "conservative split (gap 1, within class)"),
    (3, False, "as published (gap 3, within class)"),
    (3, True, "cross-class protective (gap 3)"),
    (10, False, "wider burst (gap 10, within class)"),
    (10, True, "conservative merge (gap 10, cross-class)"),
]


def load(path):
    d = pd.read_csv(path)
    need = {"cls", "file", "session"}
    if not need.issubset(d.columns):
        raise SystemExit(f"{path} needs columns {sorted(need)}")
    d["kind"] = d["session"].astype(str).str.split(":").str[0]

    def parse(f):
        m = RE_SEQ.match(str(f))
        return ((m.group("prefix") or "IMG"), int(m.group("num"))) if m else (None, None)

    d[["prefix", "num"]] = d["file"].apply(lambda f: pd.Series(parse(f)))
    return d


def build(d, gap, cross_class):
    """Rebuild the grouping. Records that were not recovered from a sequence
    (video-derived frames, messaging batches, unmatched singles) keep whatever
    session they were assigned, because the rule under test is the burst rule."""
    seq = d["kind"].isin(["burst", "single"]) & d["num"].notna()
    lab = {}
    key = ["prefix"] if cross_class else ["cls", "prefix"]
    for k, grp in d[seq].groupby(key):
        run, prev = 0, None
        for i, r in grp.sort_values("num").iterrows():
            if prev is not None and r["num"] - prev > gap:
                run += 1
            lab[i] = f"seq:{k}:{run}"
            prev = r["num"]
    for i, r in d[~seq].iterrows():
        lab[i] = r["session"]
    return pd.Series(lab).reindex(d.index)


def leak(classes, groups, seeds, ratios=(0.8, 0.1, 0.1)):
    classes = np.asarray(classes)
    g = np.asarray(groups)
    out = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        a = np.empty(len(classes), dtype=object)
        for c in np.unique(classes):
            idx = np.where(classes == c)[0]
            rng.shuffle(idx)
            n = len(idx)
            ntr = int(round(n * ratios[0]))
            nva = int(round(n * ratios[1]))
            a[idx[:ntr]] = "train"
            a[idx[ntr:ntr + nva]] = "val"
            a[idx[ntr + nva:]] = "test"
        tr = set(g[a == "train"])
        te = a == "test"
        out.append(100 * sum(1 for x in g[te] if x in tr) / te.sum())
    return float(np.mean(out)), float(np.std(out, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="sensitivity")
    args = ap.parse_args()

    d = load(args.sessions)
    seeds = list(range(args.seeds))
    published = build(d, 3, False)
    n_pub = published.nunique()
    print(f"{len(d)} images | published rule reproduces {n_pub} groups "
          f"(the paper states 73; a mismatch means the rule here has drifted "
          f"from session_split.py and the rest of this table is not comparable)\n")

    rows = []
    for gap, cc, name in RULES:
        g = build(d, gap, cc)
        m, sd = leak(d["cls"].values, g.values, seeds)
        sizes = g.value_counts()
        rows.append(dict(rule=name, gap=gap, cross_class=cc, groups=g.nunique(),
                         images_per_group=round(len(d) / g.nunique(), 2),
                         largest_group=int(sizes.max()),
                         leak_pct=round(m, 1), leak_sd=round(sd, 1)))
        print(f"  {name:<42}{g.nunique():>6} groups  largest {sizes.max():>4}  "
              f"leak {m:5.1f}% ± {sd:.1f}")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "grouping_sensitivity.csv", index=False)

    lo, hi = df.leak_pct.min(), df.leak_pct.max()
    fold = df.groups.max() / df.groups.min()
    print(f"\nAcross a {fold:.1f}-fold change in the number of groups the leakage "
          f"measurement moves from {lo}% to {hi}% ({hi - lo:.1f} pp).")
    if hi - lo <= 10:
        print("Stable. Report the published-rule value with this range stated.")
    else:
        print("Not stable. The paper must report the range, not a point estimate,")
        print("and should say which rule each headline figure came from.")

    # how much does ignoring the class label change the grouping?
    cc = build(d, 3, True)
    joined = published.groupby(cc).nunique()
    n_join = int((joined > 1).sum())
    big = pd.DataFrame({"g": cc, "cls": d["cls"]}).groupby("g")["cls"].agg(["size", "nunique"])
    worst = big.sort_values("size", ascending=False).head(1)
    print(f"\nIgnoring the disease label when merging joins {n_join} groups of "
          f"published sessions; {n_pub} sessions become {cc.nunique()}. The largest "
          f"cross-class group holds {int(worst['size'].iloc[0])} images spanning "
          f"{int(worst['nunique'].iloc[0])} classes.")
    print(f"\ntable -> {out / 'grouping_sensitivity.csv'}")


if __name__ == "__main__":
    main()
