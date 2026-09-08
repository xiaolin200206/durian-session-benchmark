#!/usr/bin/env python3
"""
aggregate.py — merge per-seed results into mean +/- s.d. tables
===============================================================
run_all.sh writes one results folder per (regime, seed). This collects them,
reports mean and standard deviation over seeds, and puts the session-level
and image-level regimes side by side.

The paired table is the point: the gap between the two columns is how much
the image-level split inflated the reported score, and it is the finding
this rerun exists to produce.

    python aggregate.py --root /root/autodl-tmp/results --out /root/autodl-tmp/summary
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

DIRPAT = re.compile(r"^(?P<mode>group|image)_s(?P<seed>-?\d+)$")

# csv name -> (key columns that identify a row, metric columns to average)
TABLES = {
    "ablation_results.csv":    ["Model"],
    "comparison_results.csv":  ["Model"],
    "cv_results.csv":          ["Fold"],
    "vietnam_validation.csv":  ["Model"],
    "robustness_results.csv":  ["Model", "Condition"],
    "per_class_metrics.csv":   ["Class"],
}

NON_METRIC = {"seed", "cv_mode", "Fold", "Model", "Class", "Condition"}


def collect(root):
    rows = []
    for d in sorted(Path(root).iterdir()):
        m = DIRPAT.match(d.name) if d.is_dir() else None
        if not m:
            continue
        if not (d / "DONE").exists():
            print(f"  [incomplete] {d.name} — no DONE marker, skipping")
            continue
        rows.append((m.group("mode"), int(m.group("seed")), d))
    return rows


def load(runs, fname, keys):
    frames = []
    for mode, seed, d in runs:
        f = d / fname
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
        except Exception as exc:
            print(f"  [unreadable] {f}: {exc}")
            continue
        df["cv_mode"] = mode
        df["seed"] = seed
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def summarise(df, keys, out_dir, stem):
    present = [k for k in keys if k in df.columns]
    if not present:
        print(f"  [skip] {stem}: none of {keys} present")
        return None

    metrics = [c for c in df.columns
               if c not in NON_METRIC and pd.api.types.is_numeric_dtype(df[c])]
    if not metrics:
        print(f"  [skip] {stem}: no numeric columns")
        return None

    g = df.groupby(["cv_mode"] + present)[metrics]
    agg = g.agg(["mean", "std", "count"]).round(2)
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_dir / f"{stem}_by_seed.csv", index=False)

    # paired view on the headline metric
    metric = next((m for m in metrics if "Macro F1" in m or "F1" in m), metrics[0])
    piv = agg.pivot_table(index=present, columns="cv_mode",
                          values=[f"{metric}_mean", f"{metric}_std"])
    piv.columns = [f"{a.replace('_mean','').replace('_std','')}"
                   f"{'_sd' if '_std' in a else ''}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()

    gcol = next((c for c in piv.columns if c.endswith("_group") and "_sd" not in c), None)
    icol = next((c for c in piv.columns if c.endswith("_image") and "_sd" not in c), None)
    if gcol and icol:
        piv["inflation_pp"] = (piv[icol] - piv[gcol]).round(2)
    piv.to_csv(out_dir / f"{stem}_paired.csv", index=False)
    return piv, metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    if not root.is_dir():
        sys.exit(f"Not found: {root}")
    out.mkdir(parents=True, exist_ok=True)

    runs = collect(root)
    if not runs:
        sys.exit(f"No completed run folders under {root}")

    modes = sorted({m for m, _, _ in runs})
    print(f"Found {len(runs)} completed runs: "
          + ", ".join(f"{m} x{sum(1 for a,_,_ in runs if a==m)}" for m in modes))
    if "image" not in modes:
        print("\n!! No image-level control runs found. The leakage comparison "
              "needs both regimes.")

    for fname, keys in TABLES.items():
        df = load(runs, fname, keys)
        if df is None:
            continue
        stem = fname.replace(".csv", "")
        print(f"\n{'='*66}\n{stem}\n{'='*66}")
        res = summarise(df, keys, out, stem)
        if res is None:
            continue
        piv, metric = res
        with pd.option_context("display.width", 200,
                               "display.max_columns", 30,
                               "display.max_rows", 60):
            print(piv.to_string(index=False))
        if "inflation_pp" in piv.columns:
            v = piv["inflation_pp"].dropna()
            if len(v):
                print(f"\n  {metric}: image-level split is higher by "
                      f"{v.mean():+.2f} pp on average "
                      f"(range {v.min():+.2f} to {v.max():+.2f})")

    print(f"\nCSVs written to {out.resolve()}")
    print("For the paper: report the session-level column as the result and "
          "the gap as the measurement.")


if __name__ == "__main__":
    main()
