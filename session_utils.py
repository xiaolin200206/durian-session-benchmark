"""
session_utils.py
================
Drop-in replacement for the image-level splitting and cross-validation in
train.py.  Put this file next to train.py and apply the edits in PATCH.md.

Why this exists
---------------
Field images are captured in sessions: a burst of shots of one lesion, a
video panned along one branch, one WhatsApp batch from one grower.  The
independent unit of observation is the SESSION, not the image.  Splitting
at image level puts near-identical frames on both sides of the train/test
boundary, which inflates every reported metric.

Requires: scikit-learn >= 0.24 (for StratifiedGroupKFold), pandas.
"""

import csv
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_SGKF = True
except ImportError:                                      # sklearn < 0.24
    from sklearn.model_selection import GroupKFold
    _HAS_SGKF = False

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


# --------------------------------------------------------------------------
# 1. session lookup
# --------------------------------------------------------------------------

def load_sessions(csv_path):
    """Read sessions.csv produced by session_split.py.

    Returns {(class_name, file_name): session_id}.  Keyed on the pair rather
    than on the path, because do_split copies files to new locations, and
    keyed on the pair rather than the bare filename, because the same
    filename can legitimately appear under two different classes.
    """
    table = {}
    with open(csv_path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            table[(row['cls'], row['file'])] = row['session']
    if not table:
        raise ValueError(f'No rows read from {csv_path}')
    return table


def session_of(sessions, cls, fname, strict=True):
    """Look up one file. Falls back to a per-file singleton if unknown."""
    key = (cls, fname)
    if key in sessions:
        return sessions[key]
    if strict:
        raise KeyError(
            f'No session for {cls}/{fname}. Regenerate sessions.csv against '
            f'the same image folder you are training on.')
    return f'unmapped:{cls}:{fname}'


# --------------------------------------------------------------------------
# 2. grouped 80/10/10 split
# --------------------------------------------------------------------------

def do_split_grouped(src, dst, sessions, ratios=(0.8, 0.1, 0.1), seed=42,
                     folder_rename=None, verbose=True):
    """Session-aware replacement for do_split().

    Whole sessions are assigned to a split; no session is ever divided.
    Sessions are placed largest-first into whichever split currently has the
    biggest shortfall against its target, which keeps the realised ratios
    close to the requested ones despite very uneven session sizes.
    """
    folder_rename = folder_rename or {}
    dst = Path(dst)
    if dst.exists():
        if verbose:
            print(f'Split exists at {dst}, skipping. Delete it to rebuild.')
        return
    rng = random.Random(seed)
    splits = ['train', 'val', 'test']

    for cls_dir in sorted(Path(src).iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = folder_rename.get(cls_dir.name, cls_dir.name)
        imgs = [f for f in cls_dir.rglob('*') if f.suffix.lower() in IMG_EXT]
        if not imgs:
            continue

        by_session = defaultdict(list)
        for f in imgs:
            by_session[session_of(sessions, cls, f.name)].append(f)

        groups = list(by_session.items())
        rng.shuffle(groups)
        groups.sort(key=lambda kv: -len(kv[1]))

        total = len(imgs)
        target = {s: total * r for s, r in zip(splits, ratios)}
        cur = {s: 0 for s in splits}
        assign = {}
        for sid, files in groups:
            s = max(splits, key=lambda k: target[k] - cur[k])
            assign[sid] = s
            cur[s] += len(files)

        # a class with >= 3 sessions must not leave val or test empty
        if len(groups) >= 3:
            for need in ('test', 'val'):
                if cur[need] == 0:
                    donor = min((sid for sid, _ in groups if assign[sid] == 'train'),
                                key=lambda sid: len(by_session[sid]), default=None)
                    if donor is not None:
                        n = len(by_session[donor])
                        cur['train'] -= n
                        cur[need] += n
                        assign[donor] = need

        for sid, files in groups:
            out = dst / assign[sid] / cls
            out.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, out / f.name)

        if verbose:
            print(f'  {cls:<22}: total={total:>4} '
                  f'train={cur["train"]:>4} val={cur["val"]:>3} test={cur["test"]:>3} '
                  f'| {len(groups)} sessions')


# --------------------------------------------------------------------------
# 3. grouped cross-validation with an inner validation split
# --------------------------------------------------------------------------

def max_usable_k(labels, groups, k_wanted=5):
    """Largest k for which every class still has >= k distinct sessions.

    With 3 Pink_disease sessions, a 5-fold split cannot put that class in
    every test fold. Running k=5 anyway produces folds where the rare class
    is absent from test, and its per-fold F1 becomes undefined.
    """
    per_class = defaultdict(set)
    for lab, g in zip(labels, groups):
        per_class[lab].add(g)
    counts = {c: len(s) for c, s in per_class.items()}
    limit = min(counts.values())
    return max(2, min(k_wanted, limit)), counts


def grouped_cv_folds(labels, groups, k=5, seed=42, inner_val_frac=0.15):
    """Yield (train_idx, inner_val_idx, test_idx) per fold.

    No session appears in more than one of the three parts. The inner
    validation set exists so checkpoint selection never touches the test
    fold -- the original code selected the best epoch on the test fold and
    then reported that same fold, which is optimistic regardless of grouping.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    rng = random.Random(seed)

    if _HAS_SGKF:
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        outer = splitter.split(np.zeros(len(labels)), labels, groups)
    else:
        splitter = GroupKFold(n_splits=k)
        outer = splitter.split(np.zeros(len(labels)), labels, groups)

    for tr_all, te in outer:
        tr_groups = sorted(set(groups[tr_all]))
        rng.shuffle(tr_groups)
        n_val = max(1, int(round(len(tr_groups) * inner_val_frac)))
        val_groups = set(tr_groups[:n_val])

        tr = np.array([i for i in tr_all if groups[i] not in val_groups])
        va = np.array([i for i in tr_all if groups[i] in val_groups])
        if len(va) == 0 or len(tr) == 0:        # degenerate, fall back
            tr, va = tr_all, tr_all[:0]
        yield tr, va, te


def describe_folds(labels, groups, class_names, k, seed=42, mode='group'):
    """Print a per-fold class/session breakdown before training starts."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    print(f'\n  fold composition (k={k}, mode={mode}):')
    for i, (tr, va, te) in enumerate(cv_folds(labels, groups, k, seed, mode)):
        te_cls = Counter(labels[te].tolist())
        missing = [class_names[c] for c in range(len(class_names))
                   if te_cls.get(c, 0) == 0]
        print(f'    fold {i+1}: train {len(tr):>4} imgs / '
              f'{len(set(groups[tr])):>3} sess | '
              f'inner-val {len(va):>3} / {len(set(groups[va])):>2} | '
              f'test {len(te):>3} / {len(set(groups[te])):>2}')
        if missing:
            print(f'             !! no test samples for: {", ".join(missing)}')


def build_cv_index(split_dir, class_names, sessions, splits=('train', 'val', 'test')):
    """Collect every image across all splits, with label and session id.

    Returns (samples, labels, groups) where samples is [(path, label), ...].
    """
    samples, labels, groups = [], [], []
    for split in splits:
        for cls in class_names:
            cls_dir = Path(split_dir) / split / cls
            if not cls_dir.exists():
                continue
            label = class_names.index(cls)
            for f in sorted(cls_dir.rglob('*')):
                if f.suffix.lower() not in IMG_EXT:
                    continue
                samples.append((str(f), label))
                labels.append(label)
                groups.append(session_of(sessions, cls, f.name))
    return samples, np.array(labels), np.array(groups)

def image_cv_folds(labels, groups, k=5, seed=42, inner_val_frac=0.15):
    """Image-level folds, used as the control condition.

    Deliberately identical to grouped_cv_folds in every respect except that
    the grouping variable is ignored, so the difference between the two sets
    of numbers isolates the split rule and nothing else. In particular the
    inner validation split is kept, so checkpoint selection never touches
    the test fold in either regime.
    """
    from sklearn.model_selection import StratifiedKFold
    labels = np.asarray(labels)
    rng = random.Random(seed)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr_all, te in skf.split(np.zeros(len(labels)), labels):
        tr_all = np.asarray(tr_all)
        idx = list(tr_all)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * inner_val_frac)))
        va = np.array(idx[:n_val])
        tr = np.array(idx[n_val:])
        if len(va) == 0 or len(tr) == 0:
            tr, va = tr_all, tr_all[:0]
        yield tr, va, te


def cv_folds(labels, groups, k=5, seed=42, mode="group", inner_val_frac=0.15):
    """Dispatch on split rule. mode is 'group' or 'image'."""
    if mode == "image":
        return image_cv_folds(labels, groups, k, seed, inner_val_frac)
    return grouped_cv_folds(labels, groups, k, seed, inner_val_frac)
