# PATCH — session-aware splitting and cross-validation for `train.py`

Three edits. Put `session_utils.py` next to `train.py` first.

Everything else in `train.py` is untouched.

---

## Before you start

Generate the session map against the **raw** class folders you feed to
`do_split` (not against an already-split folder):

```
python session_split.py --root <raw_or_split_folder> --manifest sessions.csv
```

`sessions.csv` must cover every image `train.py` will see, otherwise
`session_of()` raises a `KeyError` telling you which file is missing. That
is deliberate — a silent fallback would reintroduce leakage quietly.

---

## Edit 1 — imports

Near the other imports (around line 70, next to the sklearn import):

**Replace**

```python
from sklearn.model_selection import StratifiedKFold
```

**with**

```python
from session_utils import (
    load_sessions, do_split_grouped, build_cv_index,
    max_usable_k, grouped_cv_folds, describe_folds,
)
```

Add the CLI flag next to the other `add_argument` calls (around line 98):

```python
parser.add_argument('--sessions', type=str, default='sessions.csv',
                    help='Session map from session_split.py')
```

And after `SEED = args.seed` (around line 108):

```python
SESSIONS = load_sessions(args.sessions)
```

---

## Edit 2 — the split (around line 193–229)

**Replace the whole `do_split` function and its call** with:

```python
print('\n' + '='*60)
print('STEP 1: Split Malaysia dataset (80/10/10, grouped by capture session)')
print('='*60)
do_split_grouped(MY_DATA, SPLIT_DIR, SESSIONS,
                 ratios=(0.8, 0.1, 0.1), seed=SEED,
                 folder_rename=FOLDER_RENAME)
```

The old `do_split` shuffled a flat image list per class, so consecutive
frames of one lesion landed on both sides of the boundary. The replacement
assigns whole sessions.

**If you would rather not re-split at all**, point `--split_dir` at the
verified folder and skip this edit entirely:

```
python train.py --split_dir clean_split --sessions sessions.csv --retrain
```

`do_split_grouped` returns immediately when the destination already exists,
so it will not overwrite `clean_split`. Keep the edit anyway for the public
repo — reviewers need to see the split being produced from raw data.

---

## Edit 3 — cross-validation (around line 1124–1217)

This is the substantive one. Two independent problems in the original block:

1. `StratifiedKFold` splits at image level, so folds leak.
2. Inside each fold, `fold_test_loader` was used **both** to pick the best
   epoch and to report the final score. That is optimistic even with
   perfect grouping.

**Replace** everything from `all_disease_samples = []` down to the line
before `df_cv = pd.DataFrame(cv_rows)` with:

```python
all_disease_samples, all_labels_cv, all_groups_cv = build_cv_index(
    SPLIT_DIR, list(class_names), SESSIONS)

print(f'Total disease images for CV: {len(all_disease_samples)}')
print(f'Distinct capture sessions  : {len(set(all_groups_cv))}')

k, sess_per_class = max_usable_k(all_labels_cv, all_groups_cv, k_wanted=5)
print('\n  sessions per class:')
for lab, n in sorted(sess_per_class.items()):
    print(f'    {class_names[lab]:<22}: {n}')
if k < 5:
    rare = min(sess_per_class, key=sess_per_class.get)
    print(f'\n  !! k reduced to {k}: "{class_names[rare]}" has only '
          f'{sess_per_class[rare]} sessions.')
    print('     Report this in the paper rather than forcing k=5.')

describe_folds(all_labels_cv, all_groups_cv, list(class_names), k, seed=SEED)


class SimpleDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


cv_rows = []

for fold, (tr_idx, va_idx, te_idx) in enumerate(
        grouped_cv_folds(all_labels_cv, all_groups_cv, k=k, seed=SEED)):

    print(f'\n  Fold {fold+1}/{k}')
    tr_samp = [all_disease_samples[i] for i in tr_idx]
    va_samp = [all_disease_samples[i] for i in va_idx]
    te_samp = [all_disease_samples[i] for i in te_idx]

    # label remap must be built from ALL parts, not just train, or a class
    # present only in test would shift every index
    unique_labels = sorted(set(l for _, l in
                               tr_samp + va_samp + te_samp))
    lmap = {orig: new for new, orig in enumerate(unique_labels)}
    tr_mapped = [(p, lmap[l]) for p, l in tr_samp]
    va_mapped = [(p, lmap[l]) for p, l in va_samp]
    te_mapped = [(p, lmap[l]) for p, l in te_samp]
    fold_n_cls = len(unique_labels)

    tr_ds = SimpleDataset(tr_mapped, train_tf)
    va_ds = SimpleDataset(va_mapped, test_tf)
    te_ds = SimpleDataset(te_mapped, test_tf)

    fold_targets = [s[1] for s in tr_mapped]
    fold_counts = Counter(fold_targets)
    fold_w = torch.tensor(
        [len(fold_targets)/(fold_n_cls*max(fold_counts[i], 1))
         for i in range(fold_n_cls)], dtype=torch.float).to(device)
    fold_sw = [fold_w[t].item() for t in fold_targets]
    fold_sampler = WeightedRandomSampler(fold_sw, len(fold_sw), replacement=True)

    fold_train_loader = DataLoader(tr_ds, batch_size=BATCH,
                                   sampler=fold_sampler, num_workers=NW)
    fold_val_loader   = DataLoader(va_ds, batch_size=BATCH,
                                   shuffle=False, num_workers=NW)
    fold_test_loader  = DataLoader(te_ds, batch_size=BATCH,
                                   shuffle=False, num_workers=NW)

    fold_model = build_agri_efficientnet(fold_n_cls, attention='lfa').to(device)
    fold_crit  = nn.CrossEntropyLoss(weight=fold_w)

    def _val_acc(m):
        m.eval(); correct = total = 0
        with torch.no_grad():
            for X, y in fold_val_loader:
                X, y = X.to(device), y.to(device)
                correct += (m(X).argmax(1) == y).sum().item()
                total += y.size(0)
        return correct / max(total, 1)

    # --- Stage 1: frozen backbone, select on INNER VAL -------------------
    fold_best, fold_best_st = -1.0, None
    opt = torch.optim.Adam(
        [p for p in fold_model.parameters() if p.requires_grad], lr=1e-3)
    for ep in range(10):
        fold_model.train()
        for X, y in fold_train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad(); loss = fold_crit(fold_model(X), y)
            loss.backward(); opt.step()
        va = _val_acc(fold_model)
        if va > fold_best:
            fold_best, fold_best_st = va, copy.deepcopy(fold_model.state_dict())
    if fold_best_st is not None:
        fold_model.load_state_dict(fold_best_st)

    # --- Stage 2: full fine-tune, also selected on INNER VAL -------------
    for p in fold_model.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(fold_model.parameters(), lr=1e-5)
    stage2_best, stage2_best_st = _val_acc(fold_model), \
        copy.deepcopy(fold_model.state_dict())
    for ep in range(5):
        fold_model.train()
        for X, y in fold_train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad(); loss = fold_crit(fold_model(X), y)
            loss.backward(); opt.step()
        va = _val_acc(fold_model)
        if va > stage2_best:
            stage2_best, stage2_best_st = va, copy.deepcopy(fold_model.state_dict())
    fold_model.load_state_dict(stage2_best_st)

    # --- evaluate ONCE on the untouched test fold ------------------------
    yt_f, yp_f = get_preds(fold_model, fold_test_loader)
    acc = accuracy_score(yt_f, yp_f) * 100
    f1  = f1_score(yt_f, yp_f, average='macro', zero_division=0) * 100
    cv_rows.append({'Fold': fold+1,
                    'Test images': len(te_idx),
                    'Test sessions': len(set(all_groups_cv[te_idx])),
                    'Accuracy (%)': round(acc, 2),
                    'Macro F1 (%)': round(f1, 2)})
    print(f'    Fold {fold+1} => Acc={acc:.2f}%  F1={f1:.2f}%  '
          f'(inner-val acc {stage2_best:.3f})')
```

The `df_cv = pd.DataFrame(cv_rows)` block below it works unchanged; it will
just pick up the two extra columns.

---

## What to expect

Scores will drop. Phomopsis and Root_disease should fall the most — those
two had the largest single sessions (78 and 49 images) spanning splits, and
they are the two classes reporting near-perfect metrics in Table 4.

Keep the old numbers. The gap between the image-level split and the
session-level split is a result in its own right, and it is the part of
this work that generalises beyond durian.

## Also needs regenerating afterwards

- `Fig5_confusion_matrix`, `FigS2_per_class_f1`, `FigS3_performance_overview`
- `Fig2_` Grad-CAM — rerun `regen_gradcam.py` against the new checkpoints
- `Fig6_robustness` — `fix_robustness.py` reads the test set
- `Fig7_cross_country` — Malaysian weights changed, Vietnamese data did not
- `Fig4_efficiency_scatter` — F1 axis moves; size and latency do not
- `Fig1_class_distribution` — add capture sessions per class
