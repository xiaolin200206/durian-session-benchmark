# What a reported accuracy measures — durian disease classification

Code and evaluation protocol for *What a Reported Accuracy Measures: Capture-Session
Leakage and Cross-Country Transfer in Durian Disease Classification*.

Dataset: **〔Zenodo DOI〕**

---

## What this repository is for

The dataset contains 560 field images of five durian disease categories, but they
were produced in only **73 independent capture sessions** — bursts of one lesion,
video frames, messaging batches. If you split at image level, near-identical views
of the same specimen land on both sides of the train/test boundary and every
metric you report is inflated.

In this dataset that is not a marginal effect. Under an image-level random split,
**79.6% of images sit in sessions that straddle a partition boundary**. Re-running
the identical experiment with sessions kept whole lowers macro F1 by **11.8 points
on average across nine architectures** (range 4.7–18.7), and **reverses the ranking
of attention modules**: squeeze-and-excitation is best under the image-level
protocol and worst under the session-level one.

`sessions.csv` ships with the data so that grouped evaluation is the default.
**If you use this dataset, group by the `session` column.**

---

## Quick start

```bash
pip install -r requirements.txt

# 1. inspect the session structure and audit any split you already have
python session_split.py --root <dataset_root> --manifest sessions.csv

# 2. build the session-level partition
python session_split.py --root <dataset_root> --rebuild clean_split

# 3. train and evaluate
python train.py \
    --split_dir clean_split --malaysia_data clean_split \
    --vietnam_data <vietnam_root> --sessions sessions.csv \
    --cv_mode group --ckpt_dir ckpt --save_dir results --seed 42
```

To reproduce every number in the paper, including the image-level control:

```bash
python make_image_split.py --src <original_split> --dst image_split
bash run_all.sh                       # 4 session-level seeds + 3 image-level
python aggregate.py --root results --out summary
```

`run_all.sh` writes a `DONE` marker per run and skips completed ones, so an
interrupted run resumes by re-running the same command. Do not pass `--retrain`
when resuming: it discards the checkpoints that make resumption possible.

---

## Files

### Evaluation protocol

| File | Purpose |
|---|---|
| `session_utils.py` | Session lookup, grouped 80/10/10 split, `StratifiedGroupKFold` folds with an inner validation split, and the image-level control that differs *only* in the grouping variable |
| `train.py` | Training and full evaluation. `--cv_mode group\|image` selects the partition rule |
| `make_image_split.py` | Builds the image-level control from the original partition, verifying counts against Table 1 |
| `run_all.sh` | Both regimes across seeds, resumable |
| `aggregate.py` | Per-seed results → mean ± s.d. and the paired inflation table |

### Auditing

| File | Purpose |
|---|---|
| `session_split.py` | Recovers capture sessions from filename structure; reports how much of a given partition leaks; `--gap` varies the burst threshold |
| `audit_dataset.py` | Exact and near-duplicate detection within one dataset |
| `label_conflicts.py` | Identical images carrying two class labels |
| `audit_public.py` | Audits a *published* split: whole-split overlap, cross-split duplicates, numbering adjacency, preprocessing consistency. Works on any `root/<split>/<class>/` layout |
| `peek.py` | Folder inventory, for looking at an unfamiliar dataset before anything else |

### Preprocessing

| File | Purpose |
|---|---|
| `shrink.py` | 512 px copy for training. Full-resolution originals are what gets released |
| `make_centrecrop.py` | The centre-crop control used to bound the preprocessing contribution to the cross-country gap |

### Cross-country diagnostics

| File | Purpose |
|---|---|
| `fix_vietnam_metrics.py` | Recomputes the cross-country scores with the correct label set, reports both variants so the size of the `labels=` artefact is visible, prints the uniform, prior-matched and majority-class baselines, and writes the confusion matrices |
| `fix_crop_control.py` | Recomputes the centre-crop preprocessing control over the same three classes as the main comparison, so the source and target macros share a basis |
| `vietnam_indomain.py` | Trains on the Vietnamese data under the same four-class mapping to establish whether the mapped task is learnable at all; optional few-shot curve from a Malaysian checkpoint |
| `malaysia_subset_metrics.py` | Recomputes the source-side scores over the same class subsets the target evaluation covers, so the drop is like for like. Reproduces the published five-class figures exactly, which is how you check it is working |
| `diagnose_mapping.py` | Reads the confusion matrices and asks, per class, whether a correspondence is failing in a way that points at the mapping rather than at transfer. Two signatures: an inactive predicted class, and errors pooling on one specific wrong class |

### Release

| File | Purpose |
|---|---|
| `make_figures.py` | Regenerates every manuscript figure except Grad-CAM. Values are written at the top of the file and match the tables; if a table changes, change it there too |
| `zenodo_metadata.json` | Dataset record metadata, four fields to fill |
| `RELEASE_CHECKLIST.md` | Ordered steps from withdrawal confirmation to submission |

### Model and figures

| File | Purpose |
|---|---|
| `mobilenetv2_lfa.py` | MobileNetV2 + LFA variant |
| `regen_gradcam.py` | Grad-CAM figures — **must be run against session-level checkpoints** |
| `fix_robustness.py` | Perturbation evaluation |

---

## Reproducing the leakage audit on your own data

`session_split.py` recovers sessions from filename structure: video frames
(`IMG_9954_frame00648`), consecutive camera sequence numbers within a gap of
three, and messaging batches. It will tell you how much of your current split
leaks:

```bash
python session_split.py --report --gap 3
python session_split.py --report --gap 5
python session_split.py --report --gap 10
```

On this dataset the leaked fraction rises monotonically from 79.6% to 85.7% as
the threshold loosens, so the reported figure is a lower bound rather than an
artefact of the threshold. Run all three on your data too — if the number moves
a lot, say so rather than picking the flattering one.

If your filenames carry no capture order — many public datasets renumber after
splitting — filename structure will not recover sessions and you need perceptual
hashing instead. `audit_public.py` will still detect exact and near duplicates
across splits, which is the part that matters most.

---

## Results this repository reproduces

Session-level, macro F1 %, mean ± s.d. over four seeds (42, 1, 2, 3); the
image-level control uses three (42, 1, 2).

| Model | Image-level | Session-level | Inflation (pp) |
|---|---|---|---|
| EfficientNetV2-S | 93.4 ± 1.5 | 88.5 ± 3.1 | +4.9 |
| ResNet-101 | 94.3 ± 1.1 | 85.3 ± 5.2 | +9.0 |
| ConvNeXt-Tiny | 97.3 ± 2.3 | 81.8 ± 3.6 | +15.5 |
| EfficientNet-B0 + LFA (ours) | 82.1 ± 3.1 | 76.1 ± 3.7 | +6.0 |
| VGG-16 | 92.9 ± 6.0 | 76.0 ± 6.9 | +16.9 |
| ResNet-50 | 88.3 ± 3.2 | 75.2 ± 2.3 | +13.1 |
| MobileNetV3-Large | 92.4 ± 1.2 | 74.0 ± 2.7 | +18.4 |
| EfficientNet-B0 | 83.8 ± 1.9 | 73.0 ± 2.4 | +10.8 |
| MobileNetV2 | 87.9 ± 2.0 | 72.9 ± 4.2 | +15.0 |
| **Mean** | — | — | **+12.2** |

Grouped 3-fold cross-validation: **59.6 ± 4.5%**, against **79.3 ± 1.1%** for the
image-level control. The image-level control is *tighter as well as higher* —
when every fold contains near-duplicates of its own training data, folds agree
with each other while all overstating the same quantity.

Cross-country, zero-shot on the Vietnamese dataset: all nine architectures lose
**20.9–42.3 points** over the three classes whose correspondence is organ to
organ, finishing close to the **32.5% chance baseline**. The spread is wide and
does not follow home performance — ConvNeXt-Tiny is sixth of nine on the
Malaysian test set and first on the Vietnamese one.

Both columns of that comparison are macro-averaged over the same three classes.
Subtracting a five-class Malaysian macro from a four-class Vietnamese one, which
is what the first version of this work did, is not a transfer gap. Use
`malaysia_subset_metrics.py` to recompute the source-side scores over whatever
subset the target evaluation actually covers.

**One of the four class correspondences does not hold, and the aggregate score
could not tell us.** Leaf_Rhizoctonia (foliar) was mapped to Root_disease
(trunk and collar) on shared treatment. Pooled over nine architectures and four
seeds, recall on those images is 1.5% and the predicted class is inactive —
assigned to 1.1% of the 320 test images against 19.1% of the labels. A model
trained in-domain recovers the same 61 images at 98.4% recall, so the images are
not the problem. `diagnose_mapping.py` runs this check from the confusion
matrices; it costs one inference pass and it is the first thing to do to any
cross-dataset label mapping. An in-domain control trained on the Vietnamese data under the same four-class
mapping reaches **93.0 ± 1.4% macro F1** over three seeds
(`vietnam_indomain.py`), so the target task is learnable. The Vietnamese dataset
ships an image-level partition, so that figure is an upper bound even after
exact cross-split duplicates are removed. A matched centre-crop control on the
Malaysian data, scored on the same three classes, closes **45%** of that gap
(mean 40.5 → 22.1 points). Most of the narrowing is the target improving, not
the source degrading: Vietnamese scores rise 14.7 points while Malaysian fall
3.7. Estimated from mixed class bases the same control looked like a quarter,
which is why `fix_crop_control.py` exists.

Two things the inflation column is *not*. It does not track model capacity —
parameter count and inflation are uncorrelated (Pearson r = 0.29, p = 0.45),
and the highest inflation belongs to MobileNetV3-Large at 4.21M parameters.
And it does not preserve the ranking: the two orderings agree only at Spearman
ρ = 0.65 (p = 0.058, just short of significance with nine points), and the model proposed
in the paper is fourth under the session-level protocol and last under the
image-level one. The inflation itself is unambiguous — all nine positive, sign
test p = 0.004 — but whether leakage systematically reorders architectures is
not settled by nine points.

---

## Naming

The paper calls the proposed configuration **EfficientNet-B0 + LFA**. The code
still uses the key `agri_efficientnet`, and checkpoints are named
`cmp_agri_efficientnet.pth`, because renaming them would break every saved
checkpoint. They are the same model. The brand name was dropped from the paper
once the ablation showed the module confers no benefit that survives multi-seed,
leakage-free evaluation.

## Things worth knowing before you build on this

- **ShuffleNetV2-1.0x does not converge** under the shared two-stage protocol
  (21.0 ± 1.1% across four seeds). It is excluded from the paper rather than
  tuned separately, because a per-architecture search would break the controlled
  comparison. If you tune it, report that you did.
- **The proposed LFA module shows no benefit** that survives multi-seed,
  leakage-free evaluation. It is kept as the smallest-footprint option among
  indistinguishable alternatives. Do not cite this repository as evidence that
  it works. An earlier three-seed analysis appeared to show the module ranking
  reversing between partition rules; a fourth seed removed that, and the paper
  reports the retraction.
- **The ablation (`abl_*.pth`) and the comparison (`cmp_*.pth`) train the shared
  configurations independently.** Two instances of the same model therefore
  differ by up to 8.5 points at a matched seed. That gap is larger than any
  difference between attention modules and is reported in the paper as
  run-to-run variation. Do not mix numbers across the two experiments.
- **`calc_metrics` in `train.py` does not pass `labels=`, and this is a real
  bug, not a nicety.** For the Vietnamese evaluation, where Pink_disease has no
  counterpart, scikit-learn averages over the union of true and predicted
  labels. Every model predicted the absent class on at least one of the 320
  images (between 1 and 319 times), so the divisor was five instead of four and
  the correct value is 25% higher than the reported one. `fix_vietnam_metrics.py`
  recomputes correctly and prints both values; the numbers in this README and
  in the paper are the corrected ones. If you reuse `train.py` for a task where
  a training class is absent from the test set, fix this first.
- **Pink_disease has three capture sessions in total.** Its per-class metrics are
  not interpretable and its presence caps grouped cross-validation at k = 3.
- **Cross-validation uses a shortened 10 + 5 epoch schedule**, not the 15 + 15 of
  the main runs. Both regimes use it, so the comparison holds, but the absolute
  cross-validation numbers are not directly comparable to the held-out ones.
- **Robustness and per-class results are single-seed** (seed 42).
- **Evaluate at the resolution you trained at.** Every number here is computed
  on the 512 px copies. Running the same checkpoint over the full-resolution
  originals through the identical `Resize(256)` + `CenterCrop(224)` pipeline
  gives 77.6% instead of 72.0% on the held-out set, because the two resampling
  paths to 224 px differ. The Zenodo record ships the originals; downscale them
  before reproducing anything.
- **Verify a class mapping before you trust a cross-dataset number.** This work
  mapped four Vietnamese classes onto Malaysian ones and one of them turned out
  to be wrong. The aggregate macro F1 looked like uniform degradation; the
  per-class table showed one correspondence at 1.5% recall while another was at
  84.2%. Run `diagnose_mapping.py` and, if you can, `vietnam_indomain.py`.
- **`diagnose_mapping.py` has a verdict heuristic, not a test.** Its
  concentration threshold is arbitrary and on our own data it under-called the
  result. Read the table it prints; do not rely on the last line.
- Training uses 512 px copies; the release contains full-resolution originals.

## Citation

```bibtex
@article{〔key〕,
  title   = {What a Reported Accuracy Measures: Capture-Session Leakage and
             Cross-Country Transfer in Durian Disease Classification},
  author  = {〔author〕},
  journal = {〔journal〕},
  year    = {〔year〕},
  doi     = {〔doi〕}
}
```

## License

Code: MIT (see `LICENSE`). Dataset: see the Zenodo record.
