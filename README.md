# Sampling unit, not image count — durian disease classification

Code and evaluation protocol for *Sampling Unit, Not Image Count: Capture-Session
Structure Governs the Accuracy Reported for Image-Based Orchard Disease Diagnosis*.

Dataset: **https://doi.org/10.5281/zenodo.22177133**

\---

## What this repository is for

The dataset contains 560 field images of five durian disease categories, but they
resolve into only **73 reconstructed capture sessions** — bursts of one lesion,
video frames, messaging batches. Sessions are recovered after the fact from
filename structure, so treat them as the finest recoverable grouping unit rather
than as 73 independent observations: images within a session are certainly
dependent, sessions between themselves only approximately independent. If you
split at image level, near-identical views of the same specimen land on both
sides of the train/test boundary and every metric you report is inflated.

In this dataset that is not a marginal effect. Two figures matter and they are
easily confused. Counting any boundary, **79.6% of images sit in sessions that
appear in more than one of train/val/test**. The figure that bears on a reported
test score is larger: **58 of the 60 image-level test images (96.7%) belong to a
session that also appears in training**, through 25 shared sessions. Validation is
contaminated on the same scale (94.4%), so model selection is affected too. Under
the session-level partition the same measure is **0.0%** — verified, no session
appears in more than one split.

Re-running the identical experiment with sessions kept whole lowers macro F1 by
**12.2 points on average across nine architectures** (range 4.9–18.4, positive for
all nine). We deliberately report no p-value for that consistency: nine
architectures sharing one dataset, one task and one pair of partitions are not
nine independent replicates, and testing over them would repeat at the level of
our own inference the error this work describes. An earlier three-seed analysis
also appeared to show the ranking of attention modules reversing between the two
protocols; a fourth seed removed that, and the paper reports the retraction. Do
not cite the reversal.

`sessions.csv` ships with the data so that grouped evaluation is the default.
**If you use this dataset, group by the `session` column.**

One caveat on that column. Session identifiers are constructed within class, so a
continuous camera sequence whose frames were filed under two disease categories
becomes two sessions. Forty-one camera numbers appear under more than one class,
and under the session-level partition nine adjacent-numbered pairs (twelve images,
2.1% of the dataset) still fall on opposite sides of train/test. Grouping by
session removes most, not all, of the dependence the capture process induces.
`check\_contamination.py` reproduces every number in this paragraph.

\---

## Quick start

```bash
pip install -r requirements.txt

# 1. inspect the session structure and audit any split you already have
python session\_split.py --root <dataset\_root> --manifest sessions.csv

# 2. build the session-level partition
python session\_split.py --root <dataset\_root> --rebuild clean\_split

# 3. train and evaluate
python train.py \\
    --split\_dir clean\_split --malaysia\_data clean\_split \\
    --vietnam\_data <vietnam\_root> --sessions sessions.csv \\
    --cv\_mode group --ckpt\_dir ckpt --save\_dir results --seed 42
```

To reproduce every number in the paper, including the image-level control:

```bash
python make\_image\_split.py --src <original\_split> --dst image\_split
bash run\_all.sh                       # 4 session-level seeds + 3 image-level
python aggregate.py --root results --out summary
```

`run\_all.sh` writes a `DONE` marker per run and skips completed ones, so an
interrupted run resumes by re-running the same command. Do not pass `--retrain`
when resuming: it discards the checkpoints that make resumption possible.

\---

## Files

### Evaluation protocol

|File|Purpose|
|-|-|
|`session\_utils.py`|Session lookup, grouped 80/10/10 split, `StratifiedGroupKFold` folds with an inner validation split, and the image-level control that differs *only* in the grouping variable|
|`train.py`|Training and full evaluation. `--cv\_mode group\|image` selects the partition rule|
|`make\_image\_split.py`|Builds the image-level control from the original partition, verifying counts against Table 1|
|`run\_all.sh`|Both regimes across seeds, resumable|
|`aggregate.py`|Per-seed results → mean ± s.d. and the paired inflation table|

### Auditing

|File|Purpose|
|-|-|
|`session\_split.py`|Recovers capture sessions from filename structure; reports how much of a given partition leaks; `--gap` varies the burst threshold|
|`audit\_dataset.py`|Exact and near-duplicate detection within one dataset|
|`label\_conflicts.py`|Identical images carrying two class labels|
|`audit\_public.py`|Audits a *published* split: whole-split overlap, cross-split duplicates, numbering adjacency, preprocessing consistency. Works on any `root/<split>/<class>/` layout|
|`check\_contamination.py`|Reports what a split actually leaks: session overlap between train/val/test separately, the share of test images whose session also appears in training, and cross-class camera-sequence dependence that grouping by session does not remove. Run `python check\_contamination.py --data .`|
|`check\_provenance.py`|\| `phash\_sessions.py` \| Recovers capture sessions from the images themselves, by perceptual hash, for datasets whose filenames were renumbered after collection and therefore carry no capture structure. Used on the Vietnamese datasets \|<br />\| `classify\_redundancy.py` \| Distinguishes near-duplicate images that are one capture burst from those that are copied files, which decides whether the redundancy is a sampling artefact or a data-handling error \|<br />\| `export\_confusion.py` \| Writes the confusion matrix behind Fig. 4 as CSV, so the figure can be redrawn without rerunning inference \||
|`fill\_table\_s1.py`|Recomputes capture sessions at a range of burst-gap thresholds and measures how much of the dataset lands in boundary-crossing sessions. Reproduces Table S1. Run `python fill\_table\_s1.py --data .`|
|`peek.py`|Folder inventory, for looking at an unfamiliar dataset before anything else|

### Preprocessing

|File|Purpose|
|-|-|
|`shrink.py`|512 px copy for training. Full-resolution originals are what gets released|
|`make\_centrecrop.py`|The centre-crop control used to bound the preprocessing contribution to the cross-country gap|

### Cross-country diagnostics

|File|Purpose|
|-|-|
|`fix\_vietnam\_metrics.py`|Recomputes the cross-country scores with the correct label set, reports both variants so the size of the `labels=` artefact is visible, prints the uniform, prior-matched and majority-class baselines, and writes the confusion matrices|
|`fix\_crop\_control.py`|Recomputes the centre-crop preprocessing control over the same three classes as the main comparison, so the source and target macros share a basis|
|`vietnam\_indomain.py`|Trains on the Vietnamese data under the same four-class mapping to establish whether the mapped task is learnable at all; optional few-shot curve from a Malaysian checkpoint|
|`malaysia\_subset\_metrics.py`|Recomputes the source-side scores over the same class subsets the target evaluation covers, so the drop is like for like. Reproduces the published five-class figures exactly, which is how you check it is working|
|`diagnose\_mapping.py`|Reads the confusion matrices and asks, per class, whether a correspondence is failing in a way that points at the mapping rather than at transfer. Two signatures: an inactive predicted class, and errors pooling on one specific wrong class|

### Release

|File|Purpose|
|-|-|
|`make\_figures.py`|Regenerates every manuscript figure except Grad-CAM. Values are written at the top of the file and match the tables; if a table changes, change it there too|
|`zenodo\_metadata.json`|Dataset record metadata, four fields to fill|
|`RELEASE\_CHECKLIST.md`|Ordered steps from withdrawal confirmation to submission|

### Model and figures

|File|Purpose|
|-|-|
|`mobilenetv2\_lfa.py`|MobileNetV2 + LFA variant|
|`regen\_gradcam.py`|Grad-CAM figures — **must be run against session-level checkpoints**|
|`fix\_robustness.py`|Perturbation evaluation|

\---

## Reproducing the leakage audit on your own data

`session\_split.py` recovers sessions from filename structure: video frames
(`IMG\_9954\_frame00648`), consecutive camera sequence numbers within a gap of
three, and messaging batches. It will tell you how much of your current split
leaks:

```bash
python session\_split.py --report --gap 3
python session\_split.py --report --gap 5
python session\_split.py --report --gap 10
```

On this dataset the leaked fraction rises monotonically from 79.6% to 85.7% as
the threshold loosens, so the reported figure is a lower bound rather than an
artefact of the threshold. Run all three on your data too — if the number moves
a lot, say so rather than picking the flattering one.

If your filenames carry no capture order — many public datasets renumber after
splitting — filename structure will not recover sessions and you need perceptual
hashing instead. `audit\_public.py` will still detect exact and near duplicates
across splits, which is the part that matters most.

\---

### Session structure

`sessions.csv` resolves the 560 images into 73 sessions: 48 burst sessions (463
images), 7 video-derived (63), 4 messaging-batch (20), and 14 files that matched
no rule. A further 11 recovered groups hold one image, so 25 of the 73 sessions
are single-image sessions.

The distribution is strongly skewed and this matters for reading the per-class
results. Median session size is 4 images, the largest is 78, and the ten largest
sessions hold 295 images -- 52.7% of the dataset. Within classes the single
largest session holds 49.7% of Phomopsis and 62.8% of Root\_disease, which is why
those two classes' high per-class scores should be read as provisional. Reproduce
with:

```bash
python session\_split.py --report
```

## Results this repository reproduces

Macro F1 %, mean ± s.d. over four seeds (42, 1, 2, 3) in both partition
conditions.

|Model|Image-level|Session-level|Inflation (pp)|
|-|-|-|-|
|EfficientNetV2-S|93.4 ± 1.5|88.5 ± 3.1|+4.9|
|ResNet-101|94.3 ± 1.1|85.3 ± 5.2|+9.0|
|ConvNeXt-Tiny|97.3 ± 2.3|81.8 ± 3.6|+15.5|
|EfficientNet-B0 + LFA (ours)|82.1 ± 3.1|76.1 ± 3.7|+6.0|
|VGG-16|92.9 ± 6.0|76.0 ± 6.9|+16.9|
|ResNet-50|88.3 ± 3.2|75.2 ± 2.3|+13.1|
|MobileNetV3-Large|92.4 ± 1.2|74.0 ± 2.7|+18.4|
|EfficientNet-B0|83.8 ± 1.9|73.0 ± 2.4|+10.8|
|MobileNetV2|87.9 ± 2.0|72.9 ± 4.2|+15.0|
|**Mean**|—|—|**+12.2**|

Grouped 3-fold cross-validation: **59.6 ± 4.5%**, against **79.3 ± 1.1%** for the
image-level control. The image-level control is *tighter as well as higher* —
when every fold contains near-duplicates of its own training data, folds agree
with each other while all overstating the same quantity.

Cross-country, zero-shot on the Vietnamese dataset: all nine architectures lose
**20.9–42.3 points** over the three foliar correspondences, finishing close to the
**32.5% chance baseline**. Of those three, Phomopsis transfers at only 13.7%
recall, low enough that the correspondence should not be assumed to hold without
local checking. The spread is wide and
does not follow home performance — ConvNeXt-Tiny is sixth of nine on the
Malaysian test set and first on the Vietnamese one.

Both columns of that comparison are macro-averaged over the same three classes.
Subtracting a five-class Malaysian macro from a four-class Vietnamese one, which
is what the first version of this work did, is not a transfer gap. Use
`malaysia\_subset\_metrics.py` to recompute the source-side scores over whatever
subset the target evaluation actually covers.

**One of the four class correspondences does not hold, and the aggregate score
could not tell us.** Leaf\_Rhizoctonia (foliar) was mapped to Root\_disease
(trunk and collar) on shared treatment. Pooled over nine architectures and four
seeds, recall on those images is 1.5% and the predicted class is inactive —
assigned to 1.1% of the 320 test images against 19.1% of the labels. A model
trained in-domain recovers the same 61 images at 98.4% recall, so the images are
not the problem. `diagnose\_mapping.py` runs this check from the confusion
matrices; it costs one inference pass and it is the first thing to do to any
cross-dataset label mapping. An in-domain control trained on the Vietnamese data under the same four-class
mapping reaches **93.0 ± 1.4% macro F1** over three seeds
(`vietnam\_indomain.py`), so the target task is learnable. The Vietnamese dataset
ships an image-level partition, so that figure is an upper bound even after
exact cross-split duplicates are removed. A matched centre-crop control on the
Malaysian data, scored on the same three classes, closes **45%** of that gap
(mean 40.5 → 22.1 points). Most of the narrowing is the target improving, not
the source degrading: Vietnamese scores rise 14.7 points while Malaysian fall
3.7. Estimated from mixed class bases the same control looked like a quarter,
which is why `fix\_crop\_control.py` exists.

Two things the inflation column is *not*. It does not track model capacity —
parameter count and inflation are uncorrelated (Pearson r = 0.29, p = 0.45),
and the highest inflation belongs to MobileNetV3-Large at 4.21M parameters.
And it does not preserve the ranking: the two orderings agree only at Spearman
ρ = 0.65 (p = 0.058, just short of significance with nine points), and the model proposed
in the paper is fourth under the session-level protocol and last under the
image-level one. The inflation itself is unambiguous — all nine positive, sign
positive for all nine — but whether leakage systematically reorders architectures is
not settled by nine points.

\---

## Naming

The paper calls the proposed configuration **EfficientNet-B0 + LFA**. The code
still uses the key `agri\_efficientnet`, and checkpoints are named
`cmp\_agri\_efficientnet.pth`, because renaming them would break every saved
checkpoint. They are the same model. The brand name was dropped from the paper
once the ablation showed the module confers no benefit that survives multi-seed,
leakage-free evaluation.

## Things worth knowing before you build on this

* **ShuffleNetV2-1.0x does not converge** under the shared two-stage protocol
(21.0 ± 1.1% across four seeds). It is excluded from the paper rather than
tuned separately, because a per-architecture search would break the controlled
comparison. If you tune it, report that you did.
* **The proposed LFA module shows no benefit** that survives multi-seed,
leakage-free evaluation. It is kept as the smallest-footprint option among
indistinguishable alternatives. Do not cite this repository as evidence that
it works. An earlier three-seed analysis appeared to show the module ranking
reversing between partition rules; a fourth seed removed that, and the paper
reports the retraction.
* **The ablation (`abl\_\*.pth`) and the comparison (`cmp\_\*.pth`) train the shared
configurations independently.** Two instances of the same model therefore
differ by up to 8.5 points at a matched seed. That gap is larger than any
difference between attention modules and is reported in the paper as
run-to-run variation. Do not mix numbers across the two experiments.
* **`calc\_metrics` in `train.py` does not pass `labels=`, and this is a real
bug, not a nicety.** For the Vietnamese evaluation, where Pink\_disease has no
counterpart, scikit-learn averages over the union of true and predicted
labels. Every model predicted the absent class on at least one of the 320
images (between 1 and 319 times), so the divisor was five instead of four and
the correct value is 25% higher than the reported one. `fix\_vietnam\_metrics.py`
recomputes correctly and prints both values; the numbers in this README and
in the paper are the corrected ones. If you reuse `train.py` for a task where
a training class is absent from the test set, fix this first.
* **Pink\_disease has three capture sessions in total.** Its per-class metrics are
not interpretable and its presence caps grouped cross-validation at k = 3.
* **Cross-validation uses a shortened 10 + 5 epoch schedule**, not the 15 + 15 of
the main runs. Both regimes use it, so the comparison holds, but the absolute
cross-validation numbers are not directly comparable to the held-out ones.
* **Robustness and per-class results are single-seed** (seed 42).
* **Evaluate at the resolution you trained at.** Every number here is computed
on the 512 px copies. Running the same checkpoint over the full-resolution
originals through the identical `Resize(256)` + `CenterCrop(224)` pipeline
gives 77.6% instead of 72.0% on the held-out set, because the two resampling
paths to 224 px differ. The Zenodo record ships the originals; downscale them
before reproducing anything.
* **Verify a class mapping before you trust a cross-dataset number.** This work
mapped four Vietnamese classes onto Malaysian ones and one of them turned out
to be wrong. The aggregate macro F1 looked like uniform degradation; the
per-class table showed one correspondence at 1.5% recall while another was at
84.2%. Run `diagnose\_mapping.py` and, if you can, `vietnam\_indomain.py`.
* **`diagnose\_mapping.py` has a verdict heuristic, not a test.** Its
concentration threshold is arbitrary and on our own data it under-called the
result. Read the table it prints; do not rely on the last line.
* Training uses 512 px copies; the release contains full-resolution originals.

## What is in `results/`

Eight directories, one per (partition rule × seed) combination, holding the raw
outputs every table in the paper is computed from:

&#x20;   results/group\_s42  group\_s1  group\_s2  group\_s3     session-level partition
    results/image\_s42  image\_s1  image\_s2  image\_s3     image-level partition


Each contains:

|File|Feeds|
|-|-|
|`comparison\_table.csv`|Tables 4 and 5 of the paper, Table S4 of the supplement|
|`ablation\_results.csv`|Sect. 4.8, Table S6|
|`cv\_results.csv`|Sect. 4.4, Table S5|
|`per\_class\_metrics.csv`|Table 6|
|`robustness\_results.csv`|Table 7|
|`vietnam\_validation.csv`|Tables 8 and 9, before the averaging correction|
|`mcnemar\_results.csv`|the pairwise tests in Sect. 4.8|
|`dataset\_statistics.csv`|Tables 1 and 2|

Two notes for anyone recomputing from these files.

`vietnam\_validation.csv` holds macro F1 averaged over five classes, including
pink disease, which has no counterpart in the target data. The paper averages
over the four mapped classes instead, which is why its Vietnamese figures are
higher than the raw ones here by a factor of about 1.25. `fix\_vietnam\_metrics.py`
applies the correction. This is one of the four protocol decisions the paper
measures, and the discrepancy between these two files is what it looks like.

The ablation in Sect. 4.8 uses a replication design of two runs per
configuration per seed. `ablation\_results.csv` holds one run per seed; the
second replicate is not in this release.

## Citation

```bibtex
@article{lin2026sampling,
  title   = {Sampling Unit, Not Image Count: Capture-Session Structure Governs
             the Accuracy Reported for Image-Based Orchard Disease Diagnosis},
  author  = {Lin, Ding Shan},
  journal = {(under review)},
  year    = {2026}
}
```

## License

Code: MIT (see `LICENSE`). Dataset: see the Zenodo record.

