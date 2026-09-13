# Capture sessions, not images — durian disease classification

Code, data and evaluation protocol for *Capture sessions, not images: the sampling unit governs
reported accuracy in field image data, and content-based auditing does not recover it*
(submitted to *Ecological Informatics*).

Dataset: https://doi.org/10.5281/zenodo.22177133

---

## What this repository is for

Two things, and the second is the reason the first matters.

**1. Content-based auditing does not recover the grouping.** Once a dataset ships without
sampling-unit identifiers, the structure cannot be reconstructed from the images by the methods in
common use. A near-duplicate audit flags **4.6%** of test images where the reference grouping flags
**93.9%** — report both, never their ratio, because a proxy that over-merges flags images the
reference does not, so a ratio can exceed one and does not say which images were missed.
Self-supervised features (DINOv2) recover more but reach no threshold at which pair precision and
recall are usable together: the best pair-F1 of any configuration is 0.402, against 0.298 for the
trivial grouping that merges each class entirely. The threshold calibrated here collapses into a
degenerate grouping on a second dataset whose preprocessing differs.

We do **not** claim that no method could succeed. We claim that these do not, under single-linkage
clustering, against a reference grouping that is itself reconstructed — and that a publisher has no
means of demonstrating, on a dataset released without identifiers, that any other method has,
because demonstrating it requires the reference grouping whose absence created the problem. The
measurement moves 5.7 pp across a 2.6-fold change in the reconstruction rule
(`results/grouping_sensitivity.csv`).

**2. Splitting at the image level inflates every metric.** The dataset contains 560 field images of
five durian disease categories, produced in only **73 capture sessions** — bursts of one lesion,
video frames, messaging batches. Under an image-level random split, **58 of the 60 test images
(96.7%) belong to a session that also appears in training**, and 79.6% of the dataset sits in
sessions that straddle a partition boundary. Re-running the identical experiment with sessions kept
whole lowers macro F1 by **12.2 points on average across nine architectures** (range 4.9–18.4,
positive in all nine) and alters their rank ordering. We do not attach a *p* value to that
consistency: nine architectures on one dataset are not nine independent replicates, and testing them
as if they were would repeat the error the paper is about.

**If you use this dataset, group by the `session` column of `sessions.csv`.**

---

## Scope of the release

The released dataset is the **five disease classes** listed below and nothing else. The source
collection from which it was drawn also held six pest-damage categories, added later and outside the
disease vocabulary under study; those are not part of this release and were set aside before any
analysis reported in the paper (Sect. 3.2).

| Class (paper) | Folder | Pathogen / cause | Organ photographed |
|---|---|---|---|
| Algal leaf spot | `Algal` | *Cephaleuros virescens* | leaf |
| Leaf rot | `Leaf_rot` | *Colletotrichum* spp. and related leaf blight | leaf |
| Phomopsis leaf spot | `Phomopsis` | *Phomopsis* sp. (now largely *Diaporthe*) | leaf |
| Pink disease | `Pink_disease` | *Erythricium salmonicolor* | branch |
| Root and collar rot | `Root_disease` | *Phytophthora* spp. | trunk, collar |

The classes are **symptom classes, not confirmed aetiologies**: identification rests on
field-observable symptoms, with no isolation, culture or molecular confirmation, and no second
independent rater. Phomopsis is photographed on **leaves** here; the pathogen is also reported on
fruit and stems of durian, but no fruit or stem images are in this dataset.

Pink disease is represented by **three capture sessions** across the entire collection period. Its
per-class metrics are not interpretable at that support and should not be quoted.

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
bash run_all.sh                       # session-level and image-level seeds
python aggregate.py --root results --out summary
```

`run_all.sh` writes a `DONE` marker per run and skips completed ones, so an interrupted run resumes
by re-running the same command. Do not pass `--retrain` when resuming: it discards the checkpoints
that make resumption possible.

**The one thing you can check without a GPU:** the sensitivity of the leakage measurement to the
grouping rule needs only the manifest — no images, no training.

```bash
python grouping_sensitivity.py --sessions sessions.csv --seeds 10 --out results
```

This reproduces `results/grouping_sensitivity.csv` and Table S14 of the paper in a few seconds.

---

## Reproducing the recoverability analysis (Sect. 4.2 of the paper)

This is the part you can run on **your own** dataset, and the part we would most like others to run.

```bash
# 1. calibrate a perceptual hash against the true sessions
python validate_proxy.py --root images_512 --sessions sessions.csv --out proxy

# 2. extract features from a learned representation
python extract_features.py --root images_512 --backend hub \
    --model dinov2_vitb14 --pool cls --out dinov2_cls.npz

# 3. calibrate that representation on the same footing, hash included for contrast
python embed_proxy.py --root images_512 --sessions sessions.csv \
    --embeddings dinov2_cls.npz \
    --sims 0.70,0.80,0.85,0.86,0.87,0.88,0.90,0.95 --out proxy_dinov2

# 4. apply the calibrated threshold to datasets with no ground truth
python leakage_survey.py \
    --dataset "MY=images_512" \
    --dataset "VN-A=<vietnam_root>" \
    --embeddings "MY=dinov2_cls.npz" \
    --embeddings "VN-A=vna_dino.npz" \
    --sim-threshold 0.86 --out survey
```

Four things the tooling enforces, because each of them is a way the analysis goes wrong quietly:

- **Precision is reported against a base rate.** The share of within-class image pairs that
  genuinely share a session is 0.175 here. A clustering that merges each class entirely scores
  recall 1.0 and precision 0.175, which looks like success and is not.
- **Detection is scored per image, not as a ratio of rates.** Dividing the proxy's leak rate by the
  reference's is not a detection rate; it can exceed one, and it does not say which images were
  missed. `leak_detection.py` gives the per-image contingency instead.
- **Degenerate solutions are flagged and excluded** from the choice of operating point — by
  comparison against ground truth where it exists (`embed_proxy.py`), and by structural criteria
  where it does not (`leakage_survey.py`).
- **Copies are separated from capture bursts.** Byte-identical and re-encoded duplicates are a
  dataset-assembly fault; near-identical but genuinely distinct frames are a property of the
  collection process. Averaging the two compares quantities that are not the same quantity.

Do **not** normalise the threshold per dataset by quantile, singleton fraction or mean cluster size.
Each of those is a monotone function of how much redundancy is detected, which is what the audit is
supposed to measure, so any such normalisation forces the datasets to agree on the quantity under
test.

---

## Files

### Evaluation protocol

| File | Purpose |
|---|---|
| `session_utils.py` | Session lookup, grouped 80/10/10 split, `StratifiedGroupKFold` folds with an inner validation split, and the image-level control that differs *only* in the grouping variable |
| `train.py` | Training and full evaluation. `--cv_mode group\|image` selects the partition rule |
| `make_image_split.py` | Builds the image-level control (446/54/60) from the original partition. Those are **not** the counts of Table 1, which reports the session-level partition (446/58/56) |
| `run_all.sh` | Both regimes across seeds, resumable |
| `aggregate.py` | Per-seed results → mean ± s.d. and the paired inflation table |

### Auditing and recoverability

| File | Purpose |
|---|---|
| `session_split.py` | Recovers capture sessions from filename structure; reports how much of a given partition leaks; `--gap` varies the burst threshold |
| `audit_dataset.py` | Exact and near-duplicate detection within one dataset |
| `audit_public.py` | Audits a public dataset's own published split |
| `phash_sessions.py` | Recovers clusters by perceptual hash where filenames carry no structure |
| `classify_redundancy.py` | Separates capture-burst redundancy from duplicated files |
| `validate_proxy.py` | Calibrates a perceptual hash against filename-derived sessions; reports pair recall/precision and how much of the true leakage the proxy sees |
| `extract_features.py` | Feature extraction for the calibration (torch.hub / timm / torchvision backends; cls or mean pooling) |
| `embed_proxy.py` | The same calibration for learned representations, with the hash alongside, base-rate correction and degeneracy flagging |
| `leakage_survey.py` | Applies one grouping rule across several datasets and emits one comparable row each; supports dHash or embeddings |
| `leak_detection.py` | Per-test-image agreement between the reference grouping and a content-based proxy: how many leaked images the audit finds, and how many it invents. Supports single, complete and average linkage |
| `grouping_sensitivity.py` | Varies the session-reconstruction rule and reports how far the leakage measurement moves. Needs only `sessions.csv` |
| `label_conflicts.py` | Flags images carrying more than one class label, and camera sequence numbers appearing under two classes |
| `check_contamination.py` | Cross-split contamination check on a built partition |
| `check_provenance.py` | EXIF and filename-provenance survey of the released files |

### Cross-region

| File | Purpose |
|---|---|
| `fix_vietnam_metrics.py` | Recomputes the cross-region macro over an explicit label set, and writes the zero-shot confusion matrices |
| `diagnose_mapping.py` | Per-class zero-shot behaviour; detects an invalid class correspondence |
| `vietnam_indomain.py` | In-domain control on the Vietnamese training split |
| `make_centrecrop.py` | Matched preprocessing control |
| `shrink.py` | Produces the 512 px working copies every reported figure is computed on |
| `malaysia_subset_metrics.py` | Source-side macro over matched class subsets |
| `fill_table_s1.py` | Assembles Supplementary Table S1 from the audit output |
| `fix_crop_control.py` | Recomputes the centre-crop control metrics |
| `fix_robustness.py` | Recomputes the perturbation metrics over an explicit label set |

### Figures and reporting

| File | Purpose |
|---|---|
| `make_paper_figures.py` | Figs. 1–7 as submitted. Fig. 2 is recomputed from `sessions.csv`; the others are drawn from the manuscript tables, each named in a `SOURCE` comment. See the Fig. 5 caveat in `figures/README.md` |
| `make_figures.py` | Earlier figure script, retained for the supplementary set |
| `make_fig3.py` | Alternative path for Fig. 3 (values held inline; update them if Table 3 changes) |
| `make_figS5_similarity.py` | Fig. S5, within-class cosine similarity distributions |
| `export_confusion.py` | Confusion matrices from a checkpoint — **required to finalise Fig. 5** |
| `regen_gradcam.py` | Grad-CAM panels (Fig. S3) |
| `mobilenetv2_lfa.py` | LFA on a second backbone, for the ablation in Supplementary Sect. S1 |

### Paper and results

| Path | Content |
|---|---|
| `paper/manuscript_EI.md` | Manuscript source as submitted |
| `paper/supplementary_material.md` | Supplementary material |
| `results/group_s*`, `results/image_s*` | Per-seed outputs for both partition conditions: per-class metrics, comparison tables, cross-validation, robustness and cross-region results. Tables 6, 7, 9, 10 and 11 and Supplementary Tables S3–S6 aggregate from these |
| `results/group_s42/confusion_matrix.csv` | The matrix behind Fig. 5, so the figure can be checked without a checkpoint |
| `results/grouping_sensitivity.csv` | Table S14, reproducible from `sessions.csv` alone |
| `results/plantvillage_per_class.csv`, `results/plantvillage_summary.txt` | Table S15 and Sect. S16 |
| `figures/` | Figs. 1–7, PDF and PNG at 400 dpi |
| `CHANGELOG.md` | What changed in v2.1.0 and why |

---

## Known issues and conventions

- **Evaluate at the resolution you trained at.** Every figure in the paper is computed on the 512 px
  copies. Running the same checkpoint on the full-resolution originals through the identical
  resize-and-centre-crop pipeline moves held-out macro F1 from 72.0% to 77.6%. Reproductions from
  the archived originals will not match unless you downscale first.
- **Macro averaging is over an explicit label set.** Classes absent from a target evaluation are
  excluded from the average, not entered as zeros. The `scikit-learn` default averages over the
  union of true and predicted labels and reports values 20% lower on the cross-region task.
- **Two instances of the same configuration appear in the paper.** The attention ablation and the
  architecture comparison were run as separate experiments, so EfficientNet-B0 with and without LFA
  were each trained twice from independent random streams. Both are reported; the difference between
  them is used as an estimate of run-to-run variation.
- **LFA is not a claimed improvement, and it is not the smallest-footprint option.** Under
  leakage-free multi-seed evaluation no attention configuration separates from an attention-free
  baseline beyond run-to-run noise: the contrast is 2.1 pp with a 95% confidence interval from −7.4
  to 11.6, while training the same model twice moves the score by 5.1 pp on average and by as much
  as 8.5 pp. The attention-free baseline has 1,281 *fewer* parameters, and the design cannot
  distinguish the two on accuracy. LFA is retained as one option among alternatives this design
  cannot separate. The paper makes no architectural claim; the ablation is Supplementary Sect. S1.
- **An earlier three-seed analysis appeared to show the ranking of attention modules reversing
  between the two protocols. A fourth seed removed it.** The paper reports the retraction. Do not
  cite the reversal, and disregard any earlier version of this README or of the Zenodo record that
  repeats it.
- **Site (orchard) identifiers were never recorded** and cannot be recovered: no released file
  retains a GPS tag, capture timestamp or camera make-and-model tag. Leave-one-site-out validation
  is therefore impossible on this dataset. This is the defect the reporting protocol in the paper
  exists to prevent.

---

## Citation

See `CITATION.cff`. Dataset released under CC BY 4.0; code under the MIT licence.
