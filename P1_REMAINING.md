# P1 — what is left

Everything not on this list is done. Ordered by whether it blocks submission.

---

## Blocking

### 1. Withdrawal confirmed in writing

The previous journal must confirm in writing, and the submission system must
show `Withdrawn`, including the duplicate record. Nothing else here should be
made public first — a preprint or a Zenodo release while a submission is open
is a dual-submission problem rather than a formality.

Sent two requests. If nothing after five working days, write once more to the
editorial office (not the editor), subject line `Withdrawal request — [number]
— follow-up`, three sentences: requested on [dates], please confirm and update
the system status, please close both records.

### 2. Zenodo, then GitHub, then the four placeholders

Order matters: reserve the Zenodo DOI before publishing the record so it can go
into the manuscript, and create the GitHub repository before filling
`CITATION.cff`.

Placeholders remaining in the manuscript:

- Zenodo DOI (Data Availability)
- GitHub URL (Data Availability)
- Acknowledgements — carry over from the previous version and add the grower
  partner
- **Nguyen et al. (2025): the full author list.** A reference list may not use
  "et al."; take it from the Data in Brief record.

### 3. Figures

Every figure predating the session-level rerun contradicts the current tables.
All nine have to be regenerated.

```bash
# training-history files live beside the checkpoints, not in results/
python make_figures.py --results results --ckpt ckpt/group_s42 --out figures

# Figure 3 needs the confusion matrix, which the archive did not keep
python export_confusion.py \
    --ckpt ckpt/group_s42/cmp_agri_efficientnet.pth \
    --split_dir clean_split --out results/group_s42
python make_figures.py --results results --ckpt ckpt/group_s42 --out figures

# Figure 4, Grad-CAM, needs a GPU and the session-level checkpoint
python regen_gradcam.py --ckpt ckpt/group_s42/cmp_agri_efficientnet.pth \
    --split_dir clean_split --out figures
```

Then **read the figures against the tables**. `make_figures.py` hard-codes its
values; if a table has changed since it was written, the figure will be wrong
and nothing will say so.

### 4. Highlights: six drafted, Elsevier allows five

Drop one. The submission package marks which is most expendable.

---

## Worth doing, not blocking

### 5. The fourth image-level seed

Session-level ran seeds 42, 1, 2, 3. Image-level ran 42, 1, 2. Table 4 pairs
four seeds against three, which Section 5.4 discloses but a reviewer will still
raise. About 3.5 hours on a 3090, 7–10 on a laptop GPU.

Everything needed is local:

```bash
python make_image_split.py --src Classication_model_split_512 --dst image_split
python train.py --split_dir image_split --malaysia_data image_split \
    --vietnam_data vietnam --sessions sessions.csv --cv_mode image \
    --ckpt_dir ckpt/image_s3 --save_dir results/image_s3 --seed 3 --no_latency
python aggregate.py --root results --out summary
```

Then update Table 4's image column and the inflation figures. Six numbers move:
the image-level mean and s.d. for each architecture, the inflation per
architecture, the mean inflation, its range, and the sign-test result. Change
them in the manuscript **and** in the header of `make_figures.py`.

### 6. The one unverified cell

Table 7 gives the Vietnamese standard deviation for EfficientNetV2-S as 1.9.
The underlying value is 1.95 to two decimal places and the third decimal was
lost in aggregation, so it could be 2.0. Recompute the three-class scores per
seed from the confusion matrices in `vn_fixed_s42`, `vn_fixed_s1`,
`vn_fixed_s2` and `vn_fixed_s3`. Nothing in the paper turns on it, but it is
the only number not traced to source.

### 7. External dataset audit, into Supplementary S2

The audit of the two Vietnamese datasets is now complete and more interesting
than the version currently in the supplement:

| | VN-A (used here) | VN-B (not used) |
|---|---|---|
| Images in near-duplicate clusters | 284 (10.9%) | 1,429 (26.2%) |
| Redundant beyond one per cluster | 148 | 912 |
| **Byte-identical copies** | 7 (4.7%) | **691 (75.8%)** |
| **Genuinely distinct, same subject** | 141 (95.3%) | 221 (24.2%) |
| Clusters straddling a split | 63 | 217 |
| Their contribution | 6 copies, 63 distinct | 416 copies, 86 distinct |

Two things worth stating. VN-A's redundancy is genuine capture-burst
redundancy, but 123 of its 141 redundant images are in `Leaf_Healthy`, a class
our mapping does not use; across the four mapped classes it is 18 images, 0.7%
of the dataset. VN-B's redundancy is three-quarters duplicated files, which is
a dataset-assembly fault rather than a collection-process one, and is why we
did not use it.

Reproduced with `phash_sessions.py` and `classify_redundancy.py`.

---

## Deliberately not doing

- **Multi-seed robustness and per-class tables.** Currently seed 42 only, and
  Section 5.4 says so. Would need six more runs for a table nobody reads
  closely.
- **A few-shot curve on Vietnamese sessions.** Would turn the closing sentence
  of Section 5.3 from a suggestion into a measurement, but it belongs with the
  cross-dataset work, not here.
- **More datasets.** The redundancy comparison across datasets does not hold up
  on the evidence available: one dataset at 79.6%, one at 0.7% across the mapped
  classes, one measuring something else entirely. Two usable points do not make
  a relationship. Revisit when there are five or six datasets, each classified
  by `classify_redundancy.py` first so that they measure the same quantity.

---

## Order

1. Chase the withdrawal
2. Regenerate the figures (does not depend on anything above)
3. The fourth image-level seed, while waiting
4. Zenodo and GitHub once withdrawal is confirmed, then fill the placeholders
5. Submit
