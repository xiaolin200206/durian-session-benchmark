# Changelog

## v2.1.0 — pre-submission consistency pass

This release changes **no experimental result**. Every number in the paper is unchanged. What
changed is the set of places where the repository said something the paper does not say, or said it
in a form the paper explicitly rules out.

The five corrections below were found by checking the repository against the manuscript line by
line. Four of them were public over-claims, which matters more for a data-and-code paper than for a
normal one: a referee who opens the repository is reading a second version of the argument.

### Corrected

**1. The repository title over-claimed relative to the paper.**
`README.md` and `CITATION.cff` gave the title as "…the sampling unit of field image data governs
reported accuracy **and cannot be recovered after release**". The manuscript refuses that claim in
as many words: *"We do not claim that no method could succeed. We claim that these do not."* The
title is now the manuscript title: "…the sampling unit governs reported accuracy in field image
data, and content-based auditing does not recover it".

**2. `CITATION.cff` stated a ratio the paper forbids, and stated it wrongly.**
The abstract read: "A near-duplicate audit … detects **4.9% of the leakage** that capture metadata
exposes." Two faults in one clause. The number is wrong — the audit flags 4.6% of test images where
the reference grouping flags 93.9%. And the framing is exactly what Sect. 3.5 of the paper rules
out: a ratio of two aggregate rates is not a detection rate, it can exceed one, and it does not
identify which images were missed. The two figures are now reported side by side, with the reason.

**3. `README.md` contradicted the paper on a matter of fact.**
It said LFA "is retained as the smallest-footprint option among alternatives the design cannot
separate". The attention-free baseline has 1,281 **fewer** parameters. The README now states the
ablation result correctly: 2.1 pp contrast with a 95% CI of [−7.4, 11.6], against 5.1 pp mean
(8.5 pp maximum) between two independent runs of the same configuration.

**4. `zenodo_metadata.json` carried a stale figure and a retracted finding.**
The `notes` field said image-level splits "inflate macro F1 by **11.8** points … and **reverse the
ranking of attention modules**". The current figure is 12.2 points. The ranking reversal was an
artefact of a three-seed analysis and was removed by a fourth seed; the paper reports the
retraction, and the repository was still propagating it to anyone citing the dataset. Both are now
corrected, and the note says explicitly that the earlier values are superseded.

**5. `make_image_split.py` mislabelled its own verification block.**
The `EXPECTED` dictionary holds 446/54/60 — the image-level control — under a comment reading
"Table 1 of the manuscript". Table 1 reports the session-level partition, 446/58/56. Nothing
downstream was wrong, but a referee opening the script to check the control condition would read it
as a mismatch between code and paper. The comment, the printed header and the failure message now
all name the image-level control.

### Disclosed

**6. Scope of the release.**
`make_image_split.py` notes that the source partition holds eleven classes: the five disease classes
the paper reports plus six pest-damage classes added later. This was visible only in a code comment.
It is now stated in `README.md` ("Scope of the release"), in the Zenodo description, and in Sect. 3.2
of the manuscript.

### Added

- `results/grouping_sensitivity.csv` — Table S14 of the paper, the one result in the paper that is
  reproducible from `sessions.csv` alone, with no images and no GPU. Shipping it means a referee can
  verify a headline claim in under a minute.
- `make_paper_figures.py` — regenerates Figs. 1–7 as submitted. Fig. 2 is recomputed from
  `sessions.csv`; every other figure carries a `SOURCE` comment naming the table its values come
  from.
- `figures/Fig1`–`Fig7` in PDF and PNG at 400 dpi.
- `paper/manuscript_EI.md` and `paper/supplementary_material.md`.
- `CHANGELOG.md`, this file.

### Removed

- `figures/superseded/` (13 files, ~3 MB). It held the pre-session-rerun figure set under the old
  numbering, with values that no longer matched the tables. `figures/README.md` had already said
  "do not ship those files"; they were still being shipped.

### PlantVillage measurement (completed)

Against the leaf identifiers Mohanty et al. (2016) recorded and released, on the covered subset of
**40,490 images across 29 classes**:

- **7,524 physical leaves**, 5.38 images per leaf, median 4, maximum 33
- **99.6% ± 0.2** of test images share a leaf with training under a stratified image-level split
  (10 seeds). The durian equivalent is 96.7%.
- Coverage: 29 of 38 classes, 40,490 of 54,306 images, against the 41,112 Mohanty et al. report
  having mapped. The uncovered classes are excluded from the denominator.

16 of the 29 classes hold exactly 4.00 images per leaf, which raised the possibility of dihedral
transforms rather than separate photographs — the duplicated-file versus capture-burst distinction
that Table 4 turns on. `classify_redundancy.py` cannot settle it, because a rotation has entirely
different pixels and is scored as distinct. A dedicated dihedral test over 2,126 within-leaf pairs
from one uniform class and one non-uniform control found **0 transform duplicates**, with the two
difference distributions indistinguishable (medians 29.65 and 28.15). The four images per leaf are
four photographs. Results in `results/plantvillage_summary.txt`, per class in
`results/plantvillage_per_class.csv`, written up as Table S15 and Sect. S16 of the supplement.

### Added after the first consistency pass

- `plantvillage_leafmap.py` — converts PlantVillage's **recorded** leaf identifiers
  (`leaf-map.json`, public at `spMohanty/PlantVillage-Dataset`) into this repository's
  `cls,file,session` manifest schema, so `grouping_sensitivity.py`, `leak_detection.py`,
  `session_utils.load_sessions` and `train.py --cv_mode` all run on it unchanged. Run from the
  leaf map alone it reports 42,520 (class, key) pairs across 7,946 leaf groups, 5.35 images per
  leaf, and **99.6% ± 0.2** of test images sharing a physical leaf with training under a
  stratified image-level split. That is a measurement against a grouping recorded at collection,
  not against a proxy, on roughly seventy-five times the number of images in the durian dataset.
  Run it with `--root` against the image tree before quoting any of those figures in the paper.

### Outstanding

These are not defects of the repository but work the paper still needs. They are listed here so the
list lives with the code.

1. **Fig. 5 has two undetermined cells.** Table 8's per-class metrics fix 23 of the 25 confusion
   cells; the single Phomopsis error and the single pink-disease error cannot be assigned between
   the algal and leaf-rot columns from per-class metrics alone. They are marked with an asterisk.
   Replace them from `export_confusion.py` before submission.
2. **Per-seed result CSVs are not in the repository.** See `results/README.md`.
3. **`leakage_survey.py` has still not been run across the five or six datasets its own docstring
   names as the condition for the survey argument.** Less urgent than it was: the PlantVillage
   measurement below is against a *recorded* grouping and is not subject to the proxy's
   under-detection, so it carries the generality argument on its own. The survey would add
   breadth, not validity.
4. **Supplementary Tables S1–S13 are unfilled.** The structure, captions and cross-references are in
   `paper/supplementary_material.md`, and each slot names the script that produces it.

## v2.0.0 and earlier

Session-level rerun, figure renumbering, and the recoverability analysis. No changelog was kept.
