# Release checklist

Ordered so that each step produces what the next one needs.

## 1. Before anything is made public

- [ ] **Written confirmation from the previous journal that the manuscript is
      withdrawn**, and the submission system showing `Withdrawn`, including the
      duplicate record. Nothing below should happen first: a public preprint or
      dataset while a submission is open is a dual-submission problem.
- [ ] Grower partner has seen the Ethics and Consent statement and agrees with
      how the contributed images are described.

## 2. Zenodo

- [ ] Upload full-resolution originals under class folders
- [ ] Upload `sessions.csv` — this is what makes the release worth making
- [ ] Upload the session-level partition (`clean_split/`) and the 512 px copies
- [ ] Metadata from `zenodo_metadata.json`; fill the four bracketed fields
- [ ] Licence CC BY 4.0
- [ ] Reserve the DOI **before** publishing, so it can go into the manuscript
- [ ] Paste the DOI into: manuscript Data Availability, `README.md`,
      `CITATION.cff`, `zenodo_metadata.json` related_identifiers

## 3. GitHub

- [ ] Create the repository, push the contents of this archive
- [ ] Check `.gitignore` is doing its job: no `.pth`, no `results/`, no images
- [ ] Fill `CITATION.cff` (author, date, repository URL)
- [ ] Tag `v1.0.0`
- [ ] Paste the URL into: manuscript Data Availability, `README.md`,
      Zenodo related_identifiers

## 4. Manuscript

- [ ] Fill the four remaining placeholders: Zenodo DOI, GitHub URL,
      Acknowledgements, and the full author list for Nguyen et al. (2025)
      — a reference list may not use "et al."
- [ ] Verify the one unconfirmed cell: the Vietnamese standard deviation for
      EfficientNetV2-S in Table 7, shown as 1.9. Recompute the three-class
      scores per seed from the confusion matrices in `vn_fixed_s*` and check
      whether the third decimal rounds to 1.9 or 2.0.
- [ ] Regenerate all figures: `python make_figures.py`, then
      `python regen_gradcam.py` against `ckpt/group_s42` for Figure 4.
      Every figure predating the session-level rerun contradicts the tables.
- [ ] Read the figures against the tables once they exist. `make_figures.py`
      hard-codes its values; if a table has changed since, the figure will be
      silently wrong.
- [ ] Highlights: five maximum for Elsevier. Six are drafted; drop one.

## 5. Optional, in order of value

- [ ] A fourth image-level seed, which would make the paired comparison exact.
      About four GPU-hours. The asymmetry is currently disclosed in Section 5.4.
- [ ] Multi-seed robustness and per-class tables. Currently seed 42 only.
- [ ] A few-shot curve on Vietnamese capture sessions, which would turn the
      closing sentence of Section 5.3 from a suggestion into a measurement.

## 6. Submission

- [ ] Cover letter, Highlights, graphical abstract from the submission package
- [ ] Suggested reviewers: three to five, none from your institution
- [ ] Declaration of competing interests, CRediT statement
- [ ] Confirm the target journal's open-access terms and any waiver process
      before acceptance, not after
