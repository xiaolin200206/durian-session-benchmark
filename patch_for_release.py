#!/usr/bin/env python3
"""
patch_for_release.py — bring train.py in line with what is actually released
===========================================================================
The file still carries text from before the session-level rerun: an old paper
title, a statement that the dataset is confidential, a section labelled 5-fold
that runs 3-fold, and a summary line that raises NameError under --no_latency.

Two of these are not cosmetic. The confidentiality note contradicts a CC BY 4.0
Zenodo release, and anyone reading the repository would take it at face value.
The 5-fold label contradicts the paper's own point about Pink_disease bounding
k at 3.

    python patch_for_release.py --file train.py
    python patch_for_release.py --file train.py --dry-run

Every edit is reported. Anything already applied is skipped, so it is safe to
run twice.
"""

import argparse
import shutil
import sys
from pathlib import Path

OLD_DOC = '''Paper  : Agri-EfficientNet: A Lightweight Lesion Focus Attention Framework
         for Durian Disease Diagnosis Under Malaysian Field Conditions
         with Cross-Country Generalization Assessment'''

NEW_DOC = '''Paper  : What a Reported Accuracy Measures: Capture-Session Leakage and
         Cross-Country Transfer in Durian Disease Classification'''

OLD_DATA = '''Dataset:
  Malaysia dataset is not publicly available (commercial confidentiality).
  To request access, open an issue on the repository or contact the maintainers.
  Vietnam dataset: Nguyen et al. (2025), Data in Brief.'''

NEW_DATA = '''Dataset:
  Malaysia dataset: released on Zenodo under CC BY 4.0, with per-image capture
  session identifiers. Group your partitions by the session column; an
  image-level split leaks 79.6% of images on this data and inflates macro F1
  by 12.2 points on average across nine architectures.
  Vietnam dataset: Nguyen et al. (2025), Data in Brief, under its own terms.'''

EDITS = [
    (OLD_DOC, NEW_DOC, 'paper title in the docstring'),
    (OLD_DATA, NEW_DATA, 'dataset availability note'),

    # the pipeline summary says 5-fold; max_usable_k reduces it to 3
    ('  Step 10 — 5-fold cross-validation',
     '  Step 10 — Grouped cross-validation (k is reduced from 5 to 3 because\n'
     '            Pink_disease has only three capture sessions)',
     'pipeline summary, fold count'),

    ("print('STEP 10: 5-Fold Cross-Validation (Disease subset)')",
     "print(f'STEP 10: Grouped Cross-Validation ({CV_MODE} partition)')",
     'step 10 header'),

    ("# 14. 5-FOLD CROSS VALIDATION (Disease subset only)",
     "# 14. GROUPED CROSS VALIDATION\n"
     "#     k is chosen by max_usable_k, not fixed at 5: the rarest class bounds it",
     'section 14 comment'),

    # NameError when --no_latency is passed
    ("print(f'\\n5-Fold CV (Disease subset): F1 = {mean_f1:.2f}% ± {std_f1:.2f}%')\n"
     "print(f'LFA overhead: +{lfa_ms_-base_ms:.2f}ms ({overhead:.1f}%)')",
     "print(f'\\n{k}-fold grouped CV: F1 = {mean_f1:.2f}% ± {std_f1:.2f}%')\n"
     "if not NO_LAT:\n"
     "    print(f'LFA overhead: +{lfa_ms_-base_ms:.2f}ms ({overhead:.1f}%)')",
     'summary line that raised NameError under --no_latency'),

    # leftovers from an earlier version of the dataset that had pest classes
    ("""FOLDER_RENAME = {
    'Leaf_rot(叶腐病': 'Leaf_rot',
    'Stem_Borer(':     'Stem_Borer',
    'Weevil（象鼻虫':  'Weevil',
}""",
     """# Collection folders were named in mixed Chinese and English; this maps them
# to the ASCII class names used throughout. Kept because the Zenodo release
# ships the ASCII names and someone re-running from raw capture folders needs it.
FOLDER_RENAME = {
    'Leaf_rot(叶腐病': 'Leaf_rot',
}""",
     'folder rename map, dropping classes not in the released dataset'),

    ("""d_p = mpatches.Patch(color='#e74c3c',label='Disease (cross-country)')
p_p = mpatches.Patch(color='#3498db',label='Pest (Malaysia only)')
ax.legend(handles=[d_p,p_p])""",
     """ax.legend(handles=[mpatches.Patch(color='#e74c3c',
                   label='Disease (mapped for cross-country)')])""",
     'per-class figure legend, which named a pest category that does not exist'),

    ("type_colors = {'Disease':'#e74c3c','Pest':'#3498db','Background':'#95a5a6','Unknown':'#bdc3c7'}",
     "type_colors = {'Disease':'#e74c3c','Unknown':'#bdc3c7'}",
     'class-type colour map'),

    # deprecated import path
    ('from torch.nn.quantized import FloatFunctional',
     'from torch.ao.nn.quantized import FloatFunctional',
     'FloatFunctional import path'),

    # the usage block predates the session-level protocol
    ('''Usage:
  python train.py --malaysia_data data/malaysia --vietnam_data data/vietnam
  python train.py --retrain   # force retraining''',
     '''Usage:
  python train.py --split_dir clean_split --malaysia_data clean_split \\\\
                  --vietnam_data vietnam --sessions sessions.csv \\\\
                  --cv_mode group --seed 42 --no_latency

  --cv_mode group   keeps a capture session whole (the protocol the paper reports)
  --cv_mode image   the control condition, for reproducing the leakage estimate
  --retrain         force retraining; omit it to resume from checkpoints''',
     'usage block'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default='train.py')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    p = Path(args.file)
    if not p.is_file():
        sys.exit(f'Not found: {p}')
    t = p.read_text(encoding='utf-8')
    original = t

    applied, already, missing = 0, 0, []
    for old, new, label in EDITS:
        if new in t:
            print(f'  already   {label}')
            already += 1
            continue
        n = t.count(old)
        if n == 0:
            missing.append(label)
            print(f'  NOT FOUND {label}')
            continue
        if n > 1:
            print(f'  AMBIGUOUS {label} ({n} matches, skipped)')
            continue
        t = t.replace(old, new)
        applied += 1
        print(f'  patched   {label}')

    print(f'\n{applied} applied, {already} already done, {len(missing)} not found')

    if missing:
        print('\nNot found usually means the file has diverged from the version')
        print('this patch was written against. Check those spots by hand:')
        for m in missing:
            print(f'  - {m}')

    if t == original:
        print('\nNo change needed.')
        return

    if args.dry_run:
        print('\nDry run, nothing written.')
        return

    bak = p.with_suffix(p.suffix + '.prerelease')
    if not bak.exists():
        shutil.copy2(p, bak)
        print(f'\nbackup: {bak.name}')
    p.write_text(t, encoding='utf-8')
    print(f'written: {p}')

    try:
        compile(t, str(p), 'exec')
        print('syntax OK')
    except SyntaxError as e:
        print(f'SYNTAX ERROR at line {e.lineno}: {e.msg}')
        print(f'Restore from {bak.name} before doing anything else.')
        sys.exit(1)

    print('\nOne thing this cannot check: the docstring now says the dataset is')
    print('on Zenodo. Make sure that is true before you push.')


if __name__ == '__main__':
    main()
