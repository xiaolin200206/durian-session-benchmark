#!/usr/bin/env python3
"""
patch_runtime_strings.py — the strings a user actually sees
===========================================================
patch_for_release.py fixed the docstring. Three things it did not reach are in
the executable body, which is where they do damage:

  1. The FileNotFoundError raised when the dataset is missing still tells the
     reader to open an issue to request access. That is the message anyone
     hitting the error will read, and it contradicts the Zenodo release more
     directly than the docstring did.

  2. A figure title says "Disease & Pest Classification". There are no pest
     classes in the released dataset.

  3. Display strings still say Agri-EfficientNet, the name the paper dropped
     once the ablation showed the module confers nothing.

The third is deliberately partial. MODEL_CONFIGS keys and checkpoint filenames
keep agri_efficientnet, because renaming them would break every saved
checkpoint and every results CSV already written. Only the human-readable
labels change; the README explains the mapping.

    python patch_runtime_strings.py --file train.py --dry-run
    python patch_runtime_strings.py --file train.py
"""

import argparse
import shutil
import sys
from pathlib import Path

EDITS = [
    # the error a user actually hits
    ('''        f"Malaysia data not found at: {MY_DATA}\\n"
        "Please download the dataset and specify --malaysia_data <path>\\n"
        "To request access: open an issue on the repository or contact the maintainers."''',
     '''        f"Malaysia data not found at: {MY_DATA}\\n"
        "The dataset is on Zenodo under CC BY 4.0; download it and pass\\n"
        "--malaysia_data <path>. Use the 512 px copies, not the originals:\\n"
        "the two resampling paths to 224 px differ and the reported numbers\\n"
        "are computed on the 512 px set."''',
     'dataset-missing error message'),

    ("fig.suptitle('Model Comparison — Durian Disease & Pest Classification',fontsize=14,fontweight='bold')",
     "fig.suptitle('Architecture comparison, session-level partition',fontsize=14,fontweight='bold')",
     'figure title naming a pest category'),

    # display labels only; keys and filenames stay
    ("    ('Agri-EfficientNet + LFA (Ours)', 'lfa'),",
     "    ('EfficientNet-B0 + LFA (ours)',   'lfa'),",
     'ablation label'),

    ("    ('agri_efficientnet','Agri-EfficientNet (Ours)'),",
     "    ('agri_efficientnet','EfficientNet-B0 + LFA (ours)'),",
     'comparison label'),

    ("    'Agri-EfficientNet (LFA)':   (CKPT_DIR/'cmp_agri_efficientnet.pth', 'agri_efficientnet'),",
     "    'EfficientNet-B0 + LFA':     (CKPT_DIR/'cmp_agri_efficientnet.pth', 'agri_efficientnet'),",
     'robustness label'),

    ("rob_colors = {'Agri-EfficientNet (LFA)':'#e67e22',",
     "rob_colors = {'EfficientNet-B0 + LFA':'#e67e22',",
     'robustness colour key'),

    ("ax.set_ylabel('F1 (%)'); ax.set_title('Per-Class F1 — Agri-EfficientNet (LFA)')",
     "ax.set_ylabel('F1 (%)'); ax.set_title('Per-class F1, ablation instance')",
     'per-class figure title'),

    ("ax.set_title('Confusion Matrix (Normalized) — Agri-EfficientNet (LFA)')",
     "ax.set_title('Normalised confusion matrix, ablation instance')",
     'confusion matrix title'),

    ("""    fig.suptitle('Grad-CAM: Original | ResNet-50 (Baseline) | Agri-EfficientNet LFA (Ours)',
                 fontsize=11, fontweight='bold')""",
     """    fig.suptitle('Grad-CAM: original, ResNet-50, and EfficientNet-B0 + LFA',
                 fontsize=11, fontweight='bold')""",
     'Grad-CAM figure title'),

    ("        axes[2,col].set_title('LFA (Ours)',fontsize=7); axes[2,col].axis('off')",
     "        axes[2,col].set_title('EfficientNet-B0 + LFA',fontsize=7); axes[2,col].axis('off')",
     'Grad-CAM row label'),

    ("""                          'Agri-EfficientNet (LFA)', name)""",
     """                          'EfficientNet-B0 + LFA', name)""",
     'McNemar comparison label'),

    ("""                          'Agri-EfficientNet (LFA)', 'ResNet-50')""",
     """                          'EfficientNet-B0 + LFA', 'ResNet-50')""",
     'McNemar ResNet comparison label'),

    ("print(f'\\nMalaysia Internal Test — Agri-EfficientNet (LFA):')",
     "print(f'\\nMalaysia held-out test, EfficientNet-B0 + LFA:')",
     'final summary heading'),

    ("parser = argparse.ArgumentParser(description='Agri-EfficientNet: Durian Disease Classification')",
     "parser = argparse.ArgumentParser(\n"
     "    description='Session-level evaluation of durian disease classifiers. '\n"
     "                'Group partitions by capture session; see README.')",
     'argparse description'),

    ('''"""
Agri-EfficientNet: Durian Disease Classification
================================================''',
     '''"""
Durian disease classification under session-level evaluation
============================================================''',
     'module title line'),

    # the ABLATION_CONFIGS label is used as a dict key for the checkpoint name,
    # so make sure the CSV column stays readable
    ("DISEASE_CLASSES = list(class_names)  # All classes are disease in this version",
     "DISEASE_CLASSES = list(class_names)  # all five classes are diseases",
     'stale comment'),
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

    applied = already = 0
    missing = []
    for old, new, label in EDITS:
        if new in t:
            print(f'  already   {label}'); already += 1; continue
        n = t.count(old)
        if n == 0:
            missing.append(label); print(f'  NOT FOUND {label}'); continue
        if n > 1:
            print(f'  AMBIGUOUS {label} ({n} matches, skipped)'); continue
        t = t.replace(old, new); applied += 1
        print(f'  patched   {label}')

    print(f'\n{applied} applied, {already} already done, {len(missing)} not found')

    # what is deliberately left alone
    keys = t.count("'agri_efficientnet'")
    files = t.count('cmp_agri_efficientnet.pth') + t.count('abl_lfa.pth')
    print(f'\nLeft unchanged on purpose: {keys} uses of the agri_efficientnet key')
    print(f'and {files} checkpoint filenames. Renaming these would break every')
    print('saved checkpoint and every results CSV already written. The README')
    print('records that the key and the paper name refer to the same model.')

    if t == original:
        print('\nNo change needed.')
        return
    if args.dry_run:
        print('\nDry run, nothing written.')
        return

    bak = p.with_suffix(p.suffix + '.strings')
    if not bak.exists():
        shutil.copy2(p, bak); print(f'\nbackup: {bak.name}')
    p.write_text(t, encoding='utf-8')
    print(f'written: {p}')
    try:
        compile(t, str(p), 'exec')
        print('syntax OK')
    except SyntaxError as e:
        print(f'SYNTAX ERROR line {e.lineno}: {e.msg}')
        print(f'Restore from {bak.name}.')
        sys.exit(1)

    print('\nNote: the ablation and robustness CSVs are keyed on these labels,')
    print('so results written before this patch use the old names. That is')
    print('cosmetic, but do not concatenate old and new CSVs without checking.')


if __name__ == '__main__':
    main()
