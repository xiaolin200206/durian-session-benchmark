#!/usr/bin/env python3
"""
diagnose_mapping.py — is Leaf_Rhizoctonia -> Root_disease a sound mapping?
==========================================================================
The in-domain control showed the Vietnamese labels are learnable. It could not
show that each Vietnamese category means the same thing as the Malaysian class
we assigned it. A foliar category can be perfectly learnable in isolation and
still correspond poorly to a trunk condition.

The zero-shot confusion matrices already written by fix_vietnam_metrics.py
carry that evidence. This script reads them and asks one question per class:

    when a Malaysian model is shown Vietnamese images of class C, where do its
    predictions go, and is class C worse than the others in a way that points
    at the mapping rather than at general transfer failure?

Two signatures to distinguish:

  CONCENTRATED  most of a class's images land on one specific wrong class.
                For Root_disease, mass on the foliar classes says the model
                sees foliage and the organ-level mapping is the problem.

  DIFFUSE       predictions spread roughly in proportion to the model's overall
                output distribution. That is what transfer failure looks like:
                the model has no purchase, not a systematic misreading.

The comparison classes matter as much as the target. Algal and Phomopsis map
directly, organ to organ, so if they scatter too then everything is failing and
Root_disease is not special.

    python diagnose_mapping.py --runs /workspace/vn_fixed_s42 /workspace/vn_fixed_s1 \\
                                      /workspace/vn_fixed_s2 /workspace/vn_fixed_s3
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TARGET = 'Root_disease'      # the mapping under suspicion
DIRECT = ['Algal', 'Phomopsis']   # organ-to-organ mappings, used as controls


def entropy_ratio(row):
    """Normalised entropy of the off-diagonal mass: 0 concentrated, 1 uniform."""
    off = np.array(row, dtype=float)
    if off.sum() <= 0:
        return float('nan')
    p = off / off.sum()
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--exclude', nargs='*', default=['shufflenetv2'],
                    help='models that failed to converge carry no information here')
    args = ap.parse_args()

    mats, classes, present = {}, None, None
    for r in args.runs:
        f = Path(r) / 'vietnam_confusion.json'
        if not f.exists():
            print(f'  [skip] {f} not found')
            continue
        d = json.load(open(f))
        classes = d['class_names']
        present = sorted({classes.index(c) for c in d['composition']})
        for k, cm in d['confusion'].items():
            if k in args.exclude:
                continue
            mats.setdefault(k, []).append(np.array(cm, dtype=float))
    if not mats:
        sys.exit('No confusion matrices found. Run fix_vietnam_metrics.py first.')

    print(f'classes        : {classes}')
    print(f'present in test: {[classes[i] for i in present]}')
    print(f'models         : {len(mats)}   seeds per model: '
          f'{len(next(iter(mats.values())))}\n')

    ti = classes.index(TARGET)
    summed = {k: np.sum(v, axis=0) for k, v in mats.items()}
    grand = np.sum(list(summed.values()), axis=0)

    # ---- per-class destination profile, pooled over models and seeds -------
    print('=' * 78)
    print('WHERE EACH TRUE CLASS GOES  (pooled over models and seeds, row %)')
    print('=' * 78)
    hdr = 'true \\ predicted'.ljust(16) + ''.join(c[:11].rjust(13) for c in classes)
    print(hdr)
    for i in present:
        row = grand[i]
        tot = row.sum()
        cells = ''.join(f'{100*v/tot:>12.1f}%' for v in row)
        mark = '  <-- target' if i == ti else ''
        print(classes[i].ljust(16) + cells + mark)

    # ---- diagonal recall and concentration --------------------------------
    print('\n' + '=' * 78)
    print('PER CLASS: recall, largest wrong destination, spread of the errors')
    print('=' * 78)
    print(f"{'class':<16}{'recall':>9}{'top wrong destination':>34}{'error spread':>15}")
    stats = {}
    for i in present:
        row = grand[i].copy()
        tot = row.sum()
        rec = 100 * row[i] / tot
        off = row.copy(); off[i] = 0
        j = int(np.argmax(off))
        share = 100 * off[j] / off.sum() if off.sum() else float('nan')
        ent = entropy_ratio(np.delete(off, i))
        stats[classes[i]] = dict(recall=rec, top=classes[j], top_share=share, ent=ent)
        print(f'{classes[i]:<16}{rec:>8.1f}%'
              f'{classes[j] + f" ({share:.0f}% of errors)":>34}'
              f'{ent:>14.2f}')
    print('\nerror spread: 0 = all errors on one class, 1 = evenly spread')

    # ---- per model, target class only -------------------------------------
    print('\n' + '=' * 78)
    print(f'{TARGET} ROW BY MODEL  (row %, pooled over seeds)')
    print('=' * 78)
    print('model'.ljust(24) + ''.join(c[:11].rjust(13) for c in classes))
    for k in sorted(summed, key=lambda x: -summed[x][ti][ti]):
        row = summed[k][ti]; tot = row.sum()
        print(k.ljust(24) + ''.join(f'{100*v/tot:>12.1f}%' for v in row))

    # ---- verdict ----------------------------------------------------------
    print('\n' + '=' * 78)
    print('READING')
    print('=' * 78)
    tgt = stats[TARGET]
    ctrl = [stats[c] for c in DIRECT if c in stats]
    mean_ctrl_rec = np.mean([c['recall'] for c in ctrl]) if ctrl else float('nan')
    mean_ctrl_ent = np.mean([c['ent'] for c in ctrl]) if ctrl else float('nan')

    print(f'{TARGET:<14} recall {tgt["recall"]:.1f}%, errors {tgt["ent"]:.2f} spread, '
          f'largest sink {tgt["top"]} ({tgt["top_share"]:.0f}%)')
    print(f'{"direct maps":<14} recall {mean_ctrl_rec:.1f}%, errors {mean_ctrl_ent:.2f} spread '
          f'({", ".join(DIRECT)})')

    worse = tgt['recall'] < mean_ctrl_rec - 10
    concentrated = tgt['top_share'] > 60 or tgt['ent'] < 0.6

    print()
    if worse and concentrated:
        print(f'The target class is both markedly worse than the direct mappings and its')
        print(f'errors concentrate on {tgt["top"]}. That is the signature of a mapping')
        print(f'problem, not of general transfer failure. Report the cross-country result')
        print(f'over the classes that map organ to organ and drop this correspondence,')
        print(f'or justify it explicitly and show this table.')
    elif worse:
        print(f'The target class is worse than the direct mappings but its errors are')
        print(f'spread rather than pooled on one class. That is weak evidence for a')
        print(f'mapping problem. State the asymmetry in the limitations and keep the')
        print(f'class, or report the result both ways.')
    elif concentrated:
        print(f'The target class concentrates its errors on {tgt["top"]} but is not')
        print(f'markedly worse overall. Check whether the direct mappings concentrate')
        print(f'similarly; if they do, this is how the models fail generally.')
    else:
        print(f'The target class behaves like the directly mapped classes: no worse and')
        print(f'no more systematically misdirected. The mapping is not the thing that is')
        print(f'broken, and the cross-country result stands over all four classes.')
    print('\nWhatever the verdict, put this table in the supplement. A reviewer who')
    print('doubts the correspondence will want to see where the predictions went.')


if __name__ == '__main__':
    main()
