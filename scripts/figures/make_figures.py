#!/usr/bin/env python3
"""
make_figures.py — regenerate every figure in the manuscript
===========================================================
Every figure produced before the session-level rerun was computed under the
image-level split and contradicts the current tables. None of them can be
reused. This regenerates all nine from the archive.

    python make_figures.py --results results --sessions sessions.csv --out figures

Produces, at 300 dpi in both PNG and PDF:

    Fig 1  the sampling hierarchy: image / session / tree / orchard / region
    Fig 2  session structure: class composition, session-size distribution,
           and provenance by class (reads sessions.csv)
    Fig 3  paired image-level vs session-level macro F1 (Table 5)
    Fig 4  confusion matrix, session-level test set
    Fig 5  accuracy against parameter count
    Fig 6  cross-region, three foliar correspondences, with the crop control
    Fig 7  zero-shot per-class behaviour beside the in-domain control
    Fig S1 two-stage training curves
    Fig S2 macro F1 under the five perturbations

Figure 4 needs Grad-CAM overlays on specific images and a loaded checkpoint, so
it is produced by regen_gradcam.py against ckpt/group_s42, not here. Everything
else is drawn from CSVs and from the aggregate values below.

The aggregates are written out explicitly rather than recomputed, so that the
figures and the manuscript cannot drift apart: if a number changes, it changes
in one place and both follow. They match the manuscript tables exactly.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# --------------------------------------------------------------------------
# values as they appear in the manuscript tables
# --------------------------------------------------------------------------
CLASSES = ['Algal', 'Leaf_rot', 'Phomopsis', 'Pink_disease', 'Root_disease']
IMAGES = [162, 153, 157, 10, 78]
SESSIONS = [21, 27, 14, 3, 8]

# Table 4: image-level, session-level, inflation
PAIRED = [
    ('EfficientNetV2-S', 93.4, 1.5, 88.5, 3.1),
    ('ResNet-101', 94.3, 1.1, 85.3, 5.2),
    ('ConvNeXt-Tiny', 97.3, 2.3, 81.8, 3.6),
    ('EfficientNet-B0 + LFA', 82.1, 3.1, 76.1, 3.7),
    ('VGG-16', 92.9, 6.0, 76.0, 6.9),
    ('ResNet-50', 88.3, 3.2, 75.2, 2.3),
    ('MobileNetV3-Large', 92.4, 1.2, 74.0, 2.7),
    ('EfficientNet-B0', 83.8, 1.9, 73.0, 2.4),
    ('MobileNetV2', 87.9, 2.0, 72.9, 4.2),
]
# Table 3 params (M)
PARAMS = {'EfficientNetV2-S': 20.18, 'ResNet-101': 42.51, 'ConvNeXt-Tiny': 27.82,
          'EfficientNet-B0 + LFA': 4.02, 'VGG-16': 134.28, 'ResNet-50': 23.52,
          'MobileNetV3-Large': 4.21, 'EfficientNet-B0': 4.01, 'MobileNetV2': 2.23}
# Table 7: Malaysia 3-class, Vietnam 3-class
CROSS3 = [
    ('EfficientNetV2-S', 80.8, 5.2, 42.9, 1.9),
    ('ResNet-101', 78.5, 5.1, 42.3, 1.3),
    ('EfficientNet-B0 + LFA', 73.3, 4.5, 32.8, 2.1),
    ('MobileNetV2', 72.5, 8.4, 30.2, 5.2),
    ('VGG-16', 71.8, 11.7, 44.7, 3.4),
    ('ConvNeXt-Tiny', 69.8, 6.0, 48.9, 2.0),
    ('EfficientNet-B0', 67.2, 4.9, 28.4, 2.0),
    ('ResNet-50', 67.2, 3.2, 34.5, 5.8),
    ('MobileNetV3-Large', 66.2, 5.3, 35.5, 4.9),
]
CHANCE3 = 32.5
# Table 9 centre-crop control, three classes
CROP = [('EfficientNet-B0 + LFA', 73.3, 32.8, 65.5, 45.8),
        ('EfficientNet-B0', 67.2, 28.4, 66.6, 48.9),
        ('MobileNetV2', 72.5, 30.2, 69.8, 40.8)]
# Table 5 per-class, seed 42
PERCLASS = [('Algal', 83.3, 31.2, 45.5), ('Leaf_rot', 55.6, 100.0, 71.4),
            ('Phomopsis', 100.0, 93.3, 96.6), ('Pink_disease', 100.0, 50.0, 66.7),
            ('Root_disease', 100.0, 100.0, 100.0)]
# Table 6 robustness, seed 42
ROBUST = [('Clean', 72.0, 72.3, 78.4), ('Gaussian\nnoise', 47.1, 42.9, 84.1),
          ('Low\nbrightness', 78.5, 79.7, 82.6), ('High\nbrightness', 77.8, 69.3, 75.6),
          ('Motion\nblur', 70.9, 69.6, 76.6), ('Occlusion\n30%', 73.6, 77.7, 78.9)]
# Table 11 zero-shot per class, pooled
POOLED = [('Algal', 84.2, 49.8, 21.9), ('Leaf_rot', 43.3, 32.9, 39.4),
          ('Phomopsis', 13.7, 10.5, 19.7), ('Root_disease', 1.5, 1.1, 19.1),
          ('Pink_disease', None, 5.7, 0.0)]
# Table 10 in-domain, seed 2
INDOMAIN = [('Algal', 97.1), ('Leaf_rot', 88.9), ('Phomopsis', 95.2), ('Root_disease', 98.4)]

BLUE, RED, GREY, GREEN = '#3b6ea5', '#c0392b', '#8a8a8a', '#2e7d52'
plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 300, 'savefig.bbox': 'tight'})


def save(fig, out, name):
    for ext in ('png', 'pdf'):
        fig.savefig(Path(out) / f'{name}.{ext}')
    plt.close(fig)
    print(f'  {name}')


# --------------------------------------------------------------------------
def fig1(out):
    """The sampling hierarchy. Conceptual: no data dependency."""
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    levels = [
        dict(y=8.6, w=9.2, label='Production region', n='e.g. Malaysia vs Vietnam',
             q='Can this model be exported?', measured=True),
        dict(y=6.7, w=7.4, label='Orchard', n='the purchase / deployment unit',
             q='How well in a neighbour\u2019s orchard?', measured=False),
        dict(y=4.8, w=5.6, label='Tree', n='the management unit',
             q='(natural observation unit; approximated below)', measured=False),
        dict(y=2.9, w=3.8, label='Capture session', n='one specimen, one visit',
             q='How well on a new specimen, familiar orchard?', measured=True),
        dict(y=1.0, w=2.0, label='Image', n='one measurement',
             q='(answers no question a practitioner asks)', measured=True),
    ]
    for lv in levels:
        x0 = 5 - lv['w'] / 2
        ax.add_patch(FancyBboxPatch(
            (x0, lv['y'] - 0.525), lv['w'], 1.05,
            boxstyle='round,pad=0.02,rounding_size=0.12', linewidth=1.4,
            edgecolor=(BLUE if lv['measured'] else GREY),
            facecolor=('#eaf1f8' if lv['measured'] else '#f3f3f3'), zorder=2))
        ax.text(5, lv['y'] + 0.16, lv['label'], ha='center', va='center',
                fontsize=10.5, fontweight='bold', zorder=3)
        ax.text(5, lv['y'] - 0.24, lv['n'], ha='center', va='center',
                fontsize=8, style='italic', color='#444444', zorder=3)
        qx = x0 + lv['w'] + 0.35
        ax.plot([x0 + lv['w'], qx - 0.05], [lv['y'], lv['y']],
                color='#999999', linewidth=0.8, zorder=1)
        ax.text(qx, lv['y'], lv['q'], ha='left', va='center', fontsize=8,
                color=('#333333' if lv['measured'] else RED),
                style=('normal' if lv['measured'] else 'italic'))
    for a, b in zip(levels[:-1], levels[1:]):
        ax.annotate('', xy=(5, a['y'] - 0.545), xytext=(5, b['y'] + 0.545),
                    arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=1.3,
                                    shrinkA=0, shrinkB=0))
    ax.annotate('missing: not recorded\nat collection, and not\nrecoverable afterwards',
                xy=(5 - levels[1]['w'] / 2, levels[1]['y']), xytext=(0.15, 6.7),
                fontsize=7.3, color=RED, ha='left', va='center',
                arrowprops=dict(arrowstyle='-', color=RED, lw=0.9,
                                connectionstyle='arc3,rad=-0.15'))
    ax.text(5, 9.75,
            'image  \u2282  capture session  \u2282  tree  \u2282  orchard  \u2282  production region',
            ha='center', va='center', fontsize=9.3, color='#222222')
    ax.legend(handles=[
        Line2D([0], [0], marker='s', color='none', markerfacecolor='#eaf1f8',
               markeredgecolor=BLUE, markersize=11, label='scale measured in this study'),
        Line2D([0], [0], marker='s', color='none', markerfacecolor='#f3f3f3',
               markeredgecolor=GREY, markersize=11, label='scale not measured')],
        loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=2,
        frameon=False, fontsize=7.6)
    save(fig, out, 'Fig1_sampling_hierarchy')


# --------------------------------------------------------------------------
def fig2_class_panel(out, sessions_csv='sessions.csv'):
    """Session structure: class composition, session-size distribution,
    and provenance by class. Panels b and c read sessions.csv."""
    import csv as _csv
    import collections as _c

    rows = list(_csv.DictReader(open(sessions_csv, newline='', encoding='utf-8-sig')))
    sess_count = _c.Counter(r['session'] for r in rows)
    sizes = sorted(sess_count.values(), reverse=True)
    TYPES = ['burst', 'video', 'whatsapp', 'single']
    type_by_class = {c: _c.Counter() for c in CLASSES}
    for r in rows:
        type_by_class[r['cls']][r['session'].split(':')[0]] += 1

    fig = plt.figure(figsize=(9.6, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                          hspace=0.55, wspace=0.32)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    x = np.arange(len(CLASSES)); w = 0.38
    axA.bar(x - w/2, IMAGES, w, color=BLUE, label='Images')
    axA.bar(x + w/2, SESSIONS, w, color=RED, label='Capture sessions')
    for i, (im, se) in enumerate(zip(IMAGES, SESSIONS)):
        axA.text(i - w/2, im + 3, str(im), ha='center', fontsize=7.5)
        axA.text(i + w/2, se + 3, str(se), ha='center', fontsize=7.5, color=RED)
    axA.set_xticks(x); axA.set_xticklabels(CLASSES, rotation=20, ha='right', fontsize=7.8)
    axA.set_ylabel('Count'); axA.set_ylim(0, max(IMAGES) * 1.2)
    axA.legend(frameon=False, loc='upper right', fontsize=7.3)
    axA.set_title('a  Class composition (n = 560 images, 73 sessions)',
                  fontsize=8.6, loc='left')

    axB.bar(np.arange(1, len(sizes) + 1), sizes, width=0.9, color=GREEN)
    med = float(np.median(sizes))
    axB.axhline(med, color=GREY, lw=1, ls='--')
    axB.text(len(sizes) * 0.98, med + 1.5, f'median = {int(med)}',
             ha='right', fontsize=7.3, color=GREY)
    top10 = sum(sizes[:10])
    axB.annotate(f'10 largest sessions\n= {top10}/{sum(sizes)} images '
                 f'({top10/sum(sizes)*100:.1f}%)',
                 xy=(5.5, sizes[4]), xytext=(20, sizes[4] + 18),
                 fontsize=7.3, color=GREEN,
                 arrowprops=dict(arrowstyle='-', color=GREEN, lw=0.8))
    axB.set_xlabel('Session, ranked by size'); axB.set_ylabel('Images in session')
    axB.set_title('b  Distribution of session sizes (73 sessions)',
                  fontsize=8.6, loc='left')

    colors = {'burst': BLUE, 'video': RED, 'whatsapp': GREEN, 'single': GREY}
    bottoms = np.zeros(len(CLASSES)); xC = np.arange(len(CLASSES))
    LABEL_MIN = 10
    small = []
    for ty in TYPES:
        vals = np.array([type_by_class[c].get(ty, 0) for c in CLASSES])
        axC.bar(xC, vals, bottom=bottoms, color=colors[ty], width=0.55,
                label=f'{ty} ({int(vals.sum())})')
        for i, v in enumerate(vals):
            if v >= LABEL_MIN:
                axC.text(xC[i], bottoms[i] + v/2, str(int(v)), ha='center',
                         va='center', fontsize=7.4, color='white')
            elif v > 0:
                small.append((int(xC[i]), bottoms[i] + v/2, int(v), ty))
        bottoms += vals
    by_class = _c.defaultdict(list)
    for xi, yi, v, ty in small:
        by_class[xi].append((yi, v, ty))
    for xi, items in by_class.items():
        items.sort()
        axC.annotate(', '.join(f'{ty} {v}' for _, v, ty in items),
                     xy=(xi, items[0][0]), xytext=(xi, bottoms[xi] + 14),
                     ha='center', va='bottom', fontsize=6.9, color='#333333',
                     arrowprops=dict(arrowstyle='-', color='#999999', lw=0.7))
    axC.set_xticks(xC); axC.set_xticklabels(CLASSES, fontsize=8.2)
    axC.set_ylabel('Images'); axC.set_ylim(0, max(bottoms) * 1.28)
    axC.set_title('c  Provenance by class \u2014 all video-derived frames fall in one class',
                  fontsize=8.6, loc='left')
    axC.legend(frameon=False, ncol=4, fontsize=7.3, loc='upper right')
    vb = type_by_class['Phomopsis']['burst']; vv = type_by_class['Phomopsis']['video']
    axC.annotate(f'{vv} video frames,\nall Phomopsis', xy=(2, vb + vv/2),
                 xytext=(3.15, 110), fontsize=7.3, color=RED,
                 arrowprops=dict(arrowstyle='-', color=RED, lw=0.8,
                                 connectionstyle='arc3,rad=0.2'))
    save(fig, out, 'Fig2_session_structure')


def fig3(out):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    names = [p[0] for p in PAIRED]
    y = np.arange(len(names))[::-1]
    img = np.array([p[1] for p in PAIRED]); imgsd = np.array([p[2] for p in PAIRED])
    ses = np.array([p[3] for p in PAIRED]); sessd = np.array([p[4] for p in PAIRED])
    for i, yy in enumerate(y):
        ax.plot([ses[i], img[i]], [yy, yy], color=GREY, lw=1, zorder=1)
        ax.annotate(f'+{img[i]-ses[i]:.1f}', ((ses[i]+img[i])/2, yy + 0.28),
                    ha='center', fontsize=7.5, color=GREY)
    ax.errorbar(ses, y, xerr=sessd, fmt='o', color=BLUE, ms=5, capsize=2,
                lw=1, label='Session-level', zorder=3)
    ax.errorbar(img, y, xerr=imgsd, fmt='s', color=RED, ms=5, capsize=2,
                lw=1, label='Image-level', zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Macro F1 (%)'); ax.set_xlim(60, 102)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.16),
              ncol=2, fontsize=8)
    ax.set_ylim(-0.9, len(names) - 0.2)
    ax.set_title('Mean inflation +12.2 points (sign test p = 0.004)',
                 fontsize=9, loc='left')
    save(fig, out, 'Fig3_paired_partition')


def fig4(results, out):
    """Confusion matrix; read from the run if present, else drawn from Table 5."""
    cm = None
    p = Path(results) / 'group_s42' / 'confusion_matrix.csv'
    if p.exists():
        cm = np.loadtxt(p, delimiter=',')
    if cm is None:
        # Fall back to the matrix implied by Table 6 (seed 42, ablation
        # instance). Rows are true classes in CLASSES order, supports
        # 16/15/15/2/8. Reproduces the published figure exactly; regenerate
        # from a checkpoint with export_confusion.py if you prefer.
        cm = np.array([[5, 11, 0, 0, 0],
                       [0, 15, 0, 0, 0],
                       [1,  0, 14, 0, 0],
                       [0,  1, 0, 1, 0],
                       [0,  0, 0, 0, 8]], dtype=float)
        print('  Fig4: no confusion_matrix.csv found; using the Table 6 values.')
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    norm = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f'{norm[i,j]:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if norm[i, j] > 0.5 else 'black')
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    fig.colorbar(im, ax=ax, fraction=0.045)
    save(fig, out, 'Fig4_confusion_matrix')


def fig5(out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.5))
    infl = {n: round(i - s, 1) for n, i, _, s, _ in PAIRED}

    for name, _, _, ses, sd in PAIRED:
        p = PARAMS[name]
        light = p <= 4.5
        a.errorbar(p, ses, yerr=sd, fmt='o', ms=6, capsize=2, lw=1,
                   color=GREEN if light else BLUE, zorder=3)
    # four lightweight models sit within 2 M of each other. Rather than fight
    # the overlap with offsets, their labels are stacked to the right of the
    # cluster at fixed heights and joined by short leaders.
    light = sorted([r for r in PAIRED if PARAMS[r[0]] <= 4.5],
                   key=lambda r: -r[3])
    heavy = [r for r in PAIRED if PARAMS[r[0]] > 4.5]
    for name, _, _, ses, _ in heavy:
        a.annotate(name, (PARAMS[name], ses), textcoords='offset points',
                   xytext=(9, 4), fontsize=7)
    for i, (name, _, _, ses, _) in enumerate(light):
        a.annotate(name, (PARAMS[name], ses), xytext=(7.0, 69.5 - i * 1.9),
                   fontsize=7, ha='left', va='center',
                   arrowprops=dict(arrowstyle='-', lw=0.6, color=GREY,
                                   shrinkA=0, shrinkB=3))
    a.set_xscale('log'); a.set_xlabel('Parameters (M, log scale)')
    a.set_ylabel('Macro F1 (%), session-level')
    a.set_title('Accuracy against capacity', fontsize=9, loc='left')
    a.set_ylim(62, 97); a.set_xlim(1.6, 400)
    a.grid(alpha=0.25, ls=':')

    for name, _, _, _, _ in PAIRED:
        p = PARAMS[name]
        b.scatter(p, infl[name], s=34,
                  color=GREEN if p <= 4.5 else BLUE, zorder=3)
    b.set_xscale('log'); b.set_xlabel('Parameters (M, log scale)')
    b.set_ylabel('Inflation under an image-level split (pp)')
    b.set_title('Capacity does not predict susceptibility\n(Pearson r = 0.29, p = 0.45)',
                fontsize=8.5, loc='left')
    b.grid(alpha=0.25, ls=':')
    b.set_ylim(0, 22)
    for name in ['MobileNetV3-Large', 'EfficientNetV2-S']:
        b.annotate(name, (PARAMS[name], infl[name]), textcoords='offset points',
                   xytext=(6, -2), fontsize=7)
    fig.tight_layout()
    save(fig, out, 'Fig5_accuracy_capacity')


def fig6(out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.5),
                               gridspec_kw={'width_ratios': [2.1, 1]})
    names = [c[0] for c in CROSS3]
    x = np.arange(len(names)); w = 0.38
    my = [c[1] for c in CROSS3]; mysd = [c[2] for c in CROSS3]
    vn = [c[3] for c in CROSS3]; vnsd = [c[4] for c in CROSS3]
    a.bar(x - w/2, my, w, yerr=mysd, capsize=2, color=BLUE, label='Malaysia')
    a.bar(x + w/2, vn, w, yerr=vnsd, capsize=2, color=RED, label='Vietnam')
    a.axhline(CHANCE3, color='k', ls='--', lw=1)
    a.annotate(f'chance {CHANCE3}%', (-0.45, CHANCE3 + 2.0),
               ha='left', fontsize=7.5)
    a.set_xticks(x); a.set_xticklabels(names, rotation=35, ha='right', fontsize=7.5)
    a.set_ylabel('Macro F1 (%), three foliar correspondences')
    a.legend(frameon=False, fontsize=8); a.set_ylim(0, 95)
    a.set_title('Zero-shot transfer, like-for-like classes', fontsize=9, loc='left')

    lab = [c[0] for c in CROP]
    go = [c[1] - c[2] for c in CROP]; gc = [c[3] - c[4] for c in CROP]
    y = np.arange(len(lab))
    b.barh(y + 0.18, go, 0.34, color=GREY, label='original')
    b.barh(y - 0.18, gc, 0.34, color=GREEN, label='centre-crop matched')
    b.set_yticks(y); b.set_yticklabels(lab, fontsize=7.5)
    b.set_xlabel('Transfer gap (points)')
    b.set_xlim(0, 46)
    b.legend(frameon=False, fontsize=8, loc='upper center',
             bbox_to_anchor=(0.5, -0.22), ncol=2)
    b.set_title('Preprocessing accounts for 45%\nof the gap (40.5 → 22.1)',
                fontsize=8.5, loc='left')
    fig.tight_layout()
    save(fig, out, 'Fig6_cross_country')


def fig7(out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=False)
    lab = [p[0] for p in POOLED]
    y = np.arange(len(lab))[::-1]
    pred = [p[2] for p in POOLED]; true = [p[3] for p in POOLED]
    a.barh(y + 0.18, true, 0.34, color=GREY, label='share of labels')
    a.barh(y - 0.18, pred, 0.34, color=RED, label='share of predictions')
    a.set_yticks(y); a.set_yticklabels(lab, fontsize=8)
    a.set_xlabel('% of the 320 Vietnamese test images')
    a.legend(frameon=False, fontsize=8, loc='lower right', bbox_to_anchor=(1.0, -0.02))
    a.set_title('Zero-shot: the Root_disease output is inactive', fontsize=8.5, loc='left')
    a.annotate('1.1% predicted\nagainst 19.1%\nof labels', xy=(2.5, y[3]),
               xytext=(30, y[3] + 0.9), fontsize=7.5, va='center', ha='left',
               arrowprops=dict(arrowstyle='->', lw=0.8, color='k'))
    a.set_xlim(0, 58)

    names = [i[0] for i in INDOMAIN]
    zs = [p[1] for p in POOLED if p[0] in names]
    idm = [i[1] for i in INDOMAIN]
    xx = np.arange(len(names)); w = 0.38
    b.bar(xx - w/2, zs, w, color=RED, label='zero-shot (MY-trained)')
    b.bar(xx + w/2, idm, w, color=GREEN, label='in-domain (VN-trained)')
    b.set_xticks(xx); b.set_xticklabels(names, rotation=25, ha='right', fontsize=7.5)
    b.set_ylabel('Recall (%)'); b.set_ylim(0, 105)
    b.legend(frameon=False, fontsize=8)
    b.set_title('The same 61 Root_disease images:\n1.5% against 98.4%', fontsize=8.5, loc='left')
    fig.tight_layout()
    save(fig, out, 'Fig7_mapping_diagnosis')


def figS1(results, out, ckpt=None):
    hist = None
    for base in [Path(results) / 'group_s42', Path(ckpt)] if ckpt else [Path(results) / 'group_s42']:
        for cand in ['cmp_agri_efficientnet_hist.json', 'abl_lfa_hist.json']:
            f = base / cand
            if f.exists():
                hist = json.load(open(f)); print(f'    history from {f}'); break
        if hist: break
    if hist is None:
        print('  Fig S1: no *_hist.json found. They are written beside the')
        print('          checkpoints, so pass --ckpt ckpt/group_s42')
        return
    if 'abl' in Path(f).stem:
        print('  !! Fig S1 read the ABLATION history, but the title and the')
        print('     caption both say comparison instance. Point --ckpt at the')
        print('     comparison run, or change the title and the caption together.')
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ep = np.arange(1, len(hist['train_acc']) + 1)
    ax.plot(ep, np.array(hist['train_acc']) * 100, color=BLUE, label='train')
    ax.plot(ep, np.array(hist['val_acc']) * 100, color=RED, label='inner validation')
    ax.axvline(15.5, color=GREY, ls='--', lw=1)
    ax.annotate('stage 2 begins', (15.8, 25), fontsize=7.5, color=GREY)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.legend(frameon=False, fontsize=8)
    # Display name, not the checkpoint key. Section 5.1 of the paper drops the
    # Agri-EfficientNet name deliberately, so it must not reappear in a figure.
    ax.set_title('EfficientNet-B0 + LFA, comparison instance, seed 42',
                 fontsize=8.5, loc='left')
    save(fig, out, 'FigS1_training_curves')


def figS2(out):
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    lab = [r[0] for r in ROBUST]
    x = np.arange(len(lab)); w = 0.27
    ax.bar(x - w, [r[1] for r in ROBUST], w, color=BLUE, label='EfficientNet-B0 + LFA')
    ax.bar(x, [r[2] for r in ROBUST], w, color=GREEN, label='EfficientNet-B0')
    ax.bar(x + w, [r[3] for r in ROBUST], w, color=GREY, label='ResNet-50')
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.5)
    ax.set_ylabel('Macro F1 (%)'); ax.set_ylim(0, 95)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper center')
    ax.set_title('Single seed, 56-image test set: read as illustration, not ranking',
                 fontsize=8.5, loc='left')
    save(fig, out, 'FigS2_robustness')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results')
    ap.add_argument('--out', default='figures')
    ap.add_argument('--sessions', default='sessions.csv')
    ap.add_argument('--ckpt', default=None,
                    help='checkpoint directory, for the training-history files')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f'writing to {out.resolve()}')
    fig1(out)
    fig2_class_panel(out, args.sessions)
    fig3(out)
    fig4(args.results, out)
    fig5(out)
    fig6(out)
    fig7(out)
    figS1(args.results, out, args.ckpt)
    figS2(out)
    print('\nThe Grad-CAM figure (Online Resource 1) is not produced here. Run:')
    print('  python regen_gradcam.py --ckpt ckpt/group_s42/cmp_agri_efficientnet.pth '
          '--split_dir clean_split --out figures')
    print('and use session-level checkpoints, not the archived image-level ones.')
    print('\nEvery value drawn here is written at the top of this file and matches '
          'the manuscript tables. If a table changes, change it here too.')


if __name__ == '__main__':
    main()
