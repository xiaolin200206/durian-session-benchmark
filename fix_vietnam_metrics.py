#!/usr/bin/env python3
"""
fix_vietnam_metrics.py — recompute the cross-country numbers correctly
=====================================================================
Two defects in the original evaluation, both cheap to fix because this is
inference only:

1. `calc_metrics` called f1_score(average='macro') with no `labels=`.
   scikit-learn then averages over the union of true and predicted labels.
   The mapped Vietnamese task has four classes because Pink_disease has no
   counterpart, but a model trained on five can still predict Pink_disease.
   The moment it does so on any of the 320 images, Pink_disease enters the
   average with F1 = 0 and every score is divided by five instead of four.
   This script reports both, so the size of the artefact is visible rather
   than assumed.

2. No chance baseline was reported. A macro F1 in the twenties on a
   four-class problem is uninterpretable without one. This script computes
   the uniform, prior-matched and majority-class baselines from the actual
   test-set composition, and states for each model whether it clears them.

It also writes the confusion matrix, which is what actually shows whether a
class mapping is broken: a model that puts all Root_disease predictions on
Leaf_rot is telling you something different from one predicting at random.

    python fix_vietnam_metrics.py \
        --ckpt_dir /workspace/ckpt/group_s42 \
        --vietnam /workspace/vietnam \
        --split_dir /workspace/clean_split \
        --out /workspace/vn_fixed_s42

Run once per seed directory and the script will aggregate if you point --out
at the same folder each time.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

VN_TO_MY = {
    'Leaf_Algal':          'Algal',
    'Leaf_Phomopsis':      'Phomopsis',
    'Leaf_Blight':         'Leaf_rot',
    'Leaf_Colletotrichum': 'Leaf_rot',
    'Leaf_Rhizoctonia':    'Root_disease',
}

MODEL_KEYS = ['vgg16', 'resnet50', 'resnet101', 'mobilenetv2', 'mobilenetv3',
              'shufflenetv2', 'efficientnet_b0', 'efficientnetv2_s',
              'convnext_tiny', 'agri_efficientnet']

LABELS = {'vgg16': 'VGG-16', 'resnet50': 'ResNet-50', 'resnet101': 'ResNet-101',
          'mobilenetv2': 'MobileNetV2', 'mobilenetv3': 'MobileNetV3-Large',
          'shufflenetv2': 'ShuffleNetV2-1.0x',
          'efficientnet_b0': 'EfficientNet-B0 (No LFA)',
          'efficientnetv2_s': 'EfficientNetV2-S',
          'convnext_tiny': 'ConvNeXt-Tiny',
          'agri_efficientnet': 'EfficientNet-B0 + LFA (ours)'}


# ---------------------------------------------------------------- model defs
class LesionFocusAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        from torch.ao.nn.quantized import FloatFunctional
        self.multiply = FloatFunctional()

    def forward(self, x):
        return self.multiply.mul(x, self.sigmoid(self.attention_conv(x)))


def build_agri(num_classes):
    m = models.efficientnet_b0(weights=None)
    last_ch = m.features[-1][0].out_channels
    m.features = nn.Sequential(m.features, LesionFocusAttention(last_ch))
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                 nn.Linear(in_f, num_classes))
    return m


def build_model(name, n):
    if name == 'vgg16':
        m = models.vgg16(weights=None); m.classifier[6] = nn.Linear(4096, n)
    elif name == 'resnet50':
        m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, n)
    elif name == 'resnet101':
        m = models.resnet101(weights=None); m.fc = nn.Linear(m.fc.in_features, n)
    elif name == 'mobilenetv2':
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    elif name == 'mobilenetv3':
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n)
    elif name == 'shufflenetv2':
        m = models.shufflenet_v2_x1_0(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n)
    elif name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                     nn.Linear(m.classifier[1].in_features, n))
    elif name == 'efficientnetv2_s':
        m = models.efficientnet_v2_s(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                     nn.Linear(m.classifier[1].in_features, n))
    elif name == 'convnext_tiny':
        m = models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n)
    elif name == 'agri_efficientnet':
        m = build_agri(n)
    else:
        raise ValueError(name)
    return m


# ---------------------------------------------------------------- data
class VietnamDataset(Dataset):
    def __init__(self, root, split, class_names, transform):
        self.transform = transform
        idx = {c: i for i, c in enumerate(class_names)}
        self.samples = []
        for vn_cls, my_cls in VN_TO_MY.items():
            d = Path(root) / split / vn_cls
            if not d.exists() or my_cls not in idx:
                continue
            for f in sorted(d.rglob('*')):
                if f.suffix.lower() in IMG_EXT:
                    self.samples.append((str(f), idx[my_cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.transform(Image.open(p).convert('RGB')), y


def chance_baselines(counts, k):
    """Macro F1 under uniform, prior-matched and majority-class prediction."""
    n = sum(counts.values())
    prev = {c: v / n for c, v in counts.items()}
    uni = np.mean([2 * p * (1 / k) / (p + 1 / k) for p in prev.values()])
    prior = np.mean([p for p in prev.values()])            # 2p^2/(2p) = p
    top = max(prev, key=prev.get)
    maj = np.mean([2 * prev[top] / (prev[top] + 1) if c == top else 0.0
                   for c in prev])
    return 100 * uni, 100 * prior, 100 * maj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--vietnam', required=True)
    ap.add_argument('--split_dir', required=True,
                    help='to recover class_names in the same order as training')
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()

    ck = Path(args.ckpt_dir)
    if not ck.is_dir():
        sys.exit(f'No checkpoint directory: {ck}\n'
                 f'If the pod volume was deleted the weights are gone and the '
                 f'models have to be retrained before this can be run.')

    class_names = sorted(p.name for p in (Path(args.split_dir) / 'train').iterdir()
                         if p.is_dir())
    n_cls = len(class_names)
    shared = sorted({class_names.index(v) for v in VN_TO_MY.values()})
    absent = [i for i in range(n_cls) if i not in shared]
    print(f'class_names : {class_names}')
    print(f'shared idx  : {shared}  ({[class_names[i] for i in shared]})')
    print(f'absent idx  : {absent}  ({[class_names[i] for i in absent]})')

    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = VietnamDataset(args.vietnam, 'test', class_names, tf)
    if len(ds) == 0:
        sys.exit(f'No Vietnamese test images found under {args.vietnam}/test')
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f'Vietnam test: {len(ds)} images')

    counts = Counter(class_names[y] for _, y in ds.samples)
    print(f'composition : {dict(counts)}')
    uni, prior, maj = chance_baselines(counts, len(shared))
    print(f'\nchance baselines over {len(shared)} classes:')
    print(f'  uniform        {uni:.1f}%')
    print(f'  prior-matched  {prior:.1f}%')
    print(f'  majority class {maj:.1f}%')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device      : {dev}\n')

    rows, cms = [], {}
    for key in MODEL_KEYS:
        f = ck / f'cmp_{key}.pth'
        if not f.exists():
            print(f'  [skip] {key}: no checkpoint')
            continue
        m = build_model(key, n_cls).to(dev)
        m.load_state_dict(torch.load(str(f), map_location=dev))
        m.eval()
        yt, yp = [], []
        with torch.no_grad():
            for X, y in dl:
                yp.append(m(X.to(dev)).argmax(1).cpu().numpy())
                yt.append(y.numpy())
        yt, yp = np.concatenate(yt), np.concatenate(yp)

        pred_absent = int(np.isin(yp, absent).sum())
        union = sorted(set(yt.tolist()) | set(yp.tolist()))

        def sc(labels):
            return dict(
                acc=100 * accuracy_score(yt, yp),
                prec=100 * precision_score(yt, yp, labels=labels,
                                           average='macro', zero_division=0),
                rec=100 * recall_score(yt, yp, labels=labels,
                                       average='macro', zero_division=0),
                f1=100 * f1_score(yt, yp, labels=labels,
                                  average='macro', zero_division=0))

        correct = sc(shared)          # the four mapped classes only
        as_run = sc(union)            # what the original code computed
        rows.append({
            'Model': LABELS[key], 'key': key,
            'Macro F1 (4-class, correct)': round(correct['f1'], 1),
            'Macro F1 (as originally run)': round(as_run['f1'], 1),
            'Artefact (pp)': round(correct['f1'] - as_run['f1'], 1),
            'Accuracy (%)': round(correct['acc'], 1),
            'Precision (%)': round(correct['prec'], 1),
            'Recall (%)': round(correct['rec'], 1),
            'classes in average as run': len(union),
            'preds on absent class': pred_absent,
            'above uniform chance': correct['f1'] > uni,
        })
        cms[key] = confusion_matrix(yt, yp, labels=list(range(n_cls))).tolist()
        print(f'  {LABELS[key]:<30} correct {correct["f1"]:5.1f}   '
              f'as-run {as_run["f1"]:5.1f}   '
              f'absent-class preds {pred_absent:4d}   '
              f'{"" if correct["f1"] > uni else "AT OR BELOW CHANCE"}')

    if not rows:
        sys.exit('No checkpoints evaluated.')

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values('Macro F1 (4-class, correct)',
                                        ascending=False)
    df.to_csv(out / 'vietnam_corrected.csv', index=False)
    json.dump({'class_names': class_names, 'shared': shared,
               'composition': dict(counts),
               'chance': {'uniform': uni, 'prior': prior, 'majority': maj},
               'confusion': cms},
              open(out / 'vietnam_confusion.json', 'w'), indent=2)

    print('\n' + '=' * 74)
    print(df.drop(columns=['key']).to_string(index=False))
    print('=' * 74)
    art = df['Artefact (pp)']
    print(f'\nThe missing labels= argument cost {art.min():.1f} to {art.max():.1f} '
          f'points (mean {art.mean():.1f}).')
    n_above = int(df['above uniform chance'].sum())
    print(f'{n_above} of {len(df)} architectures clear the {uni:.1f}% uniform '
          f'chance baseline once corrected.')
    print(f'\nWritten to {out}')
    print('Read vietnam_confusion.json before drawing any conclusion about the '
          'class mapping: a model collapsing Root_disease onto Leaf_rot is a '
          'mapping failure, not a distribution-shift result.')


if __name__ == '__main__':
    main()
