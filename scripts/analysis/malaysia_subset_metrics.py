#!/usr/bin/env python3
"""
malaysia_subset_metrics.py — make the cross-country drop compare like with like
==============================================================================
Table 7 subtracts a Vietnamese macro F1 from a Malaysian one. Those two numbers
have never been averaged over the same classes. The Malaysian column is a
five-class macro; the Vietnamese column is four-class, and once the
Leaf_Rhizoctonia to Root_disease correspondence is dropped it becomes
three-class. A difference between macros over different label sets is not a
transfer gap, it is two quantities subtracted for convenience.

This recomputes the Malaysian held-out scores restricted to the same subsets,
from the same checkpoints, so the drop is a like-for-like comparison:

    5 classes   the original figure, for continuity with Table 3
    4 classes   Algal, Leaf_rot, Phomopsis, Root_disease
    3 classes   Algal, Leaf_rot, Phomopsis  (organ-to-organ mappings only)

Restriction is by `labels=`, so a class absent from the subset neither
contributes a zero nor changes the denominator. Predictions falling outside the
subset still count as errors against the classes that are in it, which is the
correct treatment: the model was trained on five classes and we are not
pretending otherwise.

    python malaysia_subset_metrics.py \
        --ckpt_dir /workspace/ckpt/group_s42 \
        --split_dir /workspace/clean_split \
        --out /workspace/my_subset_s42

Run once per seed, then aggregate.
"""

import argparse
import json
import sys
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

SUBSETS = {
    '5-class': ['Algal', 'Leaf_rot', 'Phomopsis', 'Pink_disease', 'Root_disease'],
    '4-class': ['Algal', 'Leaf_rot', 'Phomopsis', 'Root_disease'],
    '3-class': ['Algal', 'Leaf_rot', 'Phomopsis'],
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


class LesionFocusAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.attention_conv = nn.Conv2d(ch, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        from torch.ao.nn.quantized import FloatFunctional
        self.multiply = FloatFunctional()

    def forward(self, x):
        return self.multiply.mul(x, self.sigmoid(self.attention_conv(x)))


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
        m = models.efficientnet_b0(weights=None)
        ch = m.features[-1][0].out_channels
        m.features = nn.Sequential(m.features, LesionFocusAttention(ch))
        m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                     nn.Linear(m.classifier[1].in_features, n))
    else:
        raise ValueError(name)
    return m


class TestSet(Dataset):
    def __init__(self, split_dir, class_names, transform):
        self.transform = transform
        self.samples = []
        for i, c in enumerate(class_names):
            d = Path(split_dir) / 'test' / c
            if not d.exists():
                continue
            for f in sorted(d.rglob('*')):
                if f.suffix.lower() in IMG_EXT:
                    self.samples.append((str(f), i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.transform(Image.open(p).convert('RGB')), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_dir', required=True)
    ap.add_argument('--split_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()

    ck = Path(args.ckpt_dir)
    if not ck.is_dir():
        sys.exit(f'No checkpoint directory: {ck}')

    class_names = sorted(p.name for p in (Path(args.split_dir) / 'train').iterdir()
                         if p.is_dir())
    n_cls = len(class_names)
    print(f'class_names: {class_names}')

    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = TestSet(args.split_dir, class_names, tf)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f'Malaysian test set: {len(ds)} images')
    for name, sub in SUBSETS.items():
        idx = [class_names.index(c) for c in sub if c in class_names]
        n = sum(1 for _, y in ds.samples if y in idx)
        print(f'  {name}: {n} images in {len(idx)} classes')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows, cms = [], {}
    for key in MODEL_KEYS:
        f = ck / f'cmp_{key}.pth'
        if not f.exists():
            print(f'  [skip] {key}')
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
        cms[key] = confusion_matrix(yt, yp, labels=list(range(n_cls))).tolist()

        r = {'Model': LABELS[key], 'key': key}
        for name, sub in SUBSETS.items():
            idx = [class_names.index(c) for c in sub if c in class_names]
            mask = np.isin(yt, idx)     # score only images whose true class is in the subset
            if mask.sum() == 0:
                r[name] = float('nan'); continue
            r[name] = round(100 * f1_score(yt[mask], yp[mask], labels=idx,
                                           average='macro', zero_division=0), 2)
        rows.append(r)
        print(f'  {LABELS[key]:<30} ' +
              '  '.join(f'{k} {r[k]:5.1f}' for k in SUBSETS))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'malaysia_subsets.csv', index=False)
    json.dump({'class_names': class_names, 'confusion': cms},
              open(out / 'malaysia_confusion.json', 'w'), indent=2)

    print('\n' + '=' * 66)
    print(df.drop(columns=['key']).to_string(index=False))
    print('=' * 66)
    print(f'\nWritten to {out}')
    print('Use the 3-class column against the 3-class Vietnamese figure. '
          'Subtracting a five-class macro from a three-class one was never a '
          'transfer gap.')


if __name__ == '__main__':
    main()
