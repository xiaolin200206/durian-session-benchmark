#!/usr/bin/env python3
"""
fix_crop_control.py — put the centre-crop control on the same footing as Table 7
===============================================================================
Table 9 as it stands mixes bases. Its Malaysian column carries five-class macro
F1 taken from Table 3, its Vietnamese column four-class macro F1, and the gap
between them is a subtraction across different label sets. That is the error
Section 4.6 spends a page on, committed in the paper's own control experiment.

This recomputes all four columns over the three organ-matched classes:

    MY (orig)   Malaysian test set, models trained on clean_split
    VN (orig)   Vietnamese test set, same models
    MY (crop)   Malaysian centre-cropped test set, models trained on the crop
    VN (crop)   Vietnamese test set, crop-trained models

Nothing is retrained. The first two come from files you already have; the last
two are one inference pass each over 56 and 320 images.

    python fix_crop_control.py \\
        --ckpt_orig  ckpt_group_s42 \\
        --ckpt_crop  ckpt_cc_s42 \\
        --split_orig clean_split \\
        --split_crop clean_split_cc \\
        --vietnam    "C:/Users/.../Vietnam dataset/Durian_Leaf_Diseases" \\
        --out        crop_control_3class

If ckpt_group_s42 is not unpacked locally, pass --ckpt_orig none and the script
will leave the orig columns blank; those two are already known from Table 7 and
can be filled by hand (73.3 / 67.2 / 72.5 and 32.8 / 28.4 / 30.3).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
THREE = ['Algal', 'Leaf_rot', 'Phomopsis']
VN_TO_MY = {'Leaf_Algal': 'Algal', 'Leaf_Phomopsis': 'Phomopsis',
            'Leaf_Blight': 'Leaf_rot', 'Leaf_Colletotrichum': 'Leaf_rot',
            'Leaf_Rhizoctonia': 'Root_disease'}
KEYS = ['agri_efficientnet', 'efficientnet_b0', 'mobilenetv2']
NAMES = {'agri_efficientnet': 'EfficientNet-B0 + LFA',
         'efficientnet_b0': 'EfficientNet-B0',
         'mobilenetv2': 'MobileNetV2'}


class LFA(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.attention_conv = nn.Conv2d(ch, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        from torch.ao.nn.quantized import FloatFunctional
        self.multiply = FloatFunctional()

    def forward(self, x):
        return self.multiply.mul(x, self.sigmoid(self.attention_conv(x)))


def build(key, n):
    if key == 'mobilenetv2':
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
    else:
        m = models.efficientnet_b0(weights=None)
        if key == 'agri_efficientnet':
            ch = m.features[-1][0].out_channels
            m.features = nn.Sequential(m.features, LFA(ch))
        m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                     nn.Linear(m.classifier[1].in_features, n))
    return m


class Folder(Dataset):
    def __init__(self, root, split, class_names, tf, mapping=None):
        self.tf = tf
        idx = {c: i for i, c in enumerate(class_names)}
        self.samples = []
        base = Path(root) / split
        srcs = mapping.items() if mapping else [(c, c) for c in class_names]
        for src, dst in srcs:
            d = base / src
            if not d.exists() or dst not in idx:
                continue
            for f in sorted(d.rglob('*')):
                if f.suffix.lower() in IMG_EXT:
                    self.samples.append((str(f), idx[dst]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.tf(Image.open(p).convert('RGB')), y


TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def score(model, ds, keep_idx, dev, batch=32):
    if len(ds) == 0:
        return float('nan')
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    yt, yp = [], []
    with torch.no_grad():
        for X, y in dl:
            yp.append(model(X.to(dev)).argmax(1).cpu().numpy())
            yt.append(y.numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    mask = np.isin(yt, keep_idx)
    if mask.sum() == 0:
        return float('nan')
    return 100 * f1_score(yt[mask], yp[mask], labels=keep_idx,
                          average='macro', zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_orig', required=True)
    ap.add_argument('--ckpt_crop', required=True)
    ap.add_argument('--split_orig', required=True)
    ap.add_argument('--split_crop', required=True)
    ap.add_argument('--vietnam', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    class_names = sorted(p.name for p in (Path(args.split_orig) / 'train').iterdir()
                         if p.is_dir())
    n_cls = len(class_names)
    keep = [class_names.index(c) for c in THREE]
    print(f'classes {class_names}')
    print(f'scored over {THREE} -> indices {keep}\n')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    my_o = Folder(args.split_orig, 'test', class_names, TF)
    my_c = Folder(args.split_crop, 'test', class_names, TF)
    vn = Folder(args.vietnam, 'test', class_names, TF, VN_TO_MY)
    print(f'MY orig {len(my_o)}   MY crop {len(my_c)}   VN {len(vn)} images')
    n3 = sum(1 for _, y in vn.samples if y in keep)
    print(f'VN images in the three scored classes: {n3}\n')

    rows = []
    for k in KEYS:
        r = {'Model': NAMES[k]}
        for tag, ck, ds in [('MY (orig)', args.ckpt_orig, my_o),
                            ('VN (orig)', args.ckpt_orig, vn),
                            ('MY (crop)', args.ckpt_crop, my_c),
                            ('VN (crop)', args.ckpt_crop, vn)]:
            if ck.lower() == 'none':
                r[tag] = float('nan'); continue
            f = Path(ck) / f'cmp_{k}.pth'
            if not f.exists():
                print(f'  [skip] {f}')
                r[tag] = float('nan'); continue
            m = build(k, n_cls).to(dev)
            m.load_state_dict(torch.load(str(f), map_location=dev))
            m.eval()
            r[tag] = round(score(m, ds, keep, dev), 1)
        r['Gap (orig)'] = (round(r['MY (orig)'] - r['VN (orig)'], 1)
                           if not any(np.isnan([r['MY (orig)'], r['VN (orig)']])) else float('nan'))
        r['Gap (crop)'] = (round(r['MY (crop)'] - r['VN (crop)'], 1)
                           if not any(np.isnan([r['MY (crop)'], r['VN (crop)']])) else float('nan'))
        rows.append(r)
        print(f"  {NAMES[k]:<24} " + '  '.join(f'{t} {r[t]}' for t in
              ['MY (orig)', 'VN (orig)', 'Gap (orig)', 'MY (crop)', 'VN (crop)', 'Gap (crop)']))

    df = pd.DataFrame(rows)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'crop_control_3class.csv', index=False)
    print('\n' + '=' * 74)
    print(df.to_string(index=False))
    print('=' * 74)
    go, gc = df['Gap (orig)'].mean(), df['Gap (crop)'].mean()
    if not np.isnan(go) and not np.isnan(gc):
        print(f'\nmean gap {go:.1f} -> {gc:.1f}   narrowing {go-gc:.1f} '
              f'= {100*(go-gc)/go:.0f}% of the gap')
        print('Report this fraction, not the one computed from mixed bases.')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
