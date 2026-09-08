#!/usr/bin/env python3
"""
export_confusion.py — write the confusion matrix Figure 3 needs
===============================================================
The archive kept fig_confusion_matrix.png but not the numbers behind it, so the
figure cannot be redrawn without rerunning inference. This does that: one pass
over the 56-image test set with the seed-42 comparison checkpoint, writing the
matrix as CSV.

It also writes the per-class report, so the values can be checked against
Table 5 rather than assumed to match.

    python export_confusion.py \\
        --ckpt ckpt/group_s42/cmp_agri_efficientnet.pth \\
        --split_dir clean_split \\
        --out results/group_s42

Use the comparison checkpoint (cmp_*), not the ablation one (abl_*). Table 5
reports the ablation instance and Tables 3 and 6 the comparison instance; the
two differ by about four points at this seed, and Figure 3 belongs with the
tables that use the comparison instance.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


class LesionFocusAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.attention_conv = nn.Conv2d(ch, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        from torch.ao.nn.quantized import FloatFunctional
        self.multiply = FloatFunctional()

    def forward(self, x):
        return self.multiply.mul(x, self.sigmoid(self.attention_conv(x)))


def build(n, lfa=True):
    m = models.efficientnet_b0(weights=None)
    if lfa:
        ch = m.features[-1][0].out_channels
        m.features = nn.Sequential(m.features, LesionFocusAttention(ch))
    m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                 nn.Linear(m.classifier[1].in_features, n))
    return m


class TestSet(Dataset):
    def __init__(self, split_dir, class_names, tf):
        self.tf = tf
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
        return self.tf(Image.open(p).convert('RGB')), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--split_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no_lfa', action='store_true',
                    help='use for cmp_efficientnet_b0.pth')
    args = ap.parse_args()

    ck = Path(args.ckpt)
    if not ck.is_file():
        sys.exit(f'No checkpoint: {ck}\n'
                 f'Unpack ckpt_group_s42.tar.gz first.')
    if 'abl_' in ck.name:
        print('Note: this is an ablation checkpoint. Figure 3 accompanies '
              'Tables 3 and 6, which use the comparison instance (cmp_*). '
              'Continuing, but check that is what you want.')

    class_names = sorted(p.name for p in (Path(args.split_dir) / 'train').iterdir()
                         if p.is_dir())
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = TestSet(args.split_dir, class_names, tf)
    print(f'classes: {class_names}')
    print(f'test images: {len(ds)}')

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = build(len(class_names), lfa=not args.no_lfa).to(dev)
    m.load_state_dict(torch.load(str(ck), map_location=dev))
    m.eval()

    yt, yp = [], []
    with torch.no_grad():
        for X, y in DataLoader(ds, batch_size=32, num_workers=0):
            yp.append(m(X.to(dev)).argmax(1).cpu().numpy())
            yt.append(y.numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)

    cm = confusion_matrix(yt, yp, labels=list(range(len(class_names))))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / 'confusion_matrix.csv', cm, fmt='%d', delimiter=',')

    print('\nconfusion matrix (rows = true):')
    w = max(len(c) for c in class_names) + 1
    print(' ' * w + ''.join(c[:9].rjust(11) for c in class_names))
    for i, c in enumerate(class_names):
        print(c.ljust(w) + ''.join(f'{v:>11d}' for v in cm[i]))

    print('\nper-class, to check against Table 5:')
    print(classification_report(yt, yp, target_names=class_names,
                                zero_division=0, digits=3))
    print(f'written to {out / "confusion_matrix.csv"}')
    print('\nNow rerun make_figures.py; Figure 3 will pick this up.')


if __name__ == '__main__':
    main()
