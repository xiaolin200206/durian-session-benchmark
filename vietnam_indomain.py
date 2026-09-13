#!/usr/bin/env python3
"""
vietnam_indomain.py — the experiment that decides whether Section 5.3 stands
===========================================================================
Malaysian-trained models land at the chance baseline on the Vietnamese test
set. Two incompatible explanations give that same number:

  (a) the two domains are far enough apart that zero-shot transfer has no
      purchase, which is the paper's claim; or
  (b) one or more of our class mappings is wrong, so a sensible model is
      being scored against labels that do not describe what it sees.

Training on the Vietnamese data itself, under the same four-class mapping and
the same protocol, separates them. If the in-domain model reaches the range
the Malaysian in-domain models reach, the mapping is learnable and (a) holds.
If it does not, the mapping is the problem and the cross-country claim has to
be restated over the three classes that map cleanly.

    python vietnam_indomain.py --vietnam /workspace/vietnam \
        --out /workspace/vn_indomain --seeds 42 1 2

Optionally add a few-shot curve, fine-tuning the Malaysian checkpoint on N
Vietnamese training images:

    python vietnam_indomain.py --vietnam /workspace/vietnam \
        --out /workspace/vn_indomain --seeds 42 \
        --fewshot 10 25 50 100 --init_ckpt /workspace/ckpt/group_s42/cmp_agri_efficientnet.pth

CAVEAT, and it needs to appear in the paper. This dataset ships its own
train/val/test partition, and our audit found 44-48% of consecutive filename
numbers crossing split boundaries, i.e. it is an image-level split. An
in-domain score computed on it is therefore itself inflated. That is
tolerable for the purpose here, because the question is whether the mapping is
learnable *at all*: an inflated number that is still low is decisive, and an
inflated number that is high should be reported as an upper bound, not as the
Vietnamese in-domain performance. Pass --dedupe to drop exact cross-split
duplicates, which removes the crudest part of the problem.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
VN_TO_MY = {
    'Leaf_Algal':          'Algal',
    'Leaf_Phomopsis':      'Phomopsis',
    'Leaf_Blight':         'Leaf_rot',
    'Leaf_Colletotrichum': 'Leaf_rot',
    'Leaf_Rhizoctonia':    'Root_disease',
}
CLASSES = ['Algal', 'Leaf_rot', 'Phomopsis', 'Root_disease']


class LesionFocusAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.attention_conv = nn.Conv2d(ch, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        from torch.ao.nn.quantized import FloatFunctional
        self.multiply = FloatFunctional()

    def forward(self, x):
        return self.multiply.mul(x, self.sigmoid(self.attention_conv(x)))


def build(n_cls, pretrained=True):
    w = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_b0(weights=w)
    for p in m.parameters():
        p.requires_grad = False
    ch = m.features[-1][0].out_channels
    m.features = nn.Sequential(m.features, LesionFocusAttention(ch))
    inf = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                 nn.Linear(inf, n_cls))
    for p in m.classifier.parameters():
        p.requires_grad = True
    return m


class VNSet(Dataset):
    def __init__(self, root, split, transform, keep=None):
        self.transform = transform
        idx = {c: i for i, c in enumerate(CLASSES)}
        self.samples = []
        for vn, my in VN_TO_MY.items():
            d = Path(root) / split / vn
            if not d.exists():
                continue
            for f in sorted(d.rglob('*')):
                if f.suffix.lower() in IMG_EXT:
                    if keep is not None and str(f) not in keep:
                        continue
                    self.samples.append((str(f), idx[my]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        return self.transform(Image.open(p).convert('RGB')), y


def md5(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def dedupe_train(root):
    """Return the set of training paths whose bytes do not also appear in test."""
    def hashes(split):
        out = {}
        for vn in VN_TO_MY:
            d = Path(root) / split / vn
            if not d.exists():
                continue
            for f in sorted(d.rglob('*')):
                if f.suffix.lower() in IMG_EXT:
                    out.setdefault(md5(f), []).append(str(f))
        return out
    te = set(hashes('test'))
    tr = hashes('train')
    keep = {p for h, ps in tr.items() if h not in te for p in ps}
    dropped = sum(len(ps) for h, ps in tr.items() if h in te)
    print(f'  dedupe: dropped {dropped} training images duplicated in test')
    return keep


TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
TEST_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


@torch.no_grad()
def evaluate(m, dl, dev):
    m.eval(); yt, yp = [], []
    for X, y in dl:
        yp.append(m(X.to(dev)).argmax(1).cpu().numpy()); yt.append(y.numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return (100 * f1_score(yt, yp, labels=list(range(len(CLASSES))),
                           average='macro', zero_division=0),
            100 * accuracy_score(yt, yp),
            confusion_matrix(yt, yp, labels=list(range(len(CLASSES)))))


def train(m, tr_dl, va_dl, dev, s1, s2):
    tgt = [y for _, y in tr_dl.dataset.samples]
    cnt = Counter(tgt)
    w = torch.tensor([len(tgt) / (len(CLASSES) * max(cnt[i], 1))
                      for i in range(len(CLASSES))], dtype=torch.float).to(dev)
    crit = nn.CrossEntropyLoss(weight=w)
    best, best_st = -1, None
    for stage, (n_ep, lr) in enumerate([(s1, 1e-3), (s2, 1e-5)]):
        if stage == 1:
            for p in m.parameters():
                p.requires_grad = True
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
        for ep in range(n_ep):
            m.train()
            for X, y in tr_dl:
                X, y = X.to(dev), y.to(dev)
                opt.zero_grad(); crit(m(X), y).backward(); opt.step()
            f1, acc, _ = evaluate(m, va_dl, dev)
            print(f'    S{stage+1} Ep{ep+1:02d} val F1={f1:.1f} acc={acc:.1f}')
            if f1 > best:
                best, best_st = f1, {k: v.detach().clone()
                                     for k, v in m.state_dict().items()}
    if best_st:
        m.load_state_dict(best_st)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vietnam', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 1, 2])
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--s1', type=int, default=15)
    ap.add_argument('--s2', type=int, default=15)
    ap.add_argument('--dedupe', action='store_true')
    ap.add_argument('--fewshot', type=int, nargs='*', default=[])
    ap.add_argument('--init_ckpt', default=None,
                    help='Malaysian checkpoint to fine-tune for the few-shot curve')
    args = ap.parse_args()

    root = Path(args.vietnam)
    if not (root / 'train').is_dir():
        sys.exit(f'No train/ under {root}')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    keep = dedupe_train(root) if args.dedupe else None
    tr = VNSet(root, 'train', TRAIN_TF, keep)
    va = VNSet(root, 'val', TEST_TF)
    te = VNSet(root, 'test', TEST_TF)
    print(f'train {len(tr)}  val {len(va)}  test {len(te)}  device {dev}')
    print(f'train composition: {Counter(CLASSES[y] for _, y in tr.samples)}')
    print(f'test  composition: {Counter(CLASSES[y] for _, y in te.samples)}')

    va_dl = DataLoader(va, batch_size=args.batch, num_workers=0)
    te_dl = DataLoader(te, batch_size=args.batch, num_workers=0)

    rows = []
    for seed in args.seeds:
        print(f'\n=== in-domain, seed {seed} ===')
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        tgt = [y for _, y in tr.samples]
        cnt = Counter(tgt)
        sw = [len(tgt) / (len(CLASSES) * max(cnt[y], 1)) for y in tgt]
        dl = DataLoader(tr, batch_size=args.batch,
                        sampler=WeightedRandomSampler(sw, len(sw), True),
                        num_workers=0)
        m = build(len(CLASSES)).to(dev)
        m = train(m, dl, va_dl, dev, args.s1, args.s2)
        f1, acc, cm = evaluate(m, te_dl, dev)
        print(f'  TEST macro F1 {f1:.1f}  acc {acc:.1f}')
        print(f'  confusion (rows=true {CLASSES}):\n{cm}')
        rows.append({'condition': 'in-domain', 'seed': seed, 'n_train': len(tr),
                     'Macro F1 (%)': round(f1, 1), 'Accuracy (%)': round(acc, 1)})
        torch.save(m.state_dict(), out / f'vn_indomain_s{seed}.pth')
        np.savetxt(out / f'vn_indomain_cm_s{seed}.csv', cm, fmt='%d', delimiter=',')

    if args.fewshot:
        if not args.init_ckpt or not Path(args.init_ckpt).exists():
            print('\n[skip] few-shot curve needs --init_ckpt pointing at a '
                  'Malaysian checkpoint')
        else:
            print('\n=== few-shot curve ===')
            print('Note: the Malaysian checkpoint has five output units and this '
                  'task has four, so the classifier head is reinitialised and '
                  'only the backbone transfers.')
            for n in args.fewshot:
                for seed in args.seeds[:1]:
                    torch.manual_seed(seed); random.seed(seed)
                    idxs = list(range(len(tr.samples)))
                    random.shuffle(idxs)
                    sub = idxs[:n]
                    small = VNSet(root, 'train', TRAIN_TF, keep)
                    small.samples = [tr.samples[i] for i in sub]
                    if len(set(y for _, y in small.samples)) < len(CLASSES):
                        print(f'  n={n}: fewer than {len(CLASSES)} classes drawn, '
                              f'skipping')
                        continue
                    m = build(len(CLASSES)).to(dev)
                    sd = torch.load(args.init_ckpt, map_location=dev)
                    sd = {k: v for k, v in sd.items()
                          if not k.startswith('classifier')}
                    m.load_state_dict(sd, strict=False)
                    dl = DataLoader(small, batch_size=min(args.batch, len(small)),
                                    shuffle=True, num_workers=0)
                    m = train(m, dl, va_dl, dev, 10, 10)
                    f1, acc, _ = evaluate(m, te_dl, dev)
                    print(f'  n={n:4d} seed={seed}  test F1 {f1:.1f}')
                    rows.append({'condition': f'few-shot n={n}', 'seed': seed,
                                 'n_train': n, 'Macro F1 (%)': round(f1, 1),
                                 'Accuracy (%)': round(acc, 1)})

    df = pd.DataFrame(rows)
    df.to_csv(out / 'vietnam_indomain.csv', index=False)
    print('\n' + '=' * 66)
    print(df.to_string(index=False))
    print('=' * 66)

    ind = df[df.condition == 'in-domain']['Macro F1 (%)']
    if len(ind):
        print(f'\nIn-domain macro F1: {ind.mean():.1f} ± {ind.std():.1f} '
              f'over {len(ind)} seeds')
        print('Chance for this four-class test set is 24.4% (uniform).')
        if ind.mean() > 70:
            print('\nThe mapping is learnable. The zero-shot collapse is '
                  'distribution shift, and Section 5.3 stands as written.')
        elif ind.mean() > 45:
            print('\nPartially learnable. Report the in-domain ceiling '
                  'explicitly and describe the zero-shot result relative to it, '
                  'not relative to the Malaysian ceiling.')
        else:
            print('\nThe mapping is NOT learnable even in-domain. Do not '
                  'attribute the collapse to geography. Inspect the per-class '
                  'confusion, drop Root_disease from the correspondence, and '
                  'restate the cross-country result over the classes that map '
                  'cleanly.')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
