"""
NOT REPORTED IN THE PAPER.

This file tests whether LFA transfers to a MobileNetV2 backbone. It was written
before the session-level rerun and its results were produced under the
image-level split, so they are not comparable with anything in the manuscript
and are not cited by it.

It is kept here because deleting code that was run would misrepresent what was
done, but do not read its output as a result. If you want the comparison, rerun
it against clean_split with several seeds, and expect the same finding as the
EfficientNet ablation: no benefit that survives grouped evaluation.
"""

# -*- coding: utf-8 -*-
"""
mobilenetv2_lfa.py
==================
Trains MobileNetV2 + LFA and MobileNetV2 (no LFA) and adds them to:
  - comparison_table.csv  (extended)
  - robustness_results.csv (extended, IF fix_robustness.py already run)

Supplementary experiment: tests whether LFA generalises to a backbone
other than EfficientNet-B0.

Run AFTER fix_robustness.py.
Run: python mobilenetv2_lfa.py
"""

import os
from pathlib import Path
import warnings, copy, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFilter
import random

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torchvision.models import MobileNet_V2_Weights, EfficientNet_B0_Weights
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torch.nn.quantized import FloatFunctional
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# ── PATHS ─────────────────────────────────────────────────────────────────────
SPLIT_DIR = Path(os.environ.get('SPLIT_DIR', 'data/malaysia_split'))
SAVE_DIR  = Path(os.environ.get('SAVE_DIR', 'results'))
CKPT_DIR  = Path(os.environ.get('CKPT_DIR', 'checkpoints'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

BATCH = 16
NW    = 0

# ── TRANSFORMS ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomAffine(degrees=20, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.10)),
])
test_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(str(SPLIT_DIR / 'train'), transform=train_tf)
val_ds   = datasets.ImageFolder(str(SPLIT_DIR / 'val'),   transform=test_tf)
test_ds  = datasets.ImageFolder(str(SPLIT_DIR / 'test'),  transform=test_tf)

class_names = train_ds.classes
num_classes  = len(class_names)
print(f'Classes ({num_classes}): {class_names}')

train_targets = [s[1] for s in train_ds.samples]
class_counts  = Counter(train_targets)
total_train   = len(train_targets)
class_weights = torch.tensor(
    [total_train / (num_classes * class_counts[i]) for i in range(num_classes)],
    dtype=torch.float).to(device)

sample_w     = [class_weights[t].item() for t in train_targets]
sampler      = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=NW)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,  num_workers=NW)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,  num_workers=NW)

# ── MODEL DEFINITIONS ─────────────────────────────────────────────────────────
class LesionFocusAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid        = nn.Sigmoid()
        self.multiply       = FloatFunctional()
    def forward(self, x):
        return self.multiply.mul(x, self.sigmoid(self.attention_conv(x)))

def build_mobilenetv2_lfa(num_classes):
    """MobileNetV2 backbone + LFA attention module."""
    m = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    for p in m.parameters(): p.requires_grad = False
    # MobileNetV2 last conv output channels = 1280
    last_ch = m.features[-1][0].out_channels  # 1280
    m.features = nn.Sequential(m.features, LesionFocusAttention(last_ch))
    for p in m.features[-1].parameters(): p.requires_grad = True
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    for p in m.classifier.parameters(): p.requires_grad = True
    return m

def build_mobilenetv2(num_classes):
    """Standard MobileNetV2 (same as main script comparison)."""
    m = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    for p in m.parameters(): p.requires_grad = False
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    for p in m.classifier.parameters(): p.requires_grad = True
    return m

# ── TRAINING ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, crit, opt):
    model.train()
    tl, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        out  = model(X)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        tl      += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total   += X.size(0)
    return tl / total, correct / total

@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    tl, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out  = model(X)
        loss = crit(out, y)
        tl      += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total   += X.size(0)
    return tl / total, correct / total

def two_stage_train(model, save_path, label):
    crit = nn.CrossEntropyLoss(weight=class_weights)
    best_va, best_st = 0.0, None

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for ep in range(1, 16):
        tl, ta = train_one_epoch(model, train_loader, crit, opt)
        vl, va = evaluate(model, val_loader, crit)
        if va > best_va: best_va, best_st = va, copy.deepcopy(model.state_dict())
        print(f'    S1 Ep{ep:02d} | tl={tl:.4f} ta={ta:.4f} | vl={vl:.4f} va={va:.4f}')

    model.load_state_dict(best_st)
    for p in model.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    for ep in range(1, 16):
        tl, ta = train_one_epoch(model, train_loader, crit, opt)
        vl, va = evaluate(model, val_loader, crit)
        if va > best_va: best_va, best_st = va, copy.deepcopy(model.state_dict())
        print(f'    S2 Ep{ep:02d} | tl={tl:.4f} ta={ta:.4f} | vl={vl:.4f} va={va:.4f}')

    model.load_state_dict(best_st)
    torch.save(best_st, save_path)
    print(f'  => Saved: {save_path}')
    return model

@torch.no_grad()
def get_preds(model, loader):
    model.eval()
    yt, yp = [], []
    for X, y in loader:
        yp.extend(model(X.to(device)).argmax(1).cpu().numpy())
        yt.extend(y.numpy())
    return np.array(yt), np.array(yp)

def cpu_ms(model, reps=100, warmup=10):
    m = model.cpu().eval()
    d = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(warmup): m(d)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); m(d)
            ts.append((time.perf_counter() - t0) * 1000)
    return round(np.mean(ts), 2)

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def model_size_mb(ckpt_path):
    return round(ckpt_path.stat().st_size / 1024 / 1024, 2)

# ── TRAIN BOTH MODELS ─────────────────────────────────────────────────────────
experiments = [
    ('MobileNetV2 + LFA', build_mobilenetv2_lfa, CKPT_DIR / 'cmp_mobilenetv2_lfa.pth'),
    # MobileNetV2 (no LFA) is already trained — load from existing checkpoint
    # ('MobileNetV2 (No LFA)', build_mobilenetv2, CKPT_DIR / 'cmp_mobilenetv2.pth'),
]

results = {}
for label, build_fn, ckpt in experiments:
    print(f'\n[Training] {label}')
    m = build_fn(num_classes).to(device)

    if ckpt.exists():
        print(f'  Loading existing checkpoint: {ckpt.name}')
        m.load_state_dict(torch.load(str(ckpt), map_location=device))
    else:
        m = two_stage_train(m, str(ckpt), label)

    yt, yp = get_preds(m, test_loader)
    acc  = round(accuracy_score(yt, yp) * 100, 1)
    prec = round(precision_score(yt, yp, average='macro', zero_division=0) * 100, 1)
    rec  = round(recall_score(yt, yp, average='macro', zero_division=0) * 100, 1)
    f1   = round(f1_score(yt, yp, average='macro', zero_division=0) * 100, 1)
    lat  = cpu_ms(m)
    par  = round(count_params(m), 2)
    sz   = model_size_mb(ckpt)

    results[label] = {
        'Accuracy (%)': acc, 'Precision (%)': prec,
        'Recall (%)': rec,   'Macro F1 (%)': f1,
        'Params(M)': par,    'Size(MB)': sz, 'CPU(ms)': lat,
    }
    print(f'  => F1={f1}% Acc={acc}% Size={sz}MB CPU={lat}ms')

print('\n=== RESULTS ===')
for label, r in results.items():
    print(f'{label}: F1={r["Macro F1 (%)"]}%, Acc={r["Accuracy (%)"]}%, '
          f'Size={r["Size(MB)"]}MB, CPU={r["CPU(ms)"]}ms')

# ── ROBUSTNESS FOR MobileNetV2+LFA ───────────────────────────────────────────
print('\n=== Running robustness for MobileNetV2+LFA ===')

class PerturbedDataset(Dataset):
    def __init__(self, base_dataset, perturb_fn=None):
        self.base    = base_dataset
        self.perturb = perturb_fn
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        path, label = self.base.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.perturb: img = self.perturb(img)
        return test_tf(img), label

def make_perturb(ptype):
    if ptype == 'gaussian_noise':
        def fn(img):
            arr = np.array(img, dtype=np.float32) + np.random.normal(0, 25, np.array(img).shape)
            return Image.fromarray(arr.clip(0,255).astype(np.uint8))
        return fn
    elif ptype == 'low_brightness':
        return lambda img: Image.fromarray((np.array(img, np.float32)*0.4).clip(0,255).astype(np.uint8))
    elif ptype == 'high_brightness':
        return lambda img: Image.fromarray((np.array(img, np.float32)*1.6).clip(0,255).astype(np.uint8))
    elif ptype == 'blur':
        return lambda img: img.filter(ImageFilter.GaussianBlur(radius=3))
    elif ptype == 'occlusion':
        def fn(img):
            arr = np.array(img); h,w = arr.shape[:2]
            size = int(min(h,w)*0.3); x=random.randint(0,w-size); y=random.randint(0,h-size)
            arr[y:y+size, x:x+size] = 0; return Image.fromarray(arr)
        return fn
    return None

PERTURBATIONS = [
    ('Clean (Baseline)', None), ('Gaussian Noise', 'gaussian_noise'),
    ('Low Brightness', 'low_brightness'), ('High Brightness', 'high_brightness'),
    ('Motion Blur', 'blur'), ('Occlusion (30%)', 'occlusion'),
]

if (CKPT_DIR / 'cmp_mobilenetv2_lfa.pth').exists():
    mv2_lfa = build_mobilenetv2_lfa(num_classes).to(device)
    mv2_lfa.load_state_dict(torch.load(str(CKPT_DIR/'cmp_mobilenetv2_lfa.pth'), map_location=device))
    mv2_lfa.eval()

    rob_mv2lfa = []
    for pname, ptype in PERTURBATIONS:
        fn  = make_perturb(ptype) if ptype else None
        ds  = PerturbedDataset(test_ds, fn)
        dl  = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
        yt, yp = get_preds(mv2_lfa, dl)
        f1v = round(f1_score(yt, yp, average='macro', zero_division=0)*100, 1)
        rob_mv2lfa.append({'Model': 'MobileNetV2 + LFA', 'Perturbation': pname, 'Macro F1 (%)': f1v})
        print(f'  {pname:<25}: F1={f1v}%')

    # Append to existing robustness CSV
    rob_csv = SAVE_DIR / 'robustness_results.csv'
    if rob_csv.exists():
        df_existing = pd.read_csv(rob_csv)
        df_existing = df_existing[df_existing['Model'] != 'MobileNetV2 + LFA']
        df_new = pd.concat([df_existing, pd.DataFrame(rob_mv2lfa)], ignore_index=True)
    else:
        df_new = pd.DataFrame(rob_mv2lfa)
    df_new.to_csv(rob_csv, index=False)
    print(f'Saved updated: {rob_csv}')

print('\nDone. Use these results to update Table 3 and the robustness table in the paper.')
