# -*- coding: utf-8 -*-
"""
fix_robustness.py
=================
Runs the robustness analysis on its own, without retraining anything.

Applies five synthetic field perturbations to the test set and reports
Macro F1 under each. Verbose output confirms each perturbation is
actually applied to the input tensors.

Writes robustness_results.csv and fig_robustness.png.

Run: python fix_robustness.py
"""

import os
from pathlib import Path
import warnings, random
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
from torch.utils.data import DataLoader, Dataset
from torch.nn.quantized import FloatFunctional
from sklearn.metrics import f1_score

# ── PATHS (same as main script) ───────────────────────────────────────────────
SPLIT_DIR = Path(os.environ.get('SPLIT_DIR', 'data/malaysia_split'))
SAVE_DIR  = Path(os.environ.get('SAVE_DIR', 'results'))
CKPT_DIR  = Path(os.environ.get('CKPT_DIR', 'checkpoints'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# ── TRANSFORMS ────────────────────────────────────────────────────────────────
test_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_ds = datasets.ImageFolder(str(SPLIT_DIR / 'test'), transform=test_tf)
class_names = test_ds.classes
num_classes  = len(class_names)
print(f'Classes ({num_classes}): {class_names}')
print(f'Test set: {len(test_ds)} images')

# ── PERTURBATION FUNCTIONS ────────────────────────────────────────────────────
def make_perturb(ptype):
    if ptype == 'gaussian_noise':
        def fn(img):
            arr = np.array(img, dtype=np.float32)
            arr = arr + np.random.normal(0, 25, arr.shape)
            return Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        return fn
    elif ptype == 'low_brightness':
        def fn(img):
            arr = np.array(img, dtype=np.float32) * 0.4
            return Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        return fn
    elif ptype == 'high_brightness':
        def fn(img):
            arr = np.array(img, dtype=np.float32) * 1.6
            return Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        return fn
    elif ptype == 'blur':
        def fn(img): return img.filter(ImageFilter.GaussianBlur(radius=3))
        return fn
    elif ptype == 'occlusion':
        def fn(img):
            arr = np.array(img)
            h, w = arr.shape[:2]
            size = int(min(h, w) * 0.3)
            x = random.randint(0, w - size)
            y = random.randint(0, h - size)
            arr[y:y+size, x:x+size] = 0
            return Image.fromarray(arr)
        return fn
    return None

# ── DATASET WITH PERTURBATION ─────────────────────────────────────────────────
class PerturbedDataset(Dataset):
    def __init__(self, base_dataset, perturb_fn=None):
        self.base    = base_dataset
        self.perturb = perturb_fn

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        path, label = self.base.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.perturb:
            img = self.perturb(img)
        return test_tf(img), label

# ── DEBUG: verify perturbation actually changes pixels ────────────────────────
print('\n=== DEBUG: verifying perturbations change image tensors ===')
path0, _ = test_ds.samples[0]
img0     = Image.open(path0).convert('RGB')
t_clean  = test_tf(img0)

for ptype in ['gaussian_noise', 'low_brightness', 'high_brightness', 'blur', 'occlusion']:
    fn   = make_perturb(ptype)
    t_p  = test_tf(fn(img0))
    diff = (t_clean - t_p).abs().max().item()
    print(f'  {ptype:<20}: max pixel diff after test_tf = {diff:.4f}  '
          f'{"OK" if diff > 0.001 else "WARNING: NO CHANGE"}')
print()

# ── MODEL DEFINITIONS ─────────────────────────────────────────────────────────
class LesionFocusAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid        = nn.Sigmoid()
        self.multiply       = FloatFunctional()
    def forward(self, x):
        return self.multiply.mul(x, self.sigmoid(self.attention_conv(x)))

def build_agri_efficientnet(num_classes):
    m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    for p in m.parameters(): p.requires_grad = False
    last_ch = m.features[-1][0].out_channels
    m.features = nn.Sequential(m.features, LesionFocusAttention(last_ch))
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True), nn.Linear(in_f, num_classes))
    return m

def build_efficientnet_b0(num_classes):
    m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    for p in m.parameters(): p.requires_grad = False
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True), nn.Linear(in_f, num_classes))
    return m

def build_resnet50(num_classes):
    m = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    for p in m.parameters(): p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

@torch.no_grad()
def get_preds(model, loader):
    model.eval()
    yt, yp = [], []
    for X, y in tqdm(loader, desc='  Predicting', leave=False):
        yp.extend(model(X.to(device)).argmax(1).cpu().numpy())
        yt.extend(y.numpy())
    return np.array(yt), np.array(yp)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
robust_models = {
    'Agri-EfficientNet (LFA)':  (CKPT_DIR/'cmp_agri_efficientnet.pth', build_agri_efficientnet),
    'ResNet-50 (Baseline)':     (CKPT_DIR/'cmp_resnet50.pth',          build_resnet50),
    'EfficientNet-B0 (No LFA)': (CKPT_DIR/'cmp_efficientnet_b0.pth',   build_efficientnet_b0),
}

PERTURBATIONS = [
    ('Clean (Baseline)',  None),
    ('Gaussian Noise',    'gaussian_noise'),
    ('Low Brightness',    'low_brightness'),
    ('High Brightness',   'high_brightness'),
    ('Motion Blur',       'blur'),
    ('Occlusion (30%)',   'occlusion'),
]

BATCH = 16
rob_results = {name: [] for name in robust_models}

for pname, ptype in PERTURBATIONS:
    print(f'  Perturbation: {pname}')
    perturb_fn     = make_perturb(ptype) if ptype else None
    perturb_ds     = PerturbedDataset(test_ds, perturb_fn)
    perturb_loader = DataLoader(perturb_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    for mname, (ckpt, build_fn) in robust_models.items():
        if not ckpt.exists():
            print(f'    {mname}: checkpoint missing, skipping')
            rob_results[mname].append({'Perturbation': pname, 'Macro F1 (%)': 0})
            continue

        # ── Build fresh model each time, no caching ──────────────────────────
        m = build_fn(num_classes).to(device)
        m.load_state_dict(torch.load(str(ckpt), map_location=device))
        m.eval()

        yt, yp = get_preds(m, perturb_loader)
        f1 = round(f1_score(yt, yp, average='macro', zero_division=0) * 100, 1)
        rob_results[mname].append({'Perturbation': pname, 'Macro F1 (%)': f1})
        print(f'    {mname:<35}: F1={f1:.1f}%')

        # ── DEBUG: print first 5 predictions to verify they change ───────────
        if ptype is not None:
            print(f'      (first 5 preds: {yp[:5]})')

print()

# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
rob_rows = []
for mname, results in rob_results.items():
    for r in results:
        rob_rows.append({'Model': mname, 'Perturbation': r['Perturbation'], 'Macro F1 (%)': r['Macro F1 (%)']})

df_rob = pd.DataFrame(rob_rows)
df_rob.to_csv(SAVE_DIR / 'robustness_results.csv', index=False)
print(f'Saved: {SAVE_DIR}/robustness_results.csv')

df_pivot = df_rob.pivot(index='Model', columns='Perturbation', values='Macro F1 (%)')
print('\nROBUSTNESS TABLE:')
print(df_pivot.to_string())

# ── PLOT ──────────────────────────────────────────────────────────────────────
porder  = ['Clean (Baseline)', 'Gaussian Noise', 'Low Brightness',
           'High Brightness', 'Motion Blur', 'Occlusion (30%)']
colors  = {'Agri-EfficientNet (LFA)': '#e67e22',
           'ResNet-50 (Baseline)':    '#e74c3c',
           'EfficientNet-B0 (No LFA)':'#3498db'}
styles  = {'Agri-EfficientNet (LFA)': '-',
           'ResNet-50 (Baseline)':    '--',
           'EfficientNet-B0 (No LFA)':'-'}

fig, ax = plt.subplots(figsize=(12, 5))
for mname, results in rob_results.items():
    r_dict = {r['Perturbation']: r['Macro F1 (%)'] for r in results}
    ys = [r_dict[p] for p in porder]
    ax.plot(porder, ys,
            marker='o', label=mname,
            color=colors[mname], linestyle=styles[mname], linewidth=2)
    for x, y in zip(porder, ys):
        ax.annotate(f'{y}', (x, y), textcoords='offset points',
                    xytext=(0, 7), ha='center', fontsize=8,
                    color=colors[mname])

ax.set_xlabel('Perturbation Type')
ax.set_ylabel('Macro F1 (%)')
ax.set_title('Robustness Analysis: F1 under Various Field Perturbations')
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(SAVE_DIR / 'fig_robustness.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {SAVE_DIR}/fig_robustness.png')
print('\nDone. Check the table above — ResNet-50 should now show variation across perturbations.')
