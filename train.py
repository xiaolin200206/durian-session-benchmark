# -*- coding: utf-8 -*-
"""
Durian disease classification under session-level evaluation
============================================================
Paper  : What a Reported Accuracy Measures: Capture-Session Leakage and
         Cross-Country Transfer in Durian Disease Classification

Pipeline:
  Step 1  — Dataset split (80/10/10, seed=42)
  Step 2  — Dataset statistics & Fig. 1
  Step 3  — Ablation study (No Attention / SE / CBAM / LFA)
  Step 4  — 10-model comparison
  Step 5  — Per-class metrics & confusion matrix
  Step 6  — Vietnam external validation
  Step 7  — Grad-CAM visualization
  Step 8  — Robustness analysis (5 perturbation conditions)
  Step 9  — McNemar statistical significance tests
  Step 10 — Grouped cross-validation (k is reduced from 5 to 3 because
            Pink_disease has only three capture sessions)
  Step 11 — LFA latency profiling

Usage:
  python train.py --split_dir clean_split --malaysia_data clean_split \\
                  --vietnam_data vietnam --sessions sessions.csv \\
                  --cv_mode group --seed 42 --no_latency

  --cv_mode group   keeps a capture session whole (the protocol the paper reports)
  --cv_mode image   the control condition, for reproducing the leakage estimate
  --retrain         force retraining; omit it to resume from checkpoints

Dataset:
  Malaysia dataset: released on Zenodo under CC BY 4.0, with per-image capture
  session identifiers. Group your partitions by the session column; an
  image-level split leaks 79.6% of images on this data and inflates macro F1
  by 12.2 points on average across nine architectures.
  Vietnam dataset: Nguyen et al. (2025), Data in Brief, under its own terms.

Requirements:
  See requirements.txt — PyTorch 2.5.1, torchvision 0.20.1, Python 3.10+
"""

# =============================================================================
# IMPORTS
# =============================================================================
from pathlib import Path
import os, time, copy, json, shutil, random, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageFilter
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import transforms, datasets
from torchvision.models import (
    EfficientNet_B0_Weights, EfficientNet_V2_S_Weights,
    VGG16_Weights, ResNet50_Weights, ResNet101_Weights,
    MobileNet_V2_Weights, MobileNet_V3_Large_Weights,
    ShuffleNet_V2_X1_0_Weights, ConvNeXt_Tiny_Weights
)
from torch.utils.data import (
    DataLoader, WeightedRandomSampler, Subset, Dataset
)
from torch.ao.nn.quantized import FloatFunctional

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)
from session_utils import (
    load_sessions, do_split_grouped, build_cv_index,
    max_usable_k, cv_folds, describe_folds,
)
from statsmodels.stats.contingency_tables import mcnemar

print(f'PyTorch    : {torch.__version__}')
print(f'Torchvision: {torchvision.__version__}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device     : {device}')
if device == 'cuda':
    print(f'GPU        : {torch.cuda.get_device_name(0)}')

# =============================================================================
# 1. CONFIG & PATHS
# =============================================================================
import argparse

parser = argparse.ArgumentParser(
    description='Session-level evaluation of durian disease classifiers. '
                'Group partitions by capture session; see README.')
parser.add_argument('--malaysia_data', type=str, default='data/malaysia',
                    help='Path to Malaysia dataset root (class subfolders)')
parser.add_argument('--vietnam_data',  type=str, default='data/vietnam',
                    help='Path to Vietnam dataset root (class subfolders)')
parser.add_argument('--split_dir',     type=str, default='data/malaysia_split',
                    help='Where to save the 80/10/10 split')
parser.add_argument('--save_dir',      type=str, default='results',
                    help='Where to save figures and CSVs')
parser.add_argument('--ckpt_dir',      type=str, default='checkpoints',
                    help='Where to save model checkpoints')
parser.add_argument('--retrain',       action='store_true', default=False,
                    help='Force retraining even if checkpoints exist')
parser.add_argument('--seed',          type=int, default=42,
                    help='Random seed for reproducibility')
parser.add_argument('--sessions',      type=str, default='sessions.csv',
                    help='Session map produced by session_split.py')
parser.add_argument('--cv_mode',       type=str, default='group',
                    choices=['group','image'],
                    help="Cross-validation split rule. 'group' keeps a capture "
                         "session whole; 'image' is the control condition.")
parser.add_argument('--no_latency',    action='store_true',
                    help='Skip single-thread CPU latency profiling')
parser.add_argument('--dry_run',       action='store_true',
                    help='2 epochs per stage, to time the run before committing')
args = parser.parse_args()

MY_DATA   = Path(args.malaysia_data)
VN_DATA   = Path(args.vietnam_data)
SPLIT_DIR = Path(args.split_dir)
SAVE_DIR  = Path(args.save_dir)
CKPT_DIR  = Path(args.ckpt_dir)
RETRAIN   = args.retrain
SEED      = args.seed
SESSIONS  = load_sessions(args.sessions)
CV_MODE   = args.cv_mode
NO_LAT    = args.no_latency
DRY       = args.dry_run
S1, S2    = (2, 2) if DRY else (15, 15)

for d in [SAVE_DIR, CKPT_DIR, SPLIT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if not MY_DATA.exists():
    raise FileNotFoundError(
        f"Malaysia data not found at: {MY_DATA}\n"
        "The dataset is on Zenodo under CC BY 4.0; download it and pass\n"
        "--malaysia_data <path>. Use the 512 px copies, not the originals:\n"
        "the two resampling paths to 224 px differ and the reported numbers\n"
        "are computed on the 512 px set."
    )
if not VN_DATA.exists():
    raise FileNotFoundError(
        f"Vietnam data not found at: {VN_DATA}\n"
        "Please obtain the Vietnam dataset from: Nguyen et al. (2025) Data in Brief.\n"
        "Specify path with --vietnam_data <path>"
    )

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

# Collection folders were named in mixed Chinese and English; this maps them
# to the ASCII class names used throughout. Kept because the Zenodo release
# ships the ASCII names and someone re-running from raw capture folders needs it.
FOLDER_RENAME = {
    'Leaf_rot(叶腐病': 'Leaf_rot',
}

CLASS_TYPE = {
    'Algal':              'Disease',
    'Leaf_rot':           'Disease',
    'Phomopsis':          'Disease',
    'Pink_disease':       'Disease',
    'Root_disease':       'Disease',
}

SHARED_CLASSES = ['Algal', 'Phomopsis', 'Leaf_rot', 'Root_disease', 'Pink_disease']

VN_TO_MY = {
    'Leaf_Algal':          'Algal',
    'Leaf_Phomopsis':      'Phomopsis',
    'Leaf_Blight':         'Leaf_rot',
    'Leaf_Colletotrichum': 'Leaf_rot',
    'Leaf_Rhizoctonia':    'Root_disease',
    # Note: Vietnam has no Pink_disease equivalent
}

COLOR_MAP = {
    'vgg16':            '#7f8c8d',
    'resnet50':         '#e74c3c',
    'resnet101':        '#c0392b',
    'mobilenetv2':      '#27ae60',
    'mobilenetv3':      '#2ecc71',
    'shufflenetv2':     '#1abc9c',
    'efficientnet_b0':  '#3498db',
    'efficientnetv2_s': '#2980b9',
    'convnext_tiny':    '#9b59b6',
    'agri_efficientnet':'#e67e22',
}

ABLATION_CONFIGS = [
    ('EfficientNet-B0 (No Attention)', 'none'),
    ('EfficientNet-B0 + SE',           'se'),
    ('EfficientNet-B0 + CBAM',         'cbam'),
    ('EfficientNet-B0 + LFA (ours)',   'lfa'),
]

MODEL_CONFIGS = [
    ('vgg16',            'VGG-16'),
    ('resnet50',         'ResNet-50'),
    ('resnet101',        'ResNet-101'),
    ('mobilenetv2',      'MobileNetV2'),
    ('mobilenetv3',      'MobileNetV3-Large'),
    ('shufflenetv2',     'ShuffleNetV2-1.0x'),
    ('efficientnet_b0',  'EfficientNet-B0 (No LFA)'),
    ('efficientnetv2_s', 'EfficientNetV2-S'),
    ('convnext_tiny',    'ConvNeXt-Tiny'),
    ('agri_efficientnet','EfficientNet-B0 + LFA (ours)'),
]

def count_imgs(folder):
    if not Path(folder).exists(): return 0
    return sum(1 for f in Path(folder).rglob('*') if f.suffix.lower() in IMG_EXT)

# =============================================================================
# 2. SPLIT MALAYSIA DATA (80/10/10)
# =============================================================================
print('\n' + '='*60)
print('STEP 1: Split Malaysia dataset (80/10/10, grouped by capture session)')
print('='*60)
do_split_grouped(MY_DATA, SPLIT_DIR, SESSIONS,
                 ratios=(0.8, 0.1, 0.1), seed=SEED,
                 folder_rename=FOLDER_RENAME)

# =============================================================================
# 3. DATASET STATISTICS
# =============================================================================
print('\n' + '='*60)
print('STEP 2: Dataset Statistics')
print('='*60)

stat_rows = []
for cls_dir in sorted((SPLIT_DIR / 'train').iterdir()):
    if not cls_dir.is_dir(): continue
    cls = cls_dir.name
    tr  = count_imgs(SPLIT_DIR / 'train' / cls)
    va  = count_imgs(SPLIT_DIR / 'val'   / cls)
    te  = count_imgs(SPLIT_DIR / 'test'  / cls)
    stat_rows.append({
        'Class': cls, 'Type': CLASS_TYPE.get(cls, 'Unknown'),
        'Train': tr, 'Val': va, 'Test': te, 'Total': tr+va+te,
        'Cross_Country': 'Yes' if cls in SHARED_CLASSES else 'No'
    })

df_stats = pd.DataFrame(stat_rows)
totals   = {'Class':'TOTAL','Type':'','Train':df_stats.Train.sum(),
            'Val':df_stats.Val.sum(),'Test':df_stats.Test.sum(),
            'Total':df_stats.Total.sum(),'Cross_Country':''}
df_stats = pd.concat([df_stats, pd.DataFrame([totals])], ignore_index=True)
print(df_stats.to_string(index=False))
df_stats.to_csv(SAVE_DIR / 'dataset_statistics.csv', index=False, encoding='utf-8-sig')

# Dataset distribution figure
df_plot = df_stats[df_stats.Class != 'TOTAL'].copy()
type_colors = {'Disease':'#e74c3c','Unknown':'#bdc3c7'}
bar_colors  = [type_colors.get(t,'#bdc3c7') for t in df_plot['Type']]
fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(range(len(df_plot)), df_plot['Total'], color=bar_colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(df_plot)))
ax.set_xticklabels(df_plot['Class'], rotation=25, ha='right', fontsize=9)
ax.set_ylabel('Image Count')
ax.set_title(f'Malaysia Durian Dataset — Class Distribution (Total: {df_plot.Total.sum()} images)')
for bar, val in zip(bars, df_plot['Total']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            str(val), ha='center', va='bottom', fontsize=8)
patches = [mpatches.Patch(color=c, label=l) for l, c in type_colors.items() if l != 'Unknown']
ax.legend(handles=patches); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(SAVE_DIR / 'fig_class_distribution.png', dpi=150)
plt.close()
print('Saved: fig_class_distribution.png')

# =============================================================================
# 4. TRANSFORMS & DATALOADERS
# =============================================================================
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomAffine(degrees=20, translate=(0.05,0.05), scale=(0.95,1.05)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=(3,7), sigma=(0.1,1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02,0.10)),
])
test_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_ds = datasets.ImageFolder(str(SPLIT_DIR/'train'), transform=train_tf)
val_ds   = datasets.ImageFolder(str(SPLIT_DIR/'val'),   transform=test_tf)
test_ds  = datasets.ImageFolder(str(SPLIT_DIR/'test'),  transform=test_tf)

class_names = train_ds.classes
num_classes = len(class_names)
print(f'\nClasses ({num_classes}): {class_names}')
print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

train_targets = [s[1] for s in train_ds.samples]
class_counts  = Counter(train_targets)
total_train   = len(train_targets)
class_weights = torch.tensor(
    [total_train/(num_classes*class_counts[i]) for i in range(num_classes)],
    dtype=torch.float).to(device)

BATCH = 16
NW    = 0  # Windows

sample_w = [class_weights[t].item() for t in train_targets]
sampler  = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler,
                          num_workers=NW, pin_memory=(device=='cuda'))
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=NW, pin_memory=(device=='cuda'))
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                          num_workers=NW, pin_memory=(device=='cuda'))

# Vietnam loader
class VietnamDataset(Dataset):
    def __init__(self, vn_root, split, class_names, vn_to_my, transform):
        self.transform  = transform
        self.cls_to_idx = {c:i for i,c in enumerate(class_names)}
        self.samples    = []
        split_dir = Path(vn_root) / split
        for vn_cls, my_cls in vn_to_my.items():
            cls_dir = split_dir / vn_cls
            if not cls_dir.exists() or my_cls not in self.cls_to_idx: continue
            label = self.cls_to_idx[my_cls]
            for f in cls_dir.rglob('*'):
                if f.suffix.lower() in IMG_EXT:
                    self.samples.append((str(f), label))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label

vn_test_ds   = VietnamDataset(VN_DATA, 'test', class_names, VN_TO_MY, test_tf)
vn_test_loader = DataLoader(vn_test_ds, batch_size=BATCH, shuffle=False, num_workers=NW)
print(f'Vietnam test : {len(vn_test_ds)} images')

# =============================================================================
# 5. MODEL DEFINITIONS
# =============================================================================
class LesionFocusAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid        = nn.Sigmoid()
        self.multiply       = FloatFunctional()
    def forward(self, x):
        return self.multiply.mul(x, self.sigmoid(self.attention_conv(x)))

class SEModule(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        mid = max(in_ch//reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(in_ch, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, in_ch, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        return x * self.fc(self.pool(x).view(b,c)).view(b,c,1,1).expand_as(x)

class CBAMModule(nn.Module):
    def __init__(self, in_ch, reduction=16, ks=7):
        super().__init__()
        mid = max(in_ch//reduction, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx  = nn.AdaptiveMaxPool2d(1)
        self.cfc = nn.Sequential(
            nn.Linear(in_ch, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, in_ch, bias=False))
        self.sconv = nn.Conv2d(2, 1, kernel_size=ks, padding=ks//2, bias=False)
        self.sig   = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        ch = self.sig(self.cfc(self.avg(x).view(b,c)) +
                      self.cfc(self.mx(x).view(b,c))).view(b,c,1,1)
        x  = x * ch
        sp = self.sig(self.sconv(torch.cat(
            [torch.mean(x,1,keepdim=True), torch.max(x,1,keepdim=True)[0]],1)))
        return x * sp

def build_agri_efficientnet(num_classes, attention='lfa'):
    m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    for p in m.parameters(): p.requires_grad = False
    last_ch = m.features[-1][0].out_channels
    if attention == 'lfa':
        m.features = nn.Sequential(m.features, LesionFocusAttention(last_ch))
    elif attention == 'se':
        m.features = nn.Sequential(m.features, SEModule(last_ch))
    elif attention == 'cbam':
        m.features = nn.Sequential(m.features, CBAMModule(last_ch))
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True), nn.Linear(in_f, num_classes))
    for p in m.classifier.parameters(): p.requires_grad = True
    if attention != 'none':
        for p in m.features[-1].parameters(): p.requires_grad = True
    return m

def build_model(name, num_classes):
    n = name.lower()
    if n == 'vgg16':
        m = models.vgg16(weights=VGG16_Weights.DEFAULT)
        for p in m.parameters(): p.requires_grad = False
        m.classifier[6] = nn.Linear(4096, num_classes)
    elif n == 'resnet50':
        m = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for p in m.parameters(): p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif n == 'resnet101':
        m = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        for p in m.parameters(): p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif n == 'mobilenetv2':
        m = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        for p in m.parameters(): p.requires_grad = False
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        for p in m.classifier.parameters(): p.requires_grad = True
    elif n == 'mobilenetv3':
        m = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        for p in m.parameters(): p.requires_grad = False
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        for p in m.classifier.parameters(): p.requires_grad = True
    elif n == 'shufflenetv2':
        m = models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.DEFAULT)
        for p in m.parameters(): p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        for p in m.fc.parameters(): p.requires_grad = True
    elif n == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        for p in m.parameters(): p.requires_grad = False
        m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                     nn.Linear(m.classifier[1].in_features, num_classes))
        for p in m.classifier.parameters(): p.requires_grad = True
    elif n == 'efficientnetv2_s':
        m = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        for p in m.parameters(): p.requires_grad = False
        m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                     nn.Linear(m.classifier[1].in_features, num_classes))
        for p in m.classifier.parameters(): p.requires_grad = True
    elif n == 'convnext_tiny':
        m = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        for p in m.parameters(): p.requires_grad = False
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        for p in m.classifier.parameters(): p.requires_grad = True
    elif n == 'agri_efficientnet':
        return build_agri_efficientnet(num_classes, attention='lfa')
    else:
        raise ValueError(f'Unknown: {name}')
    return m

# =============================================================================
# 6. TRAINING UTILITIES
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    tl, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X); loss = criterion(out, y)
        loss.backward(); optimizer.step()
        tl += loss.item()*X.size(0)
        correct += (out.argmax(1)==y).sum().item()
        total   += X.size(0)
    return tl/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    tl, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out = model(X); loss = criterion(out, y)
        tl += loss.item()*X.size(0)
        correct += (out.argmax(1)==y).sum().item()
        total   += X.size(0)
    return tl/total, correct/total

def two_stage_train(model, save_path, s1=None, s2=None, hlr=1e-3, blr=1e-5):
    s1 = S1 if s1 is None else s1
    s2 = S2 if s2 is None else s2
    crit    = nn.CrossEntropyLoss(weight=class_weights)
    history = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}
    best_va, best_st = 0.0, None

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=hlr)
    for ep in range(1, s1+1):
        tl, ta = train_one_epoch(model, train_loader, crit, opt)
        vl, va = evaluate(model, val_loader, crit)
        history['train_loss'].append(tl); history['val_loss'].append(vl)
        history['train_acc'].append(ta);  history['val_acc'].append(va)
        if va > best_va: best_va, best_st = va, copy.deepcopy(model.state_dict())
        print(f'    S1 Ep{ep:02d} | tl={tl:.4f} ta={ta:.4f} | vl={vl:.4f} va={va:.4f}')

    model.load_state_dict(best_st)
    for p in model.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=blr)
    for ep in range(1, s2+1):
        tl, ta = train_one_epoch(model, train_loader, crit, opt)
        vl, va = evaluate(model, val_loader, crit)
        history['train_loss'].append(tl); history['val_loss'].append(vl)
        history['train_acc'].append(ta);  history['val_acc'].append(va)
        if va > best_va: best_va, best_st = va, copy.deepcopy(model.state_dict())
        print(f'    S2 Ep{ep:02d} | tl={tl:.4f} ta={ta:.4f} | vl={vl:.4f} va={va:.4f}')

    model.load_state_dict(best_st)
    torch.save(best_st, save_path)
    print(f'  => Saved: {save_path}')
    return model, history

@torch.no_grad()
def get_preds(model, loader):
    model.eval()
    yt, yp = [], []
    for X, y in tqdm(loader, desc='  Predicting', leave=False):
        yp.extend(model(X.to(device)).argmax(1).cpu().numpy())
        yt.extend(y.numpy())
    return np.array(yt), np.array(yp)

def calc_metrics(yt, yp):
    return {
        'Accuracy (%)':  round(accuracy_score(yt,yp)*100, 1),
        'Precision (%)': round(precision_score(yt,yp,average='macro',zero_division=0)*100, 1),
        'Recall (%)':    round(recall_score(yt,yp,average='macro',zero_division=0)*100, 1),
        'Macro F1 (%)':  round(f1_score(yt,yp,average='macro',zero_division=0)*100, 1),
    }

def cpu_ms(model, reps=100, warmup=10):
    if NO_LAT:
        return float('nan')
    m = model.cpu().eval()
    d = torch.randn(1,3,224,224)
    with torch.no_grad():
        for _ in range(warmup): m(d)
        ts = []
        for _ in range(reps):
            t0=time.perf_counter(); m(d)
            ts.append((time.perf_counter()-t0)*1000)
    return round(np.mean(ts),2)

def load_or_train(model, ckpt_path, hist_path, label):
    if not RETRAIN and ckpt_path.exists():
        print(f'  Loading: {ckpt_path.name}')
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        with open(hist_path) as f: history = json.load(f)
    else:
        model, history = two_stage_train(model, str(ckpt_path))
        with open(hist_path,'w') as f: json.dump(history, f)
    return model, history

# =============================================================================
# 7. ABLATION STUDY
# =============================================================================
print('\n' + '='*60)
print('STEP 3: Ablation Study')
print('='*60)

ablation_results   = []
ablation_histories = {}
lfa_model = None

for name, att in ABLATION_CONFIGS:
    ckpt = CKPT_DIR / f'abl_{att}.pth'
    hist = CKPT_DIR / f'abl_{att}_hist.json'
    print(f'\n--- {name} ---')
    model = build_agri_efficientnet(num_classes, attention=att).to(device)
    model, history = load_or_train(model, ckpt, hist, name)
    ablation_histories[att] = history
    if att == 'lfa': lfa_model = model

    yt, yp = get_preds(model, test_loader)
    m = calc_metrics(yt, yp)
    size = ckpt.stat().st_size/(1024*1024) if ckpt.exists() else 0
    ablation_results.append({'Model':name,**m,'Size(MB)':round(size,2),'CPU(ms)':cpu_ms(model)})
    print(f'  => F1={m["Macro F1 (%)"]:.1f}% Acc={m["Accuracy (%)"]:.1f}%')

df_abl = pd.DataFrame(ablation_results)
print('\nABLATION RESULTS:')
print(df_abl.to_string(index=False))
df_abl.to_csv(SAVE_DIR/'ablation_results.csv', index=False)

# Ablation bar chart
colors_abl = ['#95a5a6','#27ae60','#2980b9','#e74c3c']
fig, axes = plt.subplots(1,2,figsize=(12,4))
for ax, col in [(axes[0],'Macro F1 (%)'),(axes[1],'Accuracy (%)')]:
    bars = ax.bar(range(len(df_abl)), df_abl[col], color=colors_abl, edgecolor='black')
    ax.set_xticks(range(len(df_abl)))
    ax.set_xticklabels(df_abl['Model'], rotation=15, ha='right', fontsize=8)
    ax.set_ylabel(col); ax.set_title(f'Ablation: {col}')
    ax.set_ylim(max(0,df_abl[col].min()-5),102)
    for bar,v in zip(bars,df_abl[col]):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,
                f'{v:.1f}',ha='center',fontsize=8,fontweight='bold')
plt.tight_layout()
plt.savefig(SAVE_DIR/'fig_ablation.png',dpi=150); plt.close()
print('Saved: fig_ablation.png')

# Ablation learning curves
fig, axes = plt.subplots(1,2,figsize=(14,5))
abl_colors = {'none':'#95a5a6','se':'#27ae60','cbam':'#2980b9','lfa':'#e74c3c'}
abl_labels = {'none':'No Attention','se':'SE','cbam':'CBAM','lfa':'LFA (Ours)'}
for att, label in abl_labels.items():
    if att not in ablation_histories: continue
    h = ablation_histories[att]
    ours = att=='lfa'
    axes[0].plot([a*100 for a in h['val_acc']], color=abl_colors[att],
                 lw=2.5 if ours else 1.2, ls='-' if ours else '--', label=label)
    axes[1].plot(h['val_loss'], color=abl_colors[att],
                 lw=2.5 if ours else 1.2, ls='-' if ours else '--', label=label)
for ax,title in zip(axes,['Val Accuracy (%)','Val Loss']):
    ax.set_xlabel('Epoch'); ax.set_ylabel(title); ax.set_title(title)
    ax.legend(); ax.grid(True,alpha=0.3)
    ax.axvline(x=15,color='grey',linestyle=':',alpha=0.5)
plt.tight_layout()
plt.savefig(SAVE_DIR/'fig_learning_curves.png',dpi=150); plt.close()
print('Saved: fig_learning_curves.png')

# =============================================================================
# 8. MODEL COMPARISON
# =============================================================================
print('\n' + '='*60)
print('STEP 4: Model Comparison (10 models)')
print('='*60)

compare_results   = []
compare_histories = {}

for key, label in MODEL_CONFIGS:
    ckpt = CKPT_DIR / f'cmp_{key}.pth'
    hist = CKPT_DIR / f'cmp_{key}_hist.json'
    print(f'\n[{MODEL_CONFIGS.index((key,label))+1}/{len(MODEL_CONFIGS)}] {label}')
    model = build_model(key, num_classes).to(device)
    model, history = load_or_train(model, ckpt, hist, label)
    compare_histories[key] = history

    yt, yp = get_preds(model, test_loader)
    m      = calc_metrics(yt, yp)
    size   = ckpt.stat().st_size/(1024*1024) if ckpt.exists() else 0
    params = sum(p.numel() for p in model.parameters())/1e6
    compare_results.append({
        'Model':label,**m,
        'Params(M)':round(params,2),
        'Size(MB)':round(size,2),
        'CPU(ms)':cpu_ms(model),
        '_key':key
    })
    print(f'  => F1={m["Macro F1 (%)"]:.1f}% Acc={m["Accuracy (%)"]:.1f}%')

df_cmp  = pd.DataFrame(compare_results)
df_show = df_cmp.drop(columns=['_key']).sort_values('Macro F1 (%)',ascending=False).reset_index(drop=True)
df_show.index += 1
print('\nMODEL COMPARISON TABLE:')
print(df_show.to_string())
df_show.to_csv(SAVE_DIR/'comparison_table.csv', index=False)

# Performance overview 4-panel
df_sorted  = df_cmp.sort_values('Macro F1 (%)',ascending=False).reset_index(drop=True)
bar_colors = [COLOR_MAP[k] for k in df_sorted['_key']]
our_idx    = df_sorted[df_sorted['_key']=='agri_efficientnet'].index[0]

fig, axes = plt.subplots(2,2,figsize=(16,10))
fig.suptitle('Architecture comparison, session-level partition',fontsize=14,fontweight='bold')
for metric,ax in [('Macro F1 (%)',axes[0,0]),('Accuracy (%)',axes[0,1]),
                  ('Precision (%)',axes[1,0]),('Recall (%)',axes[1,1])]:
    bars = ax.bar(range(len(df_sorted)),df_sorted[metric],
                  color=bar_colors,edgecolor='black',linewidth=0.5)
    bars[our_idx].set_linewidth(2.5)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels([r['Model'].replace('(Ours)','★') for _,r in df_sorted.iterrows()],
                        rotation=30,ha='right',fontsize=8)
    ax.set_ylabel(metric); ax.set_title(metric)
    ax.set_ylim(max(0,df_sorted[metric].min()-10),102)
    for ii,(bar,val) in enumerate(zip(bars,df_sorted[metric])):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.3,
                f'{val:.1f}',ha='center',va='bottom',fontsize=7,
                fontweight='bold' if ii==our_idx else 'normal')
plt.tight_layout()
plt.savefig(SAVE_DIR/'fig_performance_overview.png',dpi=150); plt.close()
print('Saved: fig_performance_overview.png')

# Efficiency scatter (F1 vs Size, F1 vs Speed)
fig, axes = plt.subplots(1,2,figsize=(16,6))
fig.suptitle('Accuracy–Efficiency Trade-off',fontsize=13,fontweight='bold')
for ax,xcol,xlabel in [(axes[0],'Size(MB)','Model Size (MB)'),
                        (axes[1],'CPU(ms)','CPU Inference (ms)')]:
    for _,row in df_cmp.iterrows():
        ours = row['_key']=='agri_efficientnet'
        ax.scatter(row[xcol],row['Macro F1 (%)'],
                   color=COLOR_MAP[row['_key']],s=300 if ours else 120,
                   zorder=5 if ours else 3,
                   edgecolors='black' if ours else 'grey',
                   linewidth=2.5 if ours else 0.5,
                   marker='*' if ours else 'o')
        ax.annotate(row['Model'].replace(' (Ours)','★').replace('Agri-',''),
                    (row[xcol],row['Macro F1 (%)']),
                    textcoords='offset points',xytext=(5,3),fontsize=7,
                    fontweight='bold' if ours else 'normal')
    ax.set_xlabel(xlabel); ax.set_ylabel('Macro F1 (%)')
    ax.set_title(f'F1 vs {xlabel}'); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig(SAVE_DIR/'fig_efficiency_scatter.png',dpi=150); plt.close()
print('Saved: fig_efficiency_scatter.png')

# =============================================================================
# 9. PER-CLASS METRICS
# =============================================================================
print('\n' + '='*60)
print('STEP 5: Per-Class Metrics (LFA model)')
print('='*60)

# Reload LFA model to avoid CPU/GPU mismatch after latency test
# Reload LFA model to avoid CPU/GPU mismatch after latency measurement
lfa_ckpt_reload = str(CKPT_DIR / 'abl_lfa.pth')
lfa_model = build_agri_efficientnet(num_classes, attention='lfa').to(device)
lfa_model.load_state_dict(torch.load(lfa_ckpt_reload, map_location=device))
lfa_model.eval()
yt_lfa, yp_lfa = get_preds(lfa_model, test_loader)
rpt_str  = classification_report(yt_lfa,yp_lfa,target_names=class_names,zero_division=0)
rpt_dict = classification_report(yt_lfa,yp_lfa,target_names=class_names,
                                  output_dict=True,zero_division=0)
print(rpt_str)

pc_rows = [{'Class':cls,
             'Type':CLASS_TYPE.get(cls,'?'),
             'Precision':round(rpt_dict[cls].get('precision',0)*100,1),
             'Recall':round(rpt_dict[cls].get('recall',0)*100,1),
             'F1':round(rpt_dict[cls].get('f1-score',0)*100,1),
             'Support':int(rpt_dict[cls].get('support',0)),
             'Cross_Country':'Yes' if cls in SHARED_CLASSES else 'No'}
            for cls in class_names]
df_pc = pd.DataFrame(pc_rows)
df_pc.to_csv(SAVE_DIR/'per_class_metrics.csv',index=False,encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(14,5))
pc_colors = ['#e74c3c' if r['Cross_Country']=='Yes' else '#3498db'
              for _,r in df_pc.iterrows()]
bars = ax.bar(df_pc.Class,df_pc['F1'],color=pc_colors,edgecolor='black')
macro_f1 = rpt_dict['macro avg']['f1-score']*100
ax.axhline(macro_f1,color='black',linestyle='--',label=f'Macro F1={macro_f1:.1f}%')
ax.set_ylabel('F1 (%)'); ax.set_title('Per-class F1, ablation instance')
ax.set_ylim(0,110); plt.xticks(rotation=25,ha='right')
ax.legend(handles=[mpatches.Patch(color='#e74c3c',
                   label='Disease (mapped for cross-country)')])
for bar,val in zip(bars,df_pc['F1']):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,
            f'{val:.1f}',ha='center',va='bottom',fontsize=7)
plt.tight_layout()
plt.savefig(SAVE_DIR/'fig_per_class_f1.png',dpi=150); plt.close()
print('Saved: fig_per_class_f1.png')

# Confusion matrix
cm      = confusion_matrix(yt_lfa,yp_lfa)
cm_norm = np.nan_to_num(cm.astype(float)/cm.sum(axis=1,keepdims=True))
fig, ax = plt.subplots(figsize=(12,10))
im = ax.imshow(cm_norm,interpolation='nearest',cmap=plt.cm.Blues)
plt.colorbar(im,ax=ax)
ax.set_xticks(range(num_classes)); ax.set_xticklabels(class_names,rotation=45,ha='right')
ax.set_yticks(range(num_classes)); ax.set_yticklabels(class_names)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Normalised confusion matrix, ablation instance')
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j,i,f'{cm_norm[i,j]:.2f}',ha='center',va='center',
                color='white' if cm_norm[i,j]>0.5 else 'black',fontsize=7)
plt.tight_layout()
plt.savefig(SAVE_DIR/'fig_confusion_matrix.png',dpi=150); plt.close()
print('Saved: fig_confusion_matrix.png')

# =============================================================================
# 10. VIETNAM EXTERNAL VALIDATION
# =============================================================================
print('\n' + '='*60)
print('STEP 6: Vietnam External Validation')
print('='*60)

shared_idx = [class_names.index(c) for c in SHARED_CLASSES if c in class_names]
vn_results = []

for key, label in MODEL_CONFIGS:
    ckpt = CKPT_DIR / f'cmp_{key}.pth'
    if not ckpt.exists(): continue
    model = build_model(key, num_classes).to(device)
    model.load_state_dict(torch.load(str(ckpt),map_location=device))
    yt, yp = get_preds(model, vn_test_loader)
    mask   = np.isin(yt, shared_idx)
    if mask.sum() == 0: continue
    m = calc_metrics(yt[mask], yp[mask])
    vn_results.append({'Model':label,**m,'_key':key})
    print(f'  {label:<30} F1={m["Macro F1 (%)"]:.1f}% Acc={m["Accuracy (%)"]:.1f}%')

df_vn      = pd.DataFrame(vn_results)
df_vn_show = df_vn.drop(columns=['_key']).sort_values('Macro F1 (%)',ascending=False).reset_index(drop=True)
df_vn_show.index += 1
print('\nVIETNAM VALIDATION TABLE:')
print(df_vn_show.to_string())
df_vn_show.to_csv(SAVE_DIR/'vietnam_validation.csv',index=False)

# Cross-country grouped bar chart
keys   = [k for k,_ in MODEL_CONFIGS if k in df_cmp['_key'].values]
labels = [l for k,l in MODEL_CONFIGS if k in df_cmp['_key'].values]
my_f1  = df_cmp.set_index('_key')['Macro F1 (%)']
vn_f1  = df_vn.set_index('_key')['Macro F1 (%)'] if not df_vn.empty else {}

fig, ax = plt.subplots(figsize=(16,6))
x = np.arange(len(keys)); w = 0.35
bars1 = ax.bar(x-w/2,[my_f1.get(k,0) for k in keys],w,
               label='Malaysia (Internal Test)',color='#2980b9',edgecolor='black')
bars2 = ax.bar(x+w/2,[vn_f1.get(k,0) if hasattr(vn_f1,'get') else 0 for k in keys],w,
               label='Vietnam (External Val)',color='#e67e22',edgecolor='black',alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(labels,rotation=25,ha='right',fontsize=8)
ax.set_ylabel('Macro F1 (%)'); ax.set_ylim(0,110)
ax.set_title('Cross-Country Generalization: Malaysia → Vietnam')
ax.legend(); ax.grid(axis='y',alpha=0.3)
for bar in list(bars1)+list(bars2):
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x()+bar.get_width()/2,h+0.3,
                f'{h:.1f}',ha='center',va='bottom',fontsize=6.5)
plt.tight_layout()
plt.savefig(SAVE_DIR/'fig_cross_country.png',dpi=150); plt.close()
print('Saved: fig_cross_country.png')

# =============================================================================
# 11. GRAD-CAM
# =============================================================================
print('\n' + '='*60)
print('STEP 7: Grad-CAM Visualization')
print('='*60)

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    def get_gradcam(model, img_tensor, target_layer):
        img_t_gpu = img_tensor.unsqueeze(0).to(next(model.parameters()).device)
        cam = GradCAM(model=model, target_layers=[target_layer])
        gs  = cam(input_tensor=img_t_gpu)[0]
        img_np = img_tensor.permute(1,2,0).cpu().numpy()
        img_np = (img_np*np.array([0.229,0.224,0.225])+np.array([0.485,0.456,0.406])).clip(0,1)
        return show_cam_on_image(img_np.astype(np.float32), gs, use_rgb=True), img_np

    def get_resnet_target_layer(model):
        # ResNet: last conv block in layer4
        return list(model.layer4.children())[-1]

    def get_lfa_target_layer(model):
        # LFA model: features[0] is the original EfficientNet features Sequential
        # Last block before LFA module is features[0][-1]
        # We want the last conv inside that block
        last_block = model.features[0][-1]
        # EfficientNet MBConv block - get last conv
        for name, module in reversed(list(last_block.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module
        return last_block

    lfa_model.eval()
    resnet_model = build_model('resnet50', num_classes).to(device)
    resnet_ckpt  = CKPT_DIR / 'cmp_resnet50.pth'
    if resnet_ckpt.exists():
        resnet_model.load_state_dict(torch.load(str(resnet_ckpt),map_location=device))
    resnet_model.eval()

    resnet_tl = get_resnet_target_layer(resnet_model)
    lfa_tl    = get_lfa_target_layer(lfa_model)
    print(f'  ResNet target layer: {type(resnet_tl).__name__}')
    print(f'  LFA target layer   : {type(lfa_tl).__name__}')

    fig, axes = plt.subplots(3, len(SHARED_CLASSES), figsize=(len(SHARED_CLASSES)*3, 9))
    fig.suptitle('Grad-CAM: original, ResNet-50, and EfficientNet-B0 + LFA',
                 fontsize=11, fontweight='bold')

    for col, cls in enumerate(SHARED_CLASSES):
        cls_test_dir = SPLIT_DIR / 'test' / cls
        imgs = list(cls_test_dir.glob('*.jpg')) + list(cls_test_dir.glob('*.png')) + \
               list(cls_test_dir.glob('*.jpeg'))
        if not imgs:
            for row in range(3): axes[row,col].axis('off')
            continue
        img   = Image.open(imgs[0]).convert('RGB')
        img_t = test_tf(img)

        # Row 0: original
        img_np = img_t.permute(1,2,0).cpu().numpy()
        img_np = (img_np*np.array([0.229,0.224,0.225])+np.array([0.485,0.456,0.406])).clip(0,1)
        axes[0,col].imshow(img_np); axes[0,col].set_title(cls,fontsize=8); axes[0,col].axis('off')

        # Row 1: ResNet Grad-CAM
        try:
            cam_rs, _ = get_gradcam(resnet_model, img_t, resnet_tl)
            axes[1,col].imshow(cam_rs)
        except Exception as e:
            axes[1,col].text(0.5,0.5,f'err:{str(e)[:40]}',ha='center',fontsize=6,wrap=True)
        axes[1,col].set_title('ResNet-50',fontsize=7); axes[1,col].axis('off')

        # Row 2: LFA Grad-CAM
        try:
            cam_lfa, _ = get_gradcam(lfa_model, img_t, lfa_tl)
            axes[2,col].imshow(cam_lfa)
        except Exception as e:
            axes[2,col].text(0.5,0.5,f'err:{str(e)[:40]}',ha='center',fontsize=6,wrap=True)
        axes[2,col].set_title('EfficientNet-B0 + LFA',fontsize=7); axes[2,col].axis('off')

    plt.tight_layout()
    plt.savefig(SAVE_DIR/'fig_gradcam_comparison.png',dpi=150); plt.close()
    print('Saved: fig_gradcam_comparison.png')

except ImportError:
    print('grad-cam not installed. Run: pip install grad-cam')
except Exception as e:
    print(f'Grad-CAM error: {e}')

# =============================================================================
# 12. ROBUSTNESS ANALYSIS
# =============================================================================
print('\n' + '='*60)
print('STEP 8: Robustness Analysis')
print('='*60)

class PerturbedDataset(Dataset):
    """Apply perturbation to test set images."""
    def __init__(self, base_dataset, perturb_fn=None):
        self.base = base_dataset
        self.perturb = perturb_fn
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        # Get raw PIL image path and label
        path, label = self.base.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.perturb:
            img = self.perturb(img)
        return test_tf(img), label

def make_perturb(ptype):
    """Return a PIL image perturbation function."""
    if ptype == 'gaussian_noise':
        def fn(img):
            arr = np.array(img, dtype=np.float32)
            arr = arr + np.random.normal(0, 25, arr.shape)
            return Image.fromarray(arr.clip(0,255).astype(np.uint8))
        return fn
    elif ptype == 'low_brightness':
        def fn(img):
            arr = np.array(img, dtype=np.float32) * 0.4
            return Image.fromarray(arr.clip(0,255).astype(np.uint8))
        return fn
    elif ptype == 'high_brightness':
        def fn(img):
            arr = np.array(img, dtype=np.float32) * 1.6
            return Image.fromarray(arr.clip(0,255).astype(np.uint8))
        return fn
    elif ptype == 'blur':
        def fn(img): return img.filter(ImageFilter.GaussianBlur(radius=3))
        return fn
    elif ptype == 'occlusion':
        def fn(img):
            arr = np.array(img)
            h, w = arr.shape[:2]
            # Black square occlusion (30% of image)
            size = int(min(h,w)*0.3)
            x = random.randint(0, w-size)
            y = random.randint(0, h-size)
            arr[y:y+size, x:x+size] = 0
            return Image.fromarray(arr)
        return fn
    return None

PERTURBATIONS = [
    ('Clean (Baseline)',   None),
    ('Gaussian Noise',     'gaussian_noise'),
    ('Low Brightness',     'low_brightness'),
    ('High Brightness',    'high_brightness'),
    ('Motion Blur',        'blur'),
    ('Occlusion (30%)',    'occlusion'),
]

# Models to test robustness: LFA vs ResNet-50 vs EfficientNet-B0
robust_models = {
    'EfficientNet-B0 + LFA':     (CKPT_DIR/'cmp_agri_efficientnet.pth', 'agri_efficientnet'),
    'ResNet-50 (Baseline)':      (CKPT_DIR/'cmp_resnet50.pth',          'resnet50'),
    'EfficientNet-B0 (No LFA)':  (CKPT_DIR/'cmp_efficientnet_b0.pth',   'efficientnet_b0'),
}

rob_results = {name: [] for name in robust_models}

for pname, ptype in PERTURBATIONS:
    print(f'  Perturbation: {pname}')
    perturb_fn  = make_perturb(ptype) if ptype else None
    perturb_ds  = PerturbedDataset(test_ds, perturb_fn)
    perturb_loader = DataLoader(perturb_ds, batch_size=BATCH, shuffle=False, num_workers=NW)

    for mname, (ckpt, key) in robust_models.items():
        if not ckpt.exists():
            rob_results[mname].append({'Perturbation':pname,'Macro F1 (%)':0})
            continue
        m = build_model(key, num_classes).to(device)
        m.load_state_dict(torch.load(str(ckpt),map_location=device))
        yt, yp = get_preds(m, perturb_loader)
        f1 = round(f1_score(yt,yp,average='macro',zero_division=0)*100,1)
        rob_results[mname].append({'Perturbation':pname,'Macro F1 (%)':f1})
        print(f'    {mname:<35}: F1={f1:.1f}%')

# Robustness table
rob_rows = []
for mname, results in rob_results.items():
    for r in results:
        rob_rows.append({'Model':mname,'Perturbation':r['Perturbation'],'Macro F1 (%)':r['Macro F1 (%)']})
df_rob = pd.DataFrame(rob_rows)
df_rob_pivot = df_rob.pivot(index='Model',columns='Perturbation',values='Macro F1 (%)')
print('\nROBUSTNESS TABLE (Macro F1 % under each perturbation):')
print(df_rob_pivot.to_string())
df_rob.to_csv(SAVE_DIR/'robustness_results.csv',index=False)

# Robustness line chart
fig, ax = plt.subplots(figsize=(12,5))
rob_colors = {'EfficientNet-B0 + LFA':'#e67e22',
              'ResNet-50 (Baseline)':'#e74c3c',
              'EfficientNet-B0 (No LFA)':'#3498db'}
for mname, results in rob_results.items():
    vals = [r['Macro F1 (%)'] for r in results]
    labs = [r['Perturbation'] for r in results]
    ours = 'LFA' in mname
    ax.plot(labs, vals, marker='o', lw=2.5 if ours else 1.5,
            ls='-' if ours else '--',
            color=rob_colors.get(mname,'grey'), label=mname,
            zorder=5 if ours else 3)
    if ours:
        for x,y in zip(labs,vals):
            ax.annotate(f'{y:.1f}', (x,y), textcoords='offset points',
                        xytext=(0,6), ha='center', fontsize=7, color='#e67e22')
ax.set_xlabel('Perturbation Type'); ax.set_ylabel('Macro F1 (%)')
ax.set_title('Robustness Analysis: F1 under Various Field Perturbations')
ax.legend(); ax.grid(True,alpha=0.3); plt.xticks(rotation=15,ha='right')
plt.tight_layout()
plt.savefig(SAVE_DIR/'fig_robustness.png',dpi=150); plt.close()
print('Saved: fig_robustness.png')

# =============================================================================
# 13. McNEMAR TEST
# =============================================================================
print('\n' + '='*60)
print('STEP 9: McNemar Statistical Significance Test')
print('='*60)

def run_mcnemar(yt, yp_a, yp_b, name_a, name_b):
    correct_a = (yp_a == yt)
    correct_b = (yp_b == yt)
    a = np.sum( correct_a &  correct_b)
    b = np.sum( correct_a & ~correct_b)
    c = np.sum(~correct_a &  correct_b)
    d = np.sum(~correct_a & ~correct_b)
    table  = np.array([[a,b],[c,d]])
    result = mcnemar(table, exact=False, correction=True)
    sig = 'SIGNIFICANT ✓' if result.pvalue < 0.05 else 'not significant'
    print(f'  {name_a} vs {name_b}:')
    print(f'    χ²={result.statistic:.4f}, p={result.pvalue:.6f} → {sig}')
    return {'Comparison':f'{name_a} vs {name_b}',
            'Chi2':round(result.statistic,4),
            'p-value':round(result.pvalue,6),
            'Significant (p<0.05)':'Yes' if result.pvalue<0.05 else 'No'}

# Load predictions for all ablation models
mcn_preds = {}
for name, att in ABLATION_CONFIGS:
    ckpt = CKPT_DIR / f'abl_{att}.pth'
    if not ckpt.exists(): continue
    m = build_agri_efficientnet(num_classes, attention=att).to(device)
    m.load_state_dict(torch.load(str(ckpt),map_location=device))
    yt_tmp, yp_tmp = get_preds(m, test_loader)
    mcn_preds[att] = (yt_tmp, yp_tmp)

mcn_rows = []
yt_ref = mcn_preds.get('lfa', (None,None))[0]
yp_lfa_ref = mcn_preds.get('lfa', (None,None))[1]

if yt_ref is not None:
    for name, att in ABLATION_CONFIGS:
        if att == 'lfa' or att not in mcn_preds: continue
        _, yp_other = mcn_preds[att]
        row = run_mcnemar(yt_ref, yp_lfa_ref, yp_other,
                          'EfficientNet-B0 + LFA', name)
        mcn_rows.append(row)

    # Also vs ResNet-50
    resnet_ckpt = CKPT_DIR/'cmp_resnet50.pth'
    if resnet_ckpt.exists():
        m = build_model('resnet50', num_classes).to(device)
        m.load_state_dict(torch.load(str(resnet_ckpt),map_location=device))
        _, yp_rn = get_preds(m, test_loader)
        row = run_mcnemar(yt_ref, yp_lfa_ref, yp_rn,
                          'EfficientNet-B0 + LFA', 'ResNet-50')
        mcn_rows.append(row)

df_mcn = pd.DataFrame(mcn_rows)
print('\nMcNEMAR TEST RESULTS:')
print(df_mcn.to_string(index=False))
df_mcn.to_csv(SAVE_DIR/'mcnemar_results.csv',index=False)

# =============================================================================
# 14. GROUPED CROSS VALIDATION
#     k is chosen by max_usable_k, not fixed at 5: the rarest class bounds it
# =============================================================================
print('\n' + '='*60)
print(f'STEP 10: Grouped Cross-Validation ({CV_MODE} partition)')
print('='*60)

DISEASE_CLASSES = list(class_names)  # all five classes are diseases
print(f'Disease classes for CV: {DISEASE_CLASSES}')

# Build disease-only dataset from full split
disease_idx_map = [class_names.index(c) for c in DISEASE_CLASSES if c in class_names]

# Use all Malaysia images (train+val+test combined) for CV
all_disease_samples, all_labels_cv, all_groups_cv = build_cv_index(
    SPLIT_DIR, list(class_names), SESSIONS)

print(f'Total disease images for CV: {len(all_disease_samples)}')
print(f'Distinct capture sessions  : {len(set(all_groups_cv))}')

k, sess_per_class = max_usable_k(all_labels_cv, all_groups_cv, k_wanted=5)
print('\n  sessions per class:')
for lab, n in sorted(sess_per_class.items()):
    print(f'    {class_names[lab]:<22}: {n}')
if k < 5:
    rare = min(sess_per_class, key=sess_per_class.get)
    print(f'\n  !! k reduced to {k}: "{class_names[rare]}" has only '
          f'{sess_per_class[rare]} sessions.')
    print('     Report this in the paper rather than forcing k=5.')

describe_folds(all_labels_cv, all_groups_cv, list(class_names), k,
               seed=SEED, mode=CV_MODE)


class SimpleDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


cv_rows = []

for fold, (tr_idx, va_idx, te_idx) in enumerate(
        cv_folds(all_labels_cv, all_groups_cv, k=k, seed=SEED, mode=CV_MODE)):

    print(f'\n  Fold {fold+1}/{k}')
    tr_samp = [all_disease_samples[i] for i in tr_idx]
    va_samp = [all_disease_samples[i] for i in va_idx]
    te_samp = [all_disease_samples[i] for i in te_idx]

    # label remap must be built from ALL parts, not just train, or a class
    # present only in test would shift every index
    unique_labels = sorted(set(l for _, l in tr_samp + va_samp + te_samp))
    lmap = {orig: new for new, orig in enumerate(unique_labels)}
    tr_mapped = [(p, lmap[l]) for p, l in tr_samp]
    va_mapped = [(p, lmap[l]) for p, l in va_samp]
    te_mapped = [(p, lmap[l]) for p, l in te_samp]
    fold_n_cls = len(unique_labels)

    tr_ds = SimpleDataset(tr_mapped, train_tf)
    va_ds = SimpleDataset(va_mapped, test_tf)
    te_ds = SimpleDataset(te_mapped, test_tf)

    fold_targets = [s[1] for s in tr_mapped]
    fold_counts = Counter(fold_targets)
    fold_w = torch.tensor(
        [len(fold_targets)/(fold_n_cls*max(fold_counts[i], 1))
         for i in range(fold_n_cls)], dtype=torch.float).to(device)
    fold_sw = [fold_w[t].item() for t in fold_targets]
    fold_sampler = WeightedRandomSampler(fold_sw, len(fold_sw), replacement=True)

    fold_train_loader = DataLoader(tr_ds, batch_size=BATCH,
                                   sampler=fold_sampler, num_workers=NW)
    fold_val_loader   = DataLoader(va_ds, batch_size=BATCH,
                                   shuffle=False, num_workers=NW)
    fold_test_loader  = DataLoader(te_ds, batch_size=BATCH,
                                   shuffle=False, num_workers=NW)

    fold_model = build_agri_efficientnet(fold_n_cls, attention='lfa').to(device)
    fold_crit  = nn.CrossEntropyLoss(weight=fold_w)

    def _val_acc(m):
        m.eval(); correct = total = 0
        with torch.no_grad():
            for X, y in fold_val_loader:
                X, y = X.to(device), y.to(device)
                correct += (m(X).argmax(1) == y).sum().item()
                total += y.size(0)
        return correct / max(total, 1)

    # --- Stage 1: frozen backbone, select on INNER VAL -------------------
    fold_best, fold_best_st = -1.0, None
    opt = torch.optim.Adam(
        [p for p in fold_model.parameters() if p.requires_grad], lr=1e-3)
    for ep in range(2 if DRY else 10):
        fold_model.train()
        for X, y in fold_train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad(); loss = fold_crit(fold_model(X), y)
            loss.backward(); opt.step()
        va = _val_acc(fold_model)
        if va > fold_best:
            fold_best, fold_best_st = va, copy.deepcopy(fold_model.state_dict())
    if fold_best_st is not None:
        fold_model.load_state_dict(fold_best_st)

    # --- Stage 2: full fine-tune, also selected on INNER VAL -------------
    for p in fold_model.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(fold_model.parameters(), lr=1e-5)
    stage2_best = _val_acc(fold_model)
    stage2_best_st = copy.deepcopy(fold_model.state_dict())
    for ep in range(2 if DRY else 5):
        fold_model.train()
        for X, y in fold_train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad(); loss = fold_crit(fold_model(X), y)
            loss.backward(); opt.step()
        va = _val_acc(fold_model)
        if va > stage2_best:
            stage2_best, stage2_best_st = va, copy.deepcopy(fold_model.state_dict())
    fold_model.load_state_dict(stage2_best_st)

    # --- evaluate ONCE on the untouched test fold ------------------------
    yt_f, yp_f = get_preds(fold_model, fold_test_loader)
    acc = accuracy_score(yt_f, yp_f) * 100
    f1  = f1_score(yt_f, yp_f, average='macro', zero_division=0) * 100
    cv_rows.append({'Fold': fold+1, 'cv_mode': CV_MODE, 'seed': SEED,
                    'Test images': len(te_idx),
                    'Test sessions': len(set(all_groups_cv[te_idx])),
                    'Accuracy (%)': round(acc, 2),
                    'Macro F1 (%)': round(f1, 2)})
    print(f'    Fold {fold+1} => Acc={acc:.2f}%  F1={f1:.2f}%  '
          f'(inner-val acc {stage2_best:.3f})')


df_cv = pd.DataFrame(cv_rows)
mean_acc = df_cv['Accuracy (%)'].mean()
std_acc  = df_cv['Accuracy (%)'].std()
mean_f1  = df_cv['Macro F1 (%)'].mean()
std_f1   = df_cv['Macro F1 (%)'].std()
summary_row = {'Fold':'Mean±Std',
               'Accuracy (%)':f'{mean_acc:.2f}±{std_acc:.2f}',
               'Macro F1 (%)':f'{mean_f1:.2f}±{std_f1:.2f}'}
df_cv = pd.concat([df_cv, pd.DataFrame([summary_row])], ignore_index=True)
print(f'\n{k}-FOLD GROUPED CV RESULTS (session-level, disease subset):')
print(df_cv.to_string(index=False))
df_cv.to_csv(SAVE_DIR/'cv_results.csv',index=False)

# =============================================================================
# 15. LFA LATENCY OVERHEAD
# =============================================================================
print('\n' + '='*60)
print('STEP 11: LFA Latency Overhead')
print('='*60)

lat_rows = []
if NO_LAT:
    print('  skipped (--no_latency)')
for name, att in ([] if NO_LAT else ABLATION_CONFIGS):
    ckpt = CKPT_DIR / f'abl_{att}.pth'
    m = build_agri_efficientnet(num_classes, attention=att)
    if ckpt.exists(): m.load_state_dict(torch.load(str(ckpt),map_location='cpu'))
    ms   = cpu_ms(m)
    size = ckpt.stat().st_size/(1024*1024) if ckpt.exists() else 0
    lat_rows.append({'Model':name,'CPU (ms)':ms,'Size (MB)':round(size,2)})
    print(f'  {name:<40}: {ms} ms')

df_lat = pd.DataFrame(lat_rows)
if not NO_LAT:
    base_ms = df_lat[df_lat.Model.str.contains('No Attention')]['CPU (ms)'].values[0]
    lfa_ms_ = df_lat[df_lat.Model.str.contains('LFA')]['CPU (ms)'].values[0]
    overhead = (lfa_ms_ - base_ms)/base_ms*100
    print(f'\nLFA overhead: +{lfa_ms_-base_ms:.2f}ms ({overhead:.1f}% over baseline)')
    df_lat.to_csv(SAVE_DIR/'latency_results.csv',index=False)

# =============================================================================
# 16. FINAL SUMMARY
# =============================================================================
ours_my = df_cmp[df_cmp['_key']=='agri_efficientnet'].iloc[0]

print('\n' + '='*80)
print('FINAL SUMMARY')
print('='*80)
print(f'\nMalaysia held-out test, EfficientNet-B0 + LFA:')
print(f'  Accuracy  : {ours_my["Accuracy (%)"]:.1f}%')
print(f'  Macro F1  : {ours_my["Macro F1 (%)"]:.1f}%')
print(f'  Size      : {ours_my["Size(MB)"]} MB')
print(f'  CPU       : {ours_my["CPU(ms)"]} ms/image')

print(f'\nVs other models (Malaysia F1 delta):')
for _,row in df_cmp.iterrows():
    if row['_key']=='agri_efficientnet': continue
    delta = ours_my['Macro F1 (%)']-row['Macro F1 (%)']
    print(f'  vs {row["Model"]:<30}: {delta:+.1f}%')

if not df_vn.empty:
    ours_vn_rows = df_vn[df_vn['_key']=='agri_efficientnet']
    if not ours_vn_rows.empty:
        ours_vn = ours_vn_rows.iloc[0]
        drop = ours_my['Macro F1 (%)']-ours_vn['Macro F1 (%)']
        print(f'\nVietnam External Validation:')
        print(f'  F1 = {ours_vn["Macro F1 (%)"]:.1f}% (drop = {drop:.1f}% from Malaysia)')

print(f'\n{k}-fold grouped CV: F1 = {mean_f1:.2f}% ± {std_f1:.2f}%')
if not NO_LAT:
    print(f'LFA overhead: +{lfa_ms_-base_ms:.2f}ms ({overhead:.1f}%)')

print(f'\nAll outputs saved to: {SAVE_DIR}')
print('\nFiles generated:')
for f in sorted(SAVE_DIR.glob('*')):
    print(f'  {f.name}')
print('='*80)
