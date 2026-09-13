# -*- coding: utf-8 -*-
"""
Standalone Grad-CAM visualisation.

Produces a three-row figure for one sample image per class:
  Row 0  Original photograph
  Row 1  ResNet-50 Grad-CAM (baseline)
  Row 2  Agri-EfficientNet LFA Grad-CAM (ours)

For the LFA model the target layer is attention_conv inside
LesionFocusAttention, so the map reflects what the attention module
actually weights.
Run: python regen_gradcam.py
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights
from torch.nn.quantized import FloatFunctional

# =============================================================================
# PATHS  — adjust if needed
# =============================================================================
SPLIT_DIR = Path(os.environ.get('SPLIT_DIR', 'data/malaysia_split'))
CKPT_DIR  = Path(os.environ.get('CKPT_DIR', 'checkpoints'))
SAVE_DIR  = Path(os.environ.get('SAVE_DIR', 'results'))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

SHARED_CLASSES = ['Algal', 'Phomopsis', 'Leaf_rot', 'Root_disease', 'Pink_disease']
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}

# =============================================================================
# TRANSFORMS
# =============================================================================
test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# =============================================================================
# MODEL DEFINITIONS (copy from main script)
# =============================================================================
class LesionFocusAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid        = nn.Sigmoid()
        self.multiply       = FloatFunctional()

    def forward(self, x):
        return self.multiply.mul(x, self.sigmoid(self.attention_conv(x)))


def build_agri_efficientnet(num_classes, attention='lfa'):
    m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False
    last_ch = m.features[-1][0].out_channels
    if attention == 'lfa':
        m.features = nn.Sequential(m.features, LesionFocusAttention(last_ch))
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                  nn.Linear(in_f, num_classes))
    return m


def build_resnet50(num_classes):
    m = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


# =============================================================================
# TARGET LAYER GETTERS — FIXED
# =============================================================================
def get_resnet_target_layer(model):
    """
    Use the last BasicBlock/Bottleneck in layer4 directly.
    register_full_backward_hook on the block itself captures gradients
    flowing into the block's output, which is stable across PyTorch versions.
    """
    return list(model.layer4.children())[-1]


def get_lfa_target_layer(model):
    """
    FIX: target the attention_conv inside LesionFocusAttention (features[-1]).
    This is the 1x1 Conv2d that produces the spatial mask — the most meaningful
    layer for showing what LFA attends to.
    """
    lfa_module = model.features[-1]          # LesionFocusAttention instance
    assert isinstance(lfa_module, LesionFocusAttention), \
        f"Expected LesionFocusAttention, got {type(lfa_module)}"
    return lfa_module.attention_conv          # Conv2d(1280, 1, 1x1)


# =============================================================================
# GRAD-CAM HELPER  — manual hook implementation, no library version issues
# =============================================================================
def get_gradcam(model, img_tensor, target_layer):
    """
    Manual GradCAM using forward/backward hooks.
    Works on any Conv2d target layer regardless of pytorch-grad-cam version.
    """
    device_ = next(model.parameters()).device
    img_t_gpu = img_tensor.unsqueeze(0).to(device_)

    activations = {}
    gradients   = {}

    def fwd_hook(module, input, output):
        # output can be a tensor or tuple
        if isinstance(output, tuple):
            activations['feat'] = output[0].detach()
        else:
            activations['feat'] = output.detach()

    def bwd_hook(module, grad_in, grad_out):
        # grad_out[0] is the gradient w.r.t. the module's output
        if grad_out[0] is not None:
            gradients['grad'] = grad_out[0].detach()

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    img_t_gpu = img_t_gpu.requires_grad_(True)

    with torch.enable_grad():
        out = model(img_t_gpu)                     # forward pass
        score = out[0].max()                       # score of predicted class
        score.backward()                           # backward pass

    h_fwd.remove()
    h_bwd.remove()

    feat = activations['feat'][0]                  # C x H x W

    if 'grad' not in gradients:
        # fallback: no gradient captured, use equal weights (plain CAM)
        weights = torch.ones(feat.shape[0], 1, 1, device=feat.device) / feat.shape[0]
    else:
        grad = gradients['grad'][0]                # C x H x W
        weights = grad.mean(dim=(1, 2), keepdim=True)  # C x 1 x 1
    cam = (weights * feat).sum(dim=0)              # H x W
    cam = torch.relu(cam)

    # Normalise to [0, 1]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam_np = cam.cpu().numpy()

    # Resize to 224x224
    import cv2
    cam_resized = cv2.resize(cam_np, (224, 224))

    # Denormalise image for overlay
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225])
              + np.array([0.485, 0.456, 0.406])).clip(0, 1).astype(np.float32)

    # Apply colormap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = (0.5 * img_np + 0.5 * heatmap).clip(0, 1)
    overlay = (overlay * 255).astype(np.uint8)

    return overlay, img_np


# =============================================================================
# LOAD MODELS
# =============================================================================
# Determine num_classes from split dir
class_names = sorted([d.name for d in (SPLIT_DIR / 'train').iterdir() if d.is_dir()])
num_classes  = len(class_names)
print(f'Classes ({num_classes}): {class_names}')

# LFA model
lfa_model = build_agri_efficientnet(num_classes, attention='lfa').to(device)
lfa_ckpt  = CKPT_DIR / 'cmp_agri_efficientnet.pth'
assert lfa_ckpt.exists(), f'LFA checkpoint not found: {lfa_ckpt}'
lfa_model.load_state_dict(torch.load(str(lfa_ckpt), map_location=device))
lfa_model.eval()
print('Loaded LFA model')

# ResNet-50 model
resnet_model = build_resnet50(num_classes).to(device)
resnet_ckpt  = CKPT_DIR / 'cmp_resnet50.pth'
assert resnet_ckpt.exists(), f'ResNet-50 checkpoint not found: {resnet_ckpt}'
resnet_model.load_state_dict(torch.load(str(resnet_ckpt), map_location=device))
resnet_model.eval()
print('Loaded ResNet-50 model')

# Target layers
resnet_tl = get_resnet_target_layer(resnet_model)
lfa_tl    = get_lfa_target_layer(lfa_model)
print(f'ResNet target layer : {type(resnet_tl).__name__}')
print(f'LFA target layer    : {type(lfa_tl).__name__} (attention_conv)')

# =============================================================================
# GENERATE FIGURE  — 3 rows × 5 cols
# Row 0: Original photo
# Row 1: ResNet-50 Grad-CAM
# Row 2: LFA (Ours) Grad-CAM
# =============================================================================
import cv2  # pip install opencv-python if missing

n_cols = len(SHARED_CLASSES)
fig, axes = plt.subplots(3, n_cols, figsize=(n_cols * 3, 9))
fig.suptitle(
    'Grad-CAM: Original | ResNet-50 (Baseline) | Agri-EfficientNet LFA (Ours)',
    fontsize=11, fontweight='bold'
)

row_labels = ['Original', 'ResNet-50', 'LFA (Ours)']
for row_idx, label in enumerate(row_labels):
    axes[row_idx, 0].set_ylabel(label, fontsize=9, rotation=90,
                                 labelpad=6, va='center')

for col, cls in enumerate(SHARED_CLASSES):
    cls_test_dir = SPLIT_DIR / 'test' / cls
    imgs = sorted([f for f in cls_test_dir.iterdir()
                   if f.suffix.lower() in IMG_EXT])
    if not imgs:
        for row in range(3):
            axes[row, col].axis('off')
            axes[row, col].text(0.5, 0.5, 'no images', ha='center',
                                fontsize=7, transform=axes[row, col].transAxes)
        continue

    img_path = imgs[0]
    img_pil  = Image.open(img_path).convert('RGB')
    img_t    = test_tf(img_pil)

    # ── Row 0: Original photo (PIL, resized to 224 for consistency) ──
    img_display = img_pil.resize((224, 224), Image.BILINEAR)
    axes[0, col].imshow(img_display)
    axes[0, col].set_title(cls, fontsize=8)
    axes[0, col].axis('off')

    # ── Row 1: ResNet-50 Grad-CAM ──
    try:
        cam_rs, _ = get_gradcam(resnet_model, img_t, resnet_tl)
        axes[1, col].imshow(cam_rs)
    except Exception as e:
        axes[1, col].text(0.5, 0.5, f'err:{str(e)[:40]}',
                          ha='center', fontsize=6, wrap=True,
                          transform=axes[1, col].transAxes)
    axes[1, col].set_title('ResNet-50', fontsize=7)
    axes[1, col].axis('off')

    # ── Row 2: LFA Grad-CAM ──
    try:
        cam_lfa, _ = get_gradcam(lfa_model, img_t, lfa_tl)
        axes[2, col].imshow(cam_lfa)
    except Exception as e:
        axes[2, col].text(0.5, 0.5, f'err:{str(e)[:40]}',
                          ha='center', fontsize=6, wrap=True,
                          transform=axes[2, col].transAxes)
    axes[2, col].set_title('LFA (Ours)', fontsize=7)
    axes[2, col].axis('off')

    print(f'  Done: {cls}')

plt.tight_layout()
out_path = SAVE_DIR / 'fig_gradcam_comparison.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'\nSaved: {out_path}')
