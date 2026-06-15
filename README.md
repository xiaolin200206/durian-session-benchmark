# Agri-EfficientNet: Durian Disease Classification

Official implementation of:

> **Agri-EfficientNet: A Lightweight Lesion Focus Attention Framework for Durian Disease Diagnosis Under Malaysian Field Conditions with Cross-Country Generalization Assessment**  
> *Submitted to Engineering Applications of Artificial Intelligence (EAAI), 2026*

---

## Overview

Agri-EfficientNet integrates a **Lesion Focus Attention (LFA)** module into EfficientNet-B0, adding only 1,281 parameters while improving Macro F1 by 7.8 pp over the baseline on a 560-image expert-validated Malaysian field dataset.

| Model | Macro F1 | Size | CPU Latency |
|---|---|---|---|
| EfficientNet-B0 (baseline) | 80.0% | 15.60 MB | 18.22 ms |
| **Agri-EfficientNet (LFA)** | **87.8%** | **15.61 MB** | **18.22 ms** |
| MobileNetV2 + LFA | 96.4% | 8.75 MB | 56.03 ms |

---

## Dataset

The Malaysia durian disease dataset (560 images, 5 classes) was collected from commercial orchards across Peninsular Malaysia (July 2025 – June 2026) and is **not publicly released** due to commercial confidentiality agreements with collaborating farm operators.

**To request access to the dataset for research purposes**, contact the corresponding author via the journal submission system.

### Dataset structure expected by the training script

```
data/
  malaysia/
    Algal/          # 162 images
    Leaf_rot/       # 153 images
    Phomopsis/      # 157 images
    Pink_disease/   # 10 images
    Root_disease/   # 78 images
  vietnam/          # External validation set (Nguyen et al., 2025)
    Leaf_Algal/
    Leaf_Phomopsis/
    Leaf_Blight/
    Leaf_Colletotrichum/
    Leaf_Rhizoctonia/
```

The Vietnam dataset is available at: https://data.mendeley.com/datasets/... *(see paper for full citation)*

---

## Installation

```bash
git clone https://github.com/<your-username>/agri-efficientnet.git
cd agri-efficientnet
pip install -r requirements.txt
```

Tested on:
- Python 3.10 / 3.12
- PyTorch 2.5.1 + CUDA 12.4 (NVIDIA RTX 4050 Laptop GPU)
- Windows 11 / Ubuntu 22.04

---

## Usage

### 1. Run the full pipeline

```bash
python train.py \
  --malaysia_data data/malaysia \
  --vietnam_data  data/vietnam \
  --save_dir      results/ \
  --ckpt_dir      checkpoints/ \
  --seed          42
```

Set `RETRAIN = False` in `train.py` after the first run to reload saved checkpoints instead of retraining.

### 2. Regenerate Grad-CAM figure only

```bash
python regen_gradcam.py \
  --split_dir  data/malaysia_split \
  --ckpt_dir   checkpoints/
```

### 3. Run robustness analysis only

```bash
python fix_robustness.py \
  --split_dir  data/malaysia_split \
  --ckpt_dir   checkpoints/ \
  --save_dir   results/
```

---

## Repository structure

```
agri-efficientnet/
├── train.py                  # Full training + evaluation pipeline
├── regen_gradcam.py          # Standalone Grad-CAM regeneration
├── fix_robustness.py         # Robustness analysis with debug output
├── mobilenetv2_lfa.py        # MobileNetV2 + LFA supplementary experiment
├── requirements.txt
├── README.md
└── checkpoints/              # Saved model weights (see Releases)
```

---

## Pretrained Weights

Pretrained checkpoints for all 10 comparison models and 4 ablation variants are available in the [**Releases**](../../releases) section of this repository.

| File | Model | Malaysia Macro F1 |
|---|---|---|
| `cmp_agri_efficientnet.pth` | Agri-EfficientNet (LFA) | 87.8% |
| `cmp_mobilenetv2_lfa.pth` | MobileNetV2 + LFA | 96.4% |
| `cmp_convnext_tiny.pth` | ConvNeXt-Tiny | 97.6% |
| `cmp_resnet101.pth` | ResNet-101 | 96.4% |
| `cmp_mobilenetv2.pth` | MobileNetV2 | 92.7% |
| `cmp_vgg16.pth` | VGG-16 | 92.6% |
| `cmp_resnet50.pth` | ResNet-50 | 92.6% |
| `cmp_mobilenetv3.pth` | MobileNetV3-Large | 87.8% |
| `cmp_efficientnet_b0.pth` | EfficientNet-B0 (No LFA) | 80.0% |
| `cmp_efficientnetv2_s.pth` | EfficientNetV2-S | 68.2% |
| `cmp_shufflenetv2.pth` | ShuffleNetV2-1.0x | 20.8% |
| `abl_none.pth` | No Attention baseline | 82.7% F1 |
| `abl_se.pth` | SE attention | 78.9% F1 |
| `abl_cbam.pth` | CBAM attention | 85.3% F1 |
| `abl_lfa.pth` | LFA (ablation run) | 85.2% F1 |

---

## Results

### Table 1 — Ablation study (controlled identical conditions)

| Model | Accuracy | Macro F1 | Size (MB) | CPU (ms) |
|---|---|---|---|---|
| EfficientNet-B0 (No Attention) | 85.0% | 82.7% | 15.60 | 18.55 |
| EfficientNet-B0 + SE | 80.0% | 78.9% | 16.38 | 18.47 |
| EfficientNet-B0 + CBAM | 88.3% | 85.3% | 16.38 | 18.76 |
| **Agri-EfficientNet + LFA** | **88.3%** | **85.2%** | **15.60** | **18.39** |

### Table 2 — 10-model comparison (standard protocols)

| Rank | Model | Macro F1 | Size (MB) | CPU (ms) |
|---|---|---|---|---|
| 1 | ConvNeXt-Tiny | 97.6% | 106.21 | 43.07 |
| 2 | ResNet-101 | 96.4% | 162.77 | 77.09 |
| 3 | MobileNetV2 | 92.7% | 8.74 | 13.23 |
| 4 | VGG-16 | 92.6% | 512.25 | 103.39 |
| 5 | ResNet-50 | 92.6% | 90.02 | 41.71 |
| 6 | MobileNetV3-Large | 87.8% | 16.25 | 13.00 |
| 7 | **Agri-EfficientNet (Ours)** | **87.8%** | **15.61** | **18.22** |
| 8 | EfficientNet-B0 (No LFA) | 80.0% | 15.60 | 18.14 |
| 9 | EfficientNetV2-S | 68.2% | 77.86 | 50.48 |
| 10 | ShuffleNetV2-1.0x | 20.8% | 4.97 | 13.18 |

---

## Citation

```bibtex
@article{anonymous2026agriefficientnet,
  title   = {Agri-EfficientNet: A Lightweight Lesion Focus Attention Framework 
             for Durian Disease Diagnosis Under Malaysian Field Conditions 
             with Cross-Country Generalization Assessment},
  author  = {Anonymous},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

*(Citation will be updated upon acceptance.)*

---

## License

This code is released under the **MIT License**. The dataset is not included and is subject to separate confidentiality agreements.
