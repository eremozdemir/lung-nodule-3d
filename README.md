# Pulmonary Nodule Malignancy Classification from 3D CT Volumes
Erem Ozdemir

**CMPE 401 Instructor-Defined Project**

A multi-model deep learning system for lung nodule malignancy classification and lung cancer screening. Three specialized models cover three distinct input modalities: genuine 3D CT patches (NoduleMNIST3D), full-resolution LUNA16 CT volumes, and 2D chest radiographs.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Setup](#project-setup)
3. [Notebooks](#notebooks)
4. [Datasets](#datasets)
5. [Model Architectures](#model-architectures)
6. [Trial History and Results](#trial-history-and-results)
7. [Best Results (Trial 7)](#best-results-trial-7)
8. [Key Findings](#key-findings)
9. [Project Structure](#project-structure)
10. [References](#references)

---

## Project Overview

Lung cancer is the leading cause of cancer death in Canada. 32,900 diagnoses and 19,400 deaths are projected in 2025, resulting in roughly 54 people every day. Early detection is the single most effective intervention as 5-year survival rates jump from ~10% (stage IV) to ~60% (stage I) when caught early. Pulmonary nodules are common incidental findings on CT scans, but the majority are benign. Manual radiologist review of every nodule is time-consuming and subject to inter-reader variability.

This project builds a fast, reproducible deep learning pipeline that assigns a malignancy probability score to a nodule patch, supporting triage and second-opinion use cases. The pipeline is designed around a domain-specific architecture strategy where different input modalities require fundamentally different models, and mixing them degrades performance on the primary clinical benchmark.

**Primary benchmark:** NoduleMNIST3D AUROC (standard MedMNIST evaluation protocol)

### For full analysis, experimental motivation, and statistical interpretation, please see both:
- [Project Report](Report/Pulmonary%20Nodule%20Malignancy%20Classification%20and%20Lung%20Cancer%20Screening%20with%20Specialized%203D%20and%202D%20Deep%20Learning%20Models.pdf)
- [Implementation Notes](Implementation_notes.md)

---

## Project Setup

1. Create and activate a virtual environment:
```bash
python3.11 -m venv .venv && source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Register the kernel for Jupyter:
```bash
python -m ipykernel install --user --name lung-nodule-3d
```

4. Reload the window in VS Code:

   a. Open the Command Palette:
   - macOS: `Cmd + Shift + P`
   - Windows / Linux: `Ctrl + Shift + P`

   b. Type `Developer: Reload Window` and press Enter

5. Select the kernel in VS Code: click the kernel selector (top right of notebook) → **Python Environments** → select `.venv`

### Data Setup

The following datasets must be placed under `data/`:

| Dataset | Format | Path |
|---|---|---|
| NoduleMNIST3D | Downloaded automatically via MedMNIST API | auto |
| LUNA16 | `.mhd` / `.raw` CT volumes | `data/Luna16/` |
| IQ-OTH:NCCD | JPEG image folders | `data/IQ-OTH:NCCD/` |
| LungcancerDataSet | JPEG image folders | `data/LungcancerDataSet/` |
| Chest X-ray | JPEG train/test split folders | `data/chest_xray_lung/` |

NoduleMNIST3D downloads automatically on first run. LUNA16 requires registration at [luna16.grand-challenge.org](https://luna16.grand-challenge.org/). The remaining datasets are available on Kaggle.

---

## Notebooks

| Notebook | Description |
|---|---|
| [notebooks/01_nodulemnist3d.ipynb](notebooks/01_nodulemnist3d.ipynb) | Full pipeline: data loading, EDA, training all three models, evaluation, and visualization |

The notebook is structured as a self-contained report with markdown section headers. Run all cells top-to-bottom in a single session. Training checkpoints are saved to `results/runs/TrialXX_.../` automatically.

---

## Datasets

### NoduleMNIST3D
- **Source:** MedMNIST v2 (Yang et al., 2023), derived from the LIDC-IDRI CT collection
- **Task:** Binary malignancy classification (benign=0, malignant=1)
- **Format:** 28×28×28 voxel patches, pre-normalized to uint8 [0, 255], genuine 3D spatial structure
- **Labels:** Aggregated radiologist malignancy rating ≥ 3 = malignant
- **Splits:** Train 1,158 | Val 165 | Test 310

### LUNA16
- **Source:** Lung Nodule Analysis 2016 challenge (LIDC-IDRI, full resolution)
- **Task:** Nodule presence detection (nodule vs. background tissue)
- **Format:** Full 3D CT volumes in `.mhd/.raw`, Hounsfield Unit (HU) values. Patches extracted at annotated nodule coordinates and normalized per-patch using percentile clipping [1st, 99th] → [0, 255]
- **Splits:** ~1,560 train | ~195 val | ~195 test (80/10/10 stratified)

### IQ-OTH:NCCD
- **Source:** Kaggle IQ-OTH/NCCD Lung Cancer Dataset
- **Task:** Binary cancer classification (cancer / non-cancer)
- **Format:** 2D JPEG CT screenshots, routed to the 2D CXRClassifier
- **Note:** No genuine 3D structure, same slice repeated 28× in earlier trials; removed from 3D training in Trial 5

### LungcancerDataSet
- **Source:** Kaggle
- **Task:** Binary cancer classification
- **Format:** 2D JPEG images, routed to the 2D CXRClassifier from Trial 5 onward

### Chest X-ray (Lung Cancer)
- **Source:** Kaggle
- **Task:** Cancer vs. normal from frontal chest radiographs
- **Format:** Grayscale JPEG, train/test folder split
- **Splits (internal):** Train 4,433 | Val 783 | Test 624

---

## Model Architectures

The architecture strategy is: **one model per input modality**. Mixing modalities causes domain calibration problems that degrade performance on the primary benchmark.

### Deep3DCNN on NoduleMNIST3D Dataset (genuine 3D CT patches)

**Motivation:** The primary clinical model. NoduleMNIST3D patches contain real 3D volumetric structure (density gradients, spiculation, boundary texture across all three axes). A deep wide architecture with SE attention extracts the subtle morphological features that radiologists use for malignancy grading. Two-phase training (LUNA16 pretraining → NoduleMNIST3D fine-tuning) initializes weights with volumetric priors before fine-tuning on the smaller benchmark.

```
Input (B, 1, 28, 28, 28)
    │
    ├── Stem: Conv3d(1→32) + BN + ReLU
    ├── SEResBlock(32→64)  + MaxPool  → (B, 64,  14³)
    ├── SEResBlock(64→128) + MaxPool  → (B, 128,  7³)
    ├── SEResBlock(128→256)+ MaxPool  → (B, 256,  3³)
    ├── SEResBlock(256→256)           → (B, 256,  3³)
    └── GAP → Dropout(0.5) → FC(256→1) → logit
```
- **Parameters:** ~3.7 M
- **File:** [`src/model3d_deep.py`](src/model3d_deep.py)

---

### LUNA3DCNN on LUNA16 Dataset (full-resolution 3D CT nodule detection)

**Motivation:** LUNA16 poses a detection task (nodule vs. uniform parenchyma background) rather than malignancy grading; a cleaner decision boundary. A dedicated, simpler architecture (no SE attention, lower dropout) avoids the logit-scale calibration conflict that damaged NoduleMNIST3D performance when LUNA16 was mixed into joint training. Trained and threshold-tuned exclusively on LUNA16.

```
Input (B, 1, 28, 28, 28)
    │
    ├── Stem: Conv3d(1→32) + BN + ReLU
    ├── ResBlock3D(32→64)  + MaxPool  → (B, 64,  14³)
    ├── ResBlock3D(64→128) + MaxPool  → (B, 128,  7³)
    ├── ResBlock3D(128→256)+ MaxPool  → (B, 256,  3³)
    └── GAP → Dropout(0.3) → FC(256→1) → logit
```
- **Parameters:** ~2.0 M
- **File:** [`src/model3d_luna.py`](src/model3d_luna.py)

---

### CXRClassifier on 2D Chest Radiographs Dataset

**Motivation:** IQ-OTH:NCCD and LungcancerDataSet are 2D JPEG images with no genuine depth variation. A 3D CNN extracts nothing meaningful in the Z-axis for these inputs. A 2D ResNet-18 with ImageNet-pretrained weights provides strong texture and shape priors that transfer well to medical images with minimal fine-tuning data. This model serves the low-cost, widely deployable end of the screening pipeline: chest X-rays are available in most clinics worldwide, making it relevant for rural or resource-limited settings where CT is not accessible. Temperature scaling calibrates output probabilities post-training.

```
Input (B, 1, H, W)
    │
    ├── ResNet-18 backbone (ImageNet pretrained, 1-ch grayscale first conv)
    ├── Global Average Pooling
    ├── Dropout(0.5)
    └── FC(512→1) → logit  [temperature-scaled post-training]
```
- **Parameters:** ~11.2 M
- **File:** [`src/model_cxr.py`](src/model_cxr.py)

---

### Small3DCNN Legacy Model (Trials 0–3)

Lightweight baseline used in Trials 0–3. Designed to run on CPU-only hardware (~884K params). Replaced by the domain-specific strategy from Trial 5 onward.

```
Input (B, 1, 28, 28, 28)
    │
    ├── Stem: Conv3d(1→16) + BN + ReLU
    ├── ResBlock3D(16→32) + MaxPool  → (B, 32, 14³)
    ├── ResBlock3D(32→64) + MaxPool  → (B, 64,  7³)
    ├── ResBlock3D(64→128)           → (B, 128, 7³)
    └── GAP → Dropout(0.3) → FC(128→1) → logit
```
- **Parameters:** ~884 K
- **File:** [`src/model3d_small.py`](src/model3d_small.py)

---

## Trial History and Results

Eight successive trials on NoduleMNIST3D AUROC (primary benchmark). Each trial introduced targeted changes to address the previous trial's identified bottleneck.

| Trial | Key Change | NoduleMNIST3D AUROC | Notes |
|---|---|---|---|
| 0 | Baseline Small3DCNN, fixed thr=0.50 | 0.849 | Starting point |
| 1 | LR scheduling + validation threshold tuning | 0.871 | +0.022 |
| 2 | ResBlock3D skip connections | 0.894 | +0.023, better gradient flow |
| 3 | More data, stronger augmentation, longer training | **0.922** | Best Small3DCNN result |
| 4 | Deep3DCNN + LUNA16 joint training | 0.830 | Regression HU vs. MedMNIST domain mismatch; threshold jumped to 0.85 |
| 5 | Fixed LUNA16 preprocessing (percentile norm), dual-model | 0.847 | Domain mismatch partially resolved |
| 6 | NoduleMNIST3D-only Deep3DCNN, cutout augmentation | 0.915 | Best standalone Deep3DCNN (no LUNA16) |
| 7 | SE attention, two-phase training, dedicated LUNA3DCNN | **0.906** | Best F1/precision; LUNA3DCNN AUROC 0.991 |

Full experimental notes, per-metric tables, and confusion matrices for every trial are in [Implementation_notes.md](Implementation_notes.md).

---

## Best Results (Trial 7)

### Deep3DCNN NoduleMNIST3D Test Set (thr = 0.85)

| Metric | Value |
|---|---|
| AUROC | **0.906** |
| F1 | **0.700** |
| Recall | 0.766 |
| Precision | 0.645 |
| Specificity | 0.890 |
| Accuracy | 0.865 |

Confusion matrix: TN=219  FP=27  FN=15  TP=49

### LUNA3DCNN LUNA16 Test Set (thr = 0.55)

| Metric | Value |
|---|---|
| AUROC | **0.991** |
| F1 | **0.934** |
| Recall | 0.934 |
| Precision | 0.934 |
| Specificity | 0.970 |
| Accuracy | 0.959 |

Confusion matrix: TN=130  FP=4  FN=4  TP=57

### CXRClassifier Chest X-ray Test Set (thr = 0.25, temperature-scaled)

| Metric | Value |
|---|---|
| AUROC | **0.889** |
| F1 | **0.883** |
| Recall | **0.997** |
| Precision | 0.792 |
| Specificity | 0.564 |
| Accuracy | 0.835 |

Confusion matrix: TN=132  FP=102  FN=1  TP=389

---

## Key Findings

**1. Domain mismatch is the dominant failure mode**

Mixing LUNA16 (HU-normalized CT patches) with NoduleMNIST3D (MedMNIST percentile-normalized patches) caused the threshold to jump to 0.85 and NoduleMNIST3D AUROC to drop from 0.922 to 0.830 (Trial 4). Fix: per-patch percentile normalization for LUNA16, plus completely separate models per dataset.

**2. One model per task outperforms any joint architecture**

The dedicated LUNA3DCNN achieves AUROC 0.991 on LUNA16, the previous best from any joint model was 0.960. Architecture complexity should match task difficulty: nodule detection (spherical blob vs. background) is simpler than malignancy grading, so a lighter model with lower dropout is optimal.

**3. Pseudo-3D inputs need a 2D model**

IQ-OTH:NCCD and LungcancerDataSet are 2D JPEG slices with no depth variation. Routing them to a 2D ResNet-18 with ImageNet weights achieves AUROC 0.889 and near-perfect recall (0.997), while also making deployment tractable on standard hospital hardware.

**4. Two-phase training improves the operating point**

LUNA16 pretraining → NoduleMNIST3D fine-tuning (Trial 7) improves F1 (+0.029), precision (+0.047), and specificity (+0.024) vs. NoduleMNIST3D-only training (Trial 6), at a marginal AUROC cost (-0.009). Better precision and specificity mean fewer false positives per true malignant flagged, which is the clinically preferable tradeoff.

**5. Label smoothing and early stopping are critical regularizers**

NoduleMNIST3D has only 1,158 training samples. Without label smoothing (ε=0.05) and early stopping (patience=10 on val AUROC), the Deep3DCNN consistently overfits before reaching its best generalization.

---

## Project Structure

```
lung-nodule-3d/
├── README.md
├── requirements.txt
├── Implementation_notes.md          # Detailed trial-by-trial experimental log
├── notebooks/
│   └── 01_nodulemnist3d.ipynb       # Full pipeline notebook (training + evaluation)
├── src/
│   ├── model3d_deep.py              # Deep3DCNN with SEResBlock3D (NoduleMNIST3D)
│   ├── model3d_luna.py              # LUNA3DCNN (LUNA16 nodule detection)
│   ├── model3d_small.py             # Small3DCNN legacy (Trials 0–3)
│   ├── model3d.py                   # Backward-compat shim
│   ├── model_cxr.py                 # CXRClassifier — ResNet-18 for 2D radiographs
│   ├── dataset_utils.py             # Dataset loaders for all 5 input sources
│   ├── train_utils.py               # Training loop, early stopping, checkpointing
│   └── metrics.py                   # AUROC, F1, threshold sweep utilities
├── data/
│   ├── Luna16/                      # LUNA16 .mhd/.raw CT volumes (register to download)
│   ├── IQ-OTH:NCCD/                 # 2D JPEG CT screenshots
│   ├── LungcancerDataSet/           # 2D JPEG images
│   └── chest_xray_lung/             # 2D chest X-ray JPEG images
├── results/
│   └── runs/
│       ├── Trial00_.../             # Per-trial: figures, metrics JSON, checkpoints
│       ├── Trial01_.../
│       └── ...Trial07_.../
└── Report/
    └── Pulmonary Nodule Malignancy Classification...pdf
```

---

## References

- Yang, J. et al. (2023). *MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification*. Scientific Data.
- Armato, S. G. et al. (2011). *The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)*. Medical Physics.
- Setio, A. A. A. et al. (2017). *Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in CT images: the LUNA16 challenge*. Medical Image Analysis.
- Hu, J. et al. (2018). *Squeeze-and-Excitation Networks*. CVPR.
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
- Canadian Cancer Society. (2025). *Canadian Cancer Statistics*.
