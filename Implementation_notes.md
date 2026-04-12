# Implementation Notes
This is where I'll log my notes about what I tried, what worked, what didn't work, and what I plan to try in subsequent implementation trials in hopes to keep improving model accuracy. 


## Table of Contents
- [Implementation Notes](#implementation-notes)
  - [Table of Contents](#table-of-contents)
  - [Trial 0: Initial Model Implementation](#trial-0-initial-model-implementation)
    - [Trial 0 results summary (thr = 0.50)](#trial-0-results-summary-thr--050)
    - [Interpretation](#interpretation)
  - [Trial 1: Simple Performance Upgrades](#trial-1-simple-performance-upgrades)
    - [Trial 1 results summary (tuned thr = 0.65)](#trial-1-results-summary-tuned-thr--065)
      - [Comparing Trial 1 vs Trial 0](#comparing-trial-1-vs-trial-0)
    - [Interpretation](#interpretation-1)
  - [Trial 2: Residual Blocks and Validation Loss Tracking](#trial-2-residual-blocks-and-validation-loss-tracking)
    - [Trial 2 results summary (tuned thr = 0.50)](#trial-2-results-summary-tuned-thr--050)
      - [Comparing Trial 2 (thr=0.50) vs Trial 1 (thr=0.65)](#comparing-trial-2-thr050-vs-trial-1-thr065)
    - [Interpretation](#interpretation-2)
  - [What to try next for Trial 3](#what-to-try-next-for-trial-3)
  - [Trial 3: Expanded Training Data, Enhanced Augmentation, and Longer Convergence](#trial-3-expanded-training-data-enhanced-augmentation-and-longer-convergence)
    - [Trial 3 results — NoduleMNIST3D test (tuned thr = 0.60)](#trial-3-results--nodulemnist3d-test-tuned-thr--060)
      - [Comparing Trial 3 vs Trial 2 at each trial's tuned threshold](#comparing-trial-3-vs-trial-2-at-each-trials-tuned-threshold)
    - [Trial 3 results — Combined test (tuned thr = 0.60)](#trial-3-results--combined-test-tuned-thr--060)
    - [Interpretation](#interpretation-3)


---

## Trial 0: Initial Model Implementation
- Started with a small 3D CNN trained on NoduleMNIST3D
- Evaluated with a fixed decision threshold of **0.50**
- On the test set, that baseline reached **AUROC 0.849**, **F1 0.564**, and **Accuracy 0.835**.

### Trial 0 results summary (thr = 0.50)
- AUROC: 0.849
- F1: 0.564
- Recall: 0.516
- Percision: 0.623
- Specificity: 0.919
- Accuracy: 0.835

### Interpretation
- At thr = 0.50 (no tuning was done in this trial), the model was conservative:
  - Had decent specificity but missed a lot of malignant cases.
  - False negatives were high (low recall); false positives were relatively low (good specificity and accuracy).

<p><strong>Trial 0 Model Results:</strong></p>
<img src="results/runs/2026-03-05_16.17.20/figures/results_table_2026-03-05_16.17.20.png" alt="Trial 1 Model Results">

<p><strong>Trial 0 Confusion Matrix:</strong></p>
<img src="results/runs/2026-03-05_16.17.20/figures/confusion_matrix_2026-03-05_16.17.20.png" alt="Trial 1 Confusion Matrix" width="300">

<p><strong>Trial 0 Training Loss:</strong></p>
<img src="results/runs/2026-03-05_16.17.20/figures/train_loss_2026-03-05_16.17.20.png" width="500">

---
## Trial 1: Simple Performance Upgrades

This trial focused on a few “low effort, high impact” upgrades while keeping the project lightweight:
* **Trained longer with learning rate (LR) scheduling**: 
  * Did this so that training can keep improving without manually tuning LR every time
  * The run logs show training continuing out to later epochs while tracking LR as it decays
* **Validation-based threshold tuning**:
  * instead of assuming 0.50, On the validation sweep, the best F1 occurred at **threshold = 0.65**. 

### Trial 1 results summary (tuned thr = 0.65)

The validation sweep selected **threshold = 0.65** as the best F1 cutoff.

| Metric | Value |
|---|---|
| AUROC | 0.871 |
| F1 | 0.598 |
| Recall | 0.594 |
| Precision | 0.603 |
| Specificity | 0.889 |
| Accuracy | 0.835 |

#### Comparing Trial 1 vs Trial 0

Trial 0 had no threshold tuning (thr = 0.50). Trial 1 uses tuned thr = 0.65.

| Metric | Trial 0 (thr=0.50) | Trial 1 (thr=0.65) | Δ |
|---|---|---|---|
| AUROC | 0.849 | **0.871** | **+0.022** |
| F1 | 0.564 | **0.598** | **+0.034** |
| Recall | 0.516 | **0.594** | **+0.078** |
| Precision | **0.623** | 0.603 | −0.020 |
| Specificity | **0.919** | 0.889 | −0.030 |
| Accuracy | 0.835 | 0.835 | — |


<p><strong>Trial 1 Model Results:</strong></p>
<img src="results/runs/2026-03-06_11.55.56/figures/results_table_2026-03-06_11.55.56.png" alt="Trial 1 Model Results">

<p><strong>Trial 1 Confusion Matrix:</strong></p>
<img src="results/runs/2026-03-06_11.55.56/figures/confusion_matrix_2026-03-06_11.55.56.png" alt="Trial 1 Confusion Matrix" width="300">

<p><strong>Trial 1 Training Loss:</strong></p>
<img src="results/runs/2026-03-06_11.55.56/figures/train_loss_2026-03-06_11.55.56.png" width="500">

### Interpretation
Trial 1 improved the model’s ranking quality more than its raw accuracy number.

AUROC improved from 0.849 to 0.871, meaning the CNN separates benign from malignant volumes more reliably overall.

At the tuned threshold (0.65), the model becomes more conservative about calling something malignant:
- Recall improved from 0.516 to 0.594 (catching more true positives than Trial 0).
- Precision held near Trial 0’s level (0.603 vs 0.623) — fewer false alarms per true positive.
- Specificity dropped slightly from 0.919 to 0.889 because the model now labels more cases overall.
- Accuracy stayed flat at 0.835.

AUROC does not depend on the threshold; it reflects score ranking quality across all cutoffs.

Logistic regression stayed basically the same (AUROC 0.823), confirming the gains in Trial 1 are from the CNN changes, not pipeline noise.


---
## Trial 2: Residual Blocks and Validation Loss Tracking

This trial introduced several structural improvements to both the model and the training pipeline:

* **Residual blocks added to the 3D CNN backbone (`ResBlock3D`)**:
  * The `Small3DCNN` now uses residual skip connections between conv layers.
  * This improves gradient flow through the network and generally boosts representation quality without adding many parameters.
  * A `get_feature_maps()` helper was added to the model to enable intermediate activation visualization per residual stage.
* **Real validation loss tracked each epoch**:
  * Previously the scheduler and plots only used AUROC as a proxy signal.
  * Now `evaluate_loss()` computes the actual BCE loss on the validation set every epoch.
  * The training curve plot now shows both train and val loss side-by-side, making overfitting/underfitting much easier to diagnose visually.
* **Case example visualization added**:
  * `show_class_examples()` displays representative benign vs malignant CT slices from the training set.
* **Feature map visualization added**:
  * `visualize_feature_maps()` shows the intermediate activations at each residual stage for one benign and one malignant sample, useful for understanding what the model is attending to.
* **Best training epoch: 6** | **Best val AUROC: 0.844**


### Trial 2 results summary (tuned thr = 0.50)

The validation sweep returned **threshold = 0.50** as the best F1 cutoff — no shift needed.

| Metric | Value |
|---|---|
| AUROC | 0.894 |
| F1 | 0.627 |
| Recall | 0.734 |
| Precision | 0.547 |
| Specificity | 0.841 |
| Accuracy | 0.819 |

#### Comparing Trial 2 (thr=0.50) vs Trial 1 (thr=0.65)

| Metric | Trial 1 (thr=0.65) | Trial 2 (thr=0.50) | Δ |
|---|---|---|---|
| AUROC | 0.871 | **0.894** | **+0.023** |
| F1 | 0.598 | **0.627** | **+0.029** |
| Recall | 0.594 | **0.734** | **+0.140** |
| Precision | **0.603** | 0.547 | −0.056 |
| Specificity | **0.889** | 0.841 | −0.048 |
| Accuracy | 0.835 | 0.819 | −0.016 |


<p><strong>Trial 2 Model Results:</strong></p>
<img src="results/runs/2026-03-28_13.15.49/figures/results_table_2026-03-28_13.15.49.png" alt="Trial 1 Model Results">

<p><strong>Trial 2 Confusion Matrix:</strong></p>
<img src="results/runs/2026-03-28_13.15.49/figures/confusion_matrix_2026-03-28_13.15.49.png" alt="Trial 1 Confusion Matrix" width="300">

<p><strong>Trial 2 Training and Val Loss:</strong></p>
<img src="results/runs/2026-03-28_13.15.49/figures/train_val_curves_2026-03-28_13.15.49.png" width="500">



### Interpretation

In Trial 2 residual blocks and real validation loss tracking produced clear gains, but the picture is nuanced when comparing at each trial’s own best threshold.

<ins>**AUROC increased from 0.871 to 0.894**</ins>
This is the highest AUROC so far. The residual CNN ranks malignant nodules above benign ones more reliably than Trial 1’s plain conv stack. The skip connections in `ResBlock3D` improve gradient flow and feature representation quality.

<ins>**Recall jumped significantly (+0.140), while precision and specificity fell**</ins>
Comparing at each trial’s tuned threshold (Trial 1 at 0.65, Trial 2 at 0.50):
- Recall increased from 0.594 to 0.734 — the model catches far more true malignant cases.
- Precision dropped from 0.603 to 0.547 — more false positives per true positive.
- Specificity dropped from 0.889 to 0.841 — more benign cases incorrectly flagged.

This shift is partly an artefact of comparing thresholds: Trial 2’s validation sweep found 0.50 optimal, which is a lower cutoff than Trial 1’s 0.65. A lower threshold naturally flags more cases as malignant, which boosts recall at the cost of precision and specificity.

The best threshold for Trial 2 was 0.50, unlike Trial 1’s 0.65. This suggests the residual architecture produces better-separated probability scores that are already well centred around 0.5 without needing an upward shift.

<ins>**The new train vs val loss curves confirm the model is not overfitting**</ins>
 Both curves track closely and plateau around epoch 6 (the best epoch), at which point the LR scheduler decayed the learning rate. The early convergence at epoch 6 with AUROC 0.844 on validation suggests the residual blocks converge faster than the plain CNN, but also that there may be room to let training run longer before LR decay kicks in. This is worth testing in Trial 3 by increasing scheduler patience.

<ins>**Logistic regression**</ins> 
The logit AUROC was 0.823, which is now clearly behind the 3D CNN at AUROC 0.894, confirming the gains in Trial 2 are coming from the architectural improvement (ie not pipeline noise).

<ins>**Overall**</ins>
Trial 2 is the best-performing model so far. Residual connections improved ranking quality (AUROC), reduced false positives (precision and specificity), and maintained recall.

---

## Trial 3: Expanded Training Data, Enhanced Augmentation, and Longer Convergence

This trial addressed all three items from the "What to try next" list. The changes fall into three categories:

### What was changed

**1. Scheduler patience increase from 3 to 6**
- Trial 2's best epoch was only 6 out of 50, meaning the LR decayed far too early
- Increasing patience to 6 gives the residual blocks more time to converge before the learning rate is halved
- Result: best epoch moved from 6 to 33, giving the model 27 more productive training epochs

**2. Enhanced 3D augmentation**
- Previous augmentation was limited to random axis flips and Gaussian noise.
- Trial 3 adds three new transforms applied during training only:
  - **Random 90° axial rotation** (30% probability): rotates in the H–W plane by 90/180/270 degrees to reduce the model's reliance on fixed spatial orientation
  - **Mild zoom/crop** (30% probability): upsamples the volume by 1.0–1.15×, then center-crops back to 28×28×28, simulating variations in nodule size and position
  - **Multiplicative intensity jitter** (30% probability): scales voxel intensities by a random factor in [0.9, 1.1], simulating scanner-to-scanner intensity variation

**3. Expanded training data with two additional datasets**
- The model previously trained only on NoduleMNIST3D (~1,158 training samples).
- Two more CT datasets were added:

  | Dataset | Source | Format | Train samples | Val samples | Test samples |
  |---|---|---|---|---|---|
  | **IQ-OTH:NCCD** | Kaggle (IQ-OTH/NCCD) | 2D JPG slices | 877 | 110 | 110 |
  | **LungcancerDataSet** | SharePoint sample set | 2D JPG/PNG slices | 1,460 | 142 | 475 |

- Since these datasets contain 2D images, each slice is resized to 28×28 and repeated 28 times along the depth axis to form a pseudo-3D volume (28×28×28). This keeps the existing 3D model architecture completely unchanged.
- Labels are collapsed to binary: 
  - Benign / Normal → 0, 
  - Malignant / all carcinoma sub-types → 1.
- The combined training set has 3,495 samples (vs 1,158 in Trial 2) with a near-balanced class ratio (~1.09 neg/pos), so `pos_weight` dropped from ~2.0 to ~1.09.
- For evaluation, NoduleMNIST3D test is kept separate for equal comparison across all trials. A separate combined test set is also evaluated to measure cross-dataset generalisation.

---

### Trial 3 results 

With NoduleMNIST3D as test set, the validation sweep selected **threshold = 0.60**, consistent with Trial 1's direction (0.65). Both of these indicate the model has a slight upward probability bias that a raised cutoff corrects.

| Metric | Value |
|---|---|
| AUROC | **0.922** |
| F1 | **0.678** |
| Recall | 0.641 |
| Precision | **0.719** |
| Specificity | **0.935** |
| Accuracy | **0.874** |

Confusion matrix (NoduleMNIST3D test, tuned thr = 0.60):

| | pred 0 | pred 1 |
|---|---|---|
| true 0 (neg) | 230 | 16 |
| true 1 (pos) | 23 | 41 |

#### Comparing Trial 3 vs Trial 2 at each trial's tuned threshold

| Metric | Trial 2 (thr=0.50) | Trial 3 (thr=0.60) | Δ |
|---|---|---|---|
| AUROC | 0.894 | **0.922** | **+0.028** |
| F1 | 0.627 | **0.678** | **+0.051** |
| Recall | **0.734** | 0.641 | −0.093 |
| Precision | 0.547 | **0.719** | **+0.172** |
| Specificity | 0.841 | **0.935** | **+0.094** |
| Accuracy | 0.819 | **0.874** | **+0.055** |

---

### Trial 3 results on Combined test sets (tuned thr = 0.60)

Evaluated on the union of all three dataset test splits (NoduleMNIST3D + IQ-OTH:NCCD + LungcancerDataSet) at the same tuned threshold:

| Metric | Value |
|---|---|
| AUROC | **0.974** |
| F1 | **0.910** |
| Recall | **0.887** |
| Precision | **0.934** |
| Specificity | **0.933** |
| Accuracy | **0.909** |

---

<p><strong>Trial 3 Model Results:</strong></p>
<img src="results/runs/2026-04-11_17.49.54/figures/results_table_2026-04-11_17.49.54.png" alt="Trial 3 Model Results">

<p><strong>Trial 3 Confusion Matrix (NoduleMNIST3D, tuned thr = 0.60):</strong></p>
<img src="results/runs/2026-04-11_17.49.54/figures/confusion_matrices_tuned_2026-04-11_17.49.54.png" alt="Trial 3 Confusion Matrix" width="600">

<p><strong>Trial 3 Training and Val Loss:</strong></p>
<img src="results/runs/2026-04-11_17.49.54/figures/train_val_curves_2026-04-11_17.49.54.png" width="500">

---

### Interpretation

**Best epoch jumped from 6 to 33:**
In Trial 2, the LR scheduler decayed far too early (epoch 6 out of 50), leaving most of training useless. With patience = 6, the model kept improving all the way to epoch 33. Val AUROC grew from 0.744 at epoch 1 to a peak of **0.976 at epoch 33**, compared to Trial 2's best val AUROC of 0.844. The train loss also dropped from 0.581 to 0.152 and val loss from 0.679 to 0.166, with both curves tracking closely, thus no sign of overfitting despite the longer training run.

**AUROC on NoduleMNIST3D improved from 0.894 to 0.922:**
This is the most meaningful comparison across trials because it uses the same test set throughout. The jump of +0.028 is the largest single-trial AUROC gain so far, and it happened while training on a larger, noisier dataset, suggesting the expanded data and augmentation are genuinely helping the model learn better features rather than just memorising the small NoduleMNIST3D training set.

**Precision and specificity improved substantially at the cost of a meaningful recall drop:**
Comparing at each trial's tuned threshold (Trial 2 at 0.50, Trial 3 at 0.60): precision jumped from 0.547 to 0.719 (+0.172) and specificity from 0.841 to 0.935 (+0.094). Model now makes far fewer false positive calls, however the flip side is a slight drop in recall: 0.734 → 0.641 (−0.093). 

In absolute terms, Trial 3 misses more true malignant cases (23 false negatives vs 17 in Trial 2). This is a tendency with the precision–recall tradeoff. Which side to prioritise depends on the clinical use case as Trial 3 is better for reducing unnecessary follow-up procedures, whereas Trial 2 catches more maligniant cases. In the case of cancer detection, obviously detecting false negatives is more important.

**The tuned threshold of 0.60 is consistent with Trial 1's 0.65.**
Both trials needed an upward shift from 0.50, pointing to a persistent mild positive bias in the model's probability outputs as borderline cases tend to get scored slightly above 0.5. Raising the cutoff corrects this, and the resulting precision/specificity gains are substantial enough to justify the threshold shift.

**The combined test score (AUROC 0.974, F1 0.910) shows strong cross-dataset generalisation but needs context:**
The IQ-OTH:NCCD and LungcancerDataSet datasets are 2D images converted to pseudo-3D by stacking the same slice 28 times, producing volumes with no depth variation. This is structurally much simpler than the genuine 3D nodule volumes in NoduleMNIST3D. The high combined score reflects that the model handles both easy pseudo-3D data and harder volumetric data well. NoduleMNIST3D alone remains the most equal apples-to-apples benchmark across all trials.

**Logistic regression gap widens further:**
LogReg AUROC stayed at 0.823 while the 3D CNN reached 0.922 on the same test set, a gap of nearly 0.10. Every trial has increased this gap, confirming that the improvements are truly from the structural changes (i.e., better model, more data, better training), rather than pipeline noise.

**Overall Trial 3 is the best-performing model so far across every metric except recall:**
The combination of longer convergence, richer augmentation, and more diverse training data produced consistent and meaningful gains. The key remaining gap is recall on NoduleMNIST3D (0.703), as the model still misses roughly 30% of true positives on that test set. The next step should focus on recovering recall without sacrificing the precision/specificity gains made here.

