# Implementation Notes
This is where I'll log my notes about what I tried, what worked, what didn't work, and what I plan to try in subsequent implementation trials in hopes to keep improving model accuracy. 


## Table of Contents
- [Implementation Notes](#implementation-notes)
  - [Table of Contents](#table-of-contents)
  - [Trial 0: Initial Model Implementation](#trial-0-initial-model-implementation)
    - [Trial 0 results summary (thr = 0.50)](#trial-0-results-summary-thr--050)
    - [Interpretation](#interpretation)
  - [Trial 1: Simple Performance Upgrades](#trial-1-simple-performance-upgrades)
    - [Trial 1 results summary (default thr = 0.50)](#trial-1-results-summary-default-thr--050)
      - [Comparing Trial 1 results with Trial 0 results at default thr = 0.5](#comparing-trial-1-results-with-trial-0-results-at-default-thr--05)
    - [Trial 1 results summary (best thr = 0.65)](#trial-1-results-summary-best-thr--065)
      - [Comparing Trial 1 results between default thr = 0.5 and best thr = 0.65](#comparing-trial-1-results-between-default-thr--05-and-best-thr--065)
    - [Interpretation](#interpretation-1)
  - [Trial 2: Residual Blocks and Validation Loss Tracking](#trial-2-residual-blocks-and-validation-loss-tracking)
    - [Trial 2 results summary (best thr = 0.50)](#trial-2-results-summary-best-thr--050)
      - [Comparing Trial 2 results with best thr = 0.5, with Trial 0 with best thr = 0.65,](#comparing-trial-2-results-with-best-thr--05-with-trial-0-with-best-thr--065)
    - [Interpretation](#interpretation-2)
  - [What to try next for Trial 3](#what-to-try-next-for-trial-3)


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
- The model was conservative at the default threshold:
  -  Had decent specificity 
  -  But it missed a lot of malignant cases.
- Approx error pattern at thr 0.50:
  - False negatives were high (missed positives), which is why recall was low.
  - False positives were relatively low, which is why specificity and accuracy looked decent.

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

### Trial 1 results summary (default thr = 0.50)
#### Comparing Trial 1 results with Trial 0 results at default thr = 0.5
  1) With the default **thr = 0.50**, test metrics improved slightly in some areas but got worst in others between Trial 0 and 1:
     - **AUROC:** Increased from 0.849 to 0.871, 
     - **F1:** Increase from 0.564 to 0.599,
     - **Recall:** Increased from 0.516 to 0.734
     - **Precision:** Decreased from 0.623 to 0.505
     - **Specificity:** Decreased from 0.919 to 0.813
     - **Accuracy:** Increased from 0.797 to 0.835

### Trial 1 results summary (best thr = 0.65)
#### Comparing Trial 1 results between default thr = 0.5 and best thr = 0.65
  2) With the **tuned best thr = 0.65**:
     - **AUROC:** Increase from 0.849 to 0.871, (Same as default thr = 0.50)
     - **F1:** Increase from 0.564 to 0.598, (Same as default thr = 0.50)
     - **Recall:** Increased from 0.516 to 0.594
       -  For trial 1, the model with the best thr = 0.5 had a better Recall by 0.14, compared to the best thr = 0.65
     - **Precision:** Decreased from 0.623 to 0.603
       - For trial 1, the model with the best thr = 0.65 had a better Precision by 0.098, compared to the default thr = 0.5
     - **Specificity:** Decreased from 0.919 to 0.889
       - For trial 1, the model with the best thr = 0.65 had a better Specificity by 0.085, compared to the default thr = 0.5
     - **Accuracy:** Remained constant at 0.835
       - For trial 1, the model with the best thr = 0.65 had a better Accuracy by 0.038, compared to the default thr = 0.5


<p><strong>Trial 1 Model Results:</strong></p>
<img src="results/runs/2026-03-06_11.55.56/figures/results_table_2026-03-06_11.55.56.png" alt="Trial 1 Model Results">

<p><strong>Trial 1 Confusion Matrix:</strong></p>
<img src="results/runs/2026-03-06_11.55.56/figures/confusion_matrix_2026-03-06_11.55.56.png" alt="Trial 1 Confusion Matrix" width="300">

<p><strong>Trial 1 Training Loss:</strong></p>
<img src="results/runs/2026-03-06_11.55.56/figures/train_loss_2026-03-06_11.55.56.png" width="500">

### Interpretation
Trial 1 did improve the model, but the improvement showed up more in how well it ranks positives above negatives than accuracy number.

AUROC improved from 0.849 to 0.871, which means the CNN is separating benign vs malignant volumes more consistently overall, thus better ordering of scores.

At the default threshold (0.50), Trial 1 shifted the model toward catching more malignant cases:

Recall jumped from 0.516 to 0.734 (31 → 17 false negatives)

But that came with more false alarms: Specificity dropped from 0.919 to 0.813 (20 → 46 false positives)

So accuracy fell from 0.835 to 0.797, because the test set has more benign cases and false positives are “expensive” in accuracy.

With the tuned threshold (0.65), the tradeoff moves back toward the Trial 1 balance:

Accuracy returns to 0.835 and specificity improves to 0.898, but recall drops to 0.594 (still better than Trial 1’s 0.516).

AUROC stays 0.871, because AUROC doesn’t depend on the threshold, only on ranking.

Logistic regression stayed basically the same (AUROC 0.823, F1 0.554, accuracy 0.813), so the gains in Trial 1 are coming from the CNN changes, not noise in the pipeline.


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


### Trial 2 results summary (best thr = 0.50)
#### Comparing Trial 2 results with best thr = 0.5, with Trial 0 with best thr = 0.65,
     - **AUROC:** Increased from 0.871 to 0.894, 
     - **F1:** Increase from 0.599 to 0.627,
     - **Recall:** Stay consistent at 0.734,
     - **Precision:** Increase from 0.505 to 0.547,
     - **Specificity:** Increase from 0.813 to 0.841,
     - **Accuracy:** Decreased slightly from 0.835 to 0.819

Note: Trial 2 had a **best thr = 0.50 **


<p><strong>Trial 2 Model Results:</strong></p>
<img src="results/runs/2026-03-28_13.15.49/figures/results_table_2026-03-28_13.15.49.png" alt="Trial 1 Model Results">

<p><strong>Trial 2 Confusion Matrix:</strong></p>
<img src="results/runs/2026-03-28_13.15.49/figures/confusion_matrix_2026-03-28_13.15.49.png" alt="Trial 1 Confusion Matrix" width="300">

<p><strong>Trial 2 Training and Val Loss:</strong></p>
<img src="results/runs/2026-03-28_13.15.49/figures/train_val_curves_2026-03-28_13.15.49.png" width="500">



### Interpretation

In trial 2 I implemented residual blocks and real validation loss tracking, and the results show a clear improvement over Trial 1 across almost every metric at the default threshold (0.50).

<ins>**AUROC increase from 0.871 to 0.894**</ins> 
This is the highest so far across all trials. This means the residual CNN is ranking malignant nodules above benign ones more reliably than the plain conv stack from Trial 1. The skip connections in `ResBlock3D` are helping the model learn better feature representations without losing gradient signal in the deeper layers.

<ins> **At the default threshold (0.50) compared to Trial 1 at the same threshold**</ins> 
- Recall held consistent at **0.734**, meaning that the model is still catching the same proportion of malignant cases.
- Precision increased from **0.505 to 0.547**. This means fewer false positives per true positive, and that the model is more confident when it predicts malignant.
- Specificity increased from **0.813 to 0.841**, meaning that fewer benign cases are being incorrectly flagged.
- Accuracy decreased slightly from **0.835 to 0.819**, which is expected:
  - With more benign cases in the test set, any increase in false positives weighs more heavily on accuracy. However, this is a minor tradeoff given the gains in AUROC and precision.

The best threshold for Trial 2 was 0.50, unlike Trial 1 where 0.65 performed better. This suggests the model’s probability outputs are already well calibrated at the midpoint because the residual architecture produces more decisive, better-separated scores that do not require a shifted cutoff to perform well.

<ins>**The new train vs val loss curves confirm the model is not overfitting**</ins>
 Both curves track closely and plateau around epoch 6 (the best epoch), at which point the LR scheduler decayed the learning rate. The early convergence at epoch 6 with AUROC 0.844 on validation suggests the residual blocks converge faster than the plain CNN, but also that there may be room to let training run longer before LR decay kicks in. This is worth testing in Trial 3 by increasing scheduler patience.

<ins>**Logistic regression**</ins> 
The logit AUROC was 0.823, which is now clearly behind the 3D CNN at AUROC 0.894, confirming the gains in Trial 2 are coming from the architectural improvement (ie not pipeline noise).

<ins>**Overall**</ins>
Trial 2 is the best-performing model so far. Residual connections improved ranking quality (AUROC), reduced false positives (precision and specificity), and maintained recall.

---

## What to try next for Trial 3

A couple options that tend to move the needle without blowing up scope:

* **Adjust scheduler patience**
  * best epoch was only 8 out of 50; increasing patience from 3 to 6 may allow the residual blocks more time to converge before LR decays.
* **Better augmentation for 3D volumes:**
  * Small rotations, mild zoom/crop, random intensity jitter
  * Re-run threshold tuning and see if recall can climb without sacrificing specificity.
* **Start training on larger datasets:**
  * Currently training on the small, curated **NoduleMNIST3D / MedMNIST** subset, which is great for fast iteration but can cap how much the model generalizes to real CT variability (ie scanner differences, slice thickness, noise, pathology diversity).
  * The next step would be to scale up with more realistic CT volumes and labels, then fine tune back on NoduleMNIST3D for an apples to apples comparison while keeping the same metrics, threshold sweep, and logging pipeline.
  * But need to look into the following and similar datasets for more in depth training:

1. Lung Cancer Sample images dataset:

* [https://qnm8.sharepoint.com/Lung%20Cancer%20Detection%20%20Sample%20Dataset/Forms/AllItems.aspx?viewid=dc7c7db3%2Dac9c%2D4ef9%2Dab96%2D46c65f13c50a&p=true](https://qnm8.sharepoint.com/Lung%20Cancer%20Detection%20%20Sample%20Dataset/Forms/AllItems.aspx?viewid=dc7c7db3%2Dac9c%2D4ef9%2Dab96%2D46c65f13c50a&p=true)

2. CT scans of patients diagnosed with Lung Cancer:

* [https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset)

