import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_binary_metrics(y_true: np.ndarray, logits: np.ndarray, threshold: float = 0.5) -> dict:
    """
    y_true: shape (N,) values in {0,1}
    logits: shape (N,)
    """
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(int)

    # AUROC can fail if only one class exists in y_true
    auc = None
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, probs))

    f1 = float(f1_score(y_true, preds))
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()

    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    acc = float((tp + tn) / (tp + tn + fp + fn))

    return {
        "auroc": auc,
        "f1": f1,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "confusion_matrix": cm.tolist(),
    }