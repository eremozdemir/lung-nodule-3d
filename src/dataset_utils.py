"""
External dataset loaders for IQ-OTH:NCCD and LungcancerDataSet.

Both datasets contain 2D CT scan images (JPG/PNG).  We convert each 2D
slice to a pseudo-3D volume (28×28×28) by resizing the slice to 28×28
and repeating it 28 times along the depth axis.  This keeps the existing
3D model architecture and collate_fn unchanged.

Label mapping (binary nodule classification):
    0 — benign / normal (no malignancy)
    1 — malignant
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


# ── Label maps ────────────────────────────────────────────────────────────────

# IQ-OTH:NCCD: flat layout under "The IQ-OTHNCCD lung cancer dataset/"
IQOTH_LABEL_MAP = {
    "Bengin cases":    0,
    "Malignant cases": 1,
    "Normal cases":    0,
}

# LungcancerDataSet: split into train / valid / test sub-trees
LUNGCANCER_LABEL_MAP = {
    # generic folders
    "Bengin cases":                                        0,
    "BenginCases":                                         0,
    "normal":                                              0,
    "Malignant cases":                                     1,
    "MalignantCases":                                      1,
    # specific carcinoma sub-types (all malignant)
    "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa":   1,
    "squamous.cell.carcinoma":                             1,
    "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib":         1,
    "adenocarcinoma":                                      1,
    "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa":      1,
    "large.cell.carcinoma":                                1,
}


# ── Image helpers ─────────────────────────────────────────────────────────────

def _load_as_pseudo3d(path: str, size: int = 28) -> np.ndarray:
    """
    Load a 2-D image and turn it into a pseudo-3D volume.

    Pipeline:
        1. Open as grayscale (L).
        2. Resize to (size × size) with bilinear interpolation.
        3. Stack the 2-D slice `size` times along a new depth axis
           → (D=size, H=size, W=size).
        4. Prepend a channel dimension → (1, D, H, W).

    The returned array has dtype uint8 in [0, 255], matching
    the format returned by MedMNIST's NoduleMNIST3D.
    """
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)          # (H, W)
    vol = np.stack([arr] * size, axis=0)         # (D, H, W)
    return vol[np.newaxis, ...]                  # (1, D, H, W)


def _collect_samples(
    root_dir: str,
    label_map: dict,
) -> List[Tuple[str, int]]:
    """
    Walk the immediate sub-directories of *root_dir*, applying *label_map*
    to assign a binary label to every image file found.

    Unrecognised sub-directories are silently skipped.
    """
    samples: List[Tuple[str, int]] = []
    root = Path(root_dir)
    for subfolder, label in label_map.items():
        folder = root / subfolder
        if not folder.exists():
            continue
        for fp in sorted(folder.iterdir()):
            if fp.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((str(fp), label))
    return samples


# ── Base dataset ──────────────────────────────────────────────────────────────

class Pseudo3DDataset:
    """
    PyTorch-compatible dataset wrapping a list of (file_path, label) pairs.
    Each 2-D image is converted to a pseudo-3D volume on demand.

    Returns (volume, label) where:
        volume — uint8 numpy array of shape (1, D, H, W)
        label  — int64 numpy array of shape (1,)

    Compatible with the existing ``collate_fn`` in the notebook:
    collate_fn already handles both (D,H,W) and (1,D,H,W) inputs via the
    ``unsqueeze(1)`` guard.
    """

    def __init__(self, samples: List[Tuple[str, int]], vol_size: int = 28):
        self.samples = samples
        self.vol_size = vol_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        path, label = self.samples[idx]
        vol = _load_as_pseudo3d(path, self.vol_size)
        return vol, np.array([label], dtype=np.int64)


# ── IQ-OTH:NCCD ──────────────────────────────────────────────────────────────

class IQOTHNCCDDataset(Pseudo3DDataset):
    """
    IQ-OTH:NCCD lung cancer dataset.

    This dataset has no pre-defined splits, so we create stratified
    train / val / test splits here using a fixed random seed.

    Expected folder structure under *root*:
        <root>/
          The IQ-OTHNCCD lung cancer dataset/
            The IQ-OTHNCCD lung cancer dataset/
              Bengin cases/     ← label 0
              Malignant cases/  ← label 1
              Normal cases/     ← label 0

    The "Test cases/" sibling folder contains unlabelled PNG slices
    (patient IDs only, no class information) and is intentionally ignored.

    Parameters
    ----------
    root : str
        Path to the ``IQ-OTH:NCCD`` directory inside ``data/``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    val_frac : float
        Fraction of all labelled samples to reserve for validation.
    test_frac : float
        Fraction of all labelled samples to reserve for testing.
    seed : int
        Random seed for reproducible splits.
    vol_size : int
        Target spatial dimension for the pseudo-3D volume.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_frac: float = 0.10,
        test_frac: float = 0.10,
        seed: int = 42,
        vol_size: int = 28,
    ):
        data_root = os.path.join(
            root,
            "The IQ-OTHNCCD lung cancer dataset",
            "The IQ-OTHNCCD lung cancer dataset",
        )
        all_samples = _collect_samples(data_root, IQOTH_LABEL_MAP)

        paths  = [s[0] for s in all_samples]
        labels = [s[1] for s in all_samples]

        # First carve out the test set
        p_trainval, p_test, y_trainval, y_test = train_test_split(
            paths, labels,
            test_size=test_frac,
            stratify=labels,
            random_state=seed,
        )
        # Then split val out of the remaining data
        adjusted_val_frac = val_frac / (1.0 - test_frac)
        p_train, p_val, y_train, y_val = train_test_split(
            p_trainval, y_trainval,
            test_size=adjusted_val_frac,
            stratify=y_trainval,
            random_state=seed,
        )

        split_samples = {
            "train": list(zip(p_train, y_train)),
            "val":   list(zip(p_val,   y_val)),
            "test":  list(zip(p_test,  y_test)),
        }
        super().__init__(split_samples[split], vol_size)


# ── LungcancerDataSet ─────────────────────────────────────────────────────────

class LungCancerDataset(Pseudo3DDataset):
    """
    LungcancerDataSet — already split into train / valid / test sub-trees.

    Expected folder structure under *root*:
        <root>/
          Data/
            train/
              Bengin cases/
              Malignant cases/
              normal/
              squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa/
              adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/
              large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/
            valid/          ← same structure as train
            test/
              BenginCases/
              MalignantCases/
              normal/
              squamous.cell.carcinoma/
              adenocarcinoma/
              large.cell.carcinoma/

    Parameters
    ----------
    root : str
        Path to the ``LungcancerDataSet`` directory inside ``data/``.
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    vol_size : int
        Target spatial dimension for the pseudo-3D volume.
    """

    # Map our split names to the on-disk folder names
    _SPLIT_FOLDER = {"train": "train", "val": "valid", "test": "test"}

    def __init__(self, root: str, split: str = "train", vol_size: int = 28):
        folder = self._SPLIT_FOLDER[split]
        data_root = os.path.join(root, "Data", folder)
        samples = _collect_samples(data_root, LUNGCANCER_LABEL_MAP)
        super().__init__(samples, vol_size)
