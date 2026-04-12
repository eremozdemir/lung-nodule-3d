"""
External dataset loaders for IQ-OTH:NCCD, LungcancerDataSet, and LUNA16.

IQ-OTH:NCCD and LungcancerDataSet contain 2D CT scan images (JPG/PNG).
We convert each 2D slice to a pseudo-3D volume (28×28×28) by resizing
the slice to 28×28 and repeating it 28 times along the depth axis.

LUNA16 contains genuine 3D CT volumes in MetaImage (.mhd/.raw) format.
Nodule patches are extracted at construction time using the annotated
world coordinates, windowed to the standard lung HU range, and resized
to 28×28×28 — making them directly compatible with the rest of the pipeline.

Label mapping (binary nodule classification):
    0 — benign / normal / non-nodule candidate
    1 — malignant / confirmed nodule
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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

def _apply_lung_window(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """
    Percentile-based window/level normalization, inspired by CT Hounsfield Unit windowing.

    In proper DICOM pipelines the lung window is set to roughly HU [-1350, 150]
    (centre -600, width 1500).  Since IQ-OTH:NCCD and LungcancerDataSet are JPEG
    screenshots rather than raw DICOM files, we cannot recover true HU values.
    Instead we clip to the [p_low, p_high] percentile range of the image and
    rescale to 0-255.  This removes scanner-border artefacts (the bright/dark
    edges that dominate simple grayscale normalisation) and stretches the
    contrast of the lung tissue region, which is what the model should focus on.
    """
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if hi == lo:
        return arr  # uniform image — nothing to stretch
    arr = np.clip(arr, lo, hi)
    arr = ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return arr


def _load_as_pseudo3d(path: str, size: int = 28) -> np.ndarray:
    """
    Load a 2-D CT screenshot and convert it to a pseudo-3D volume.

    Pipeline:
        1. Open as grayscale (L).
        2. Resize to (size × size) with bilinear interpolation.
        3. Apply lung-window-inspired contrast normalisation to suppress
           scanner-border artefacts and emphasise lung tissue density.
        4. Stack the 2-D slice `size` times along a new depth axis
           → (D=size, H=size, W=size).
        5. Prepend a channel dimension → (1, D, H, W).

    The returned array has dtype uint8 in [0, 255], matching
    the format returned by MedMNIST's NoduleMNIST3D.
    """
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)          # (H, W)
    arr = _apply_lung_window(arr)                # percentile window/level
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


# ── LUNA16 ────────────────────────────────────────────────────────────────────

def _extract_luna16_patch(
    hu_arr: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    cx: float,
    cy: float,
    cz: float,
    crop_mm: float = 40.0,
    out_size: int = 28,
) -> np.ndarray:
    """
    Extract a (1, out_size, out_size, out_size) uint8 patch centred at world
    coordinates (cx, cy, cz) from a HU-valued CT volume.

    Coordinate conventions follow SimpleITK:
        hu_arr  — (Z, Y, X) float32 array in Hounsfield Units
        origin  — (X, Y, Z) scan origin in mm
        spacing — (X, Y, Z) mm per voxel

    Pipeline
    --------
    1. Convert world (X,Y,Z) mm → voxel indices (xi, yi, zi).
    2. Compute axis-wise half-radii so that the physical crop is always
       crop_mm × crop_mm × crop_mm mm regardless of scan anisotropy.
    3. Edge-pad the scan to handle candidates near the boundary.
    4. Apply lung HU window [-1000, 400] → rescale to uint8 [0, 255].
    5. Resize to (out_size, out_size, out_size) with trilinear interpolation.
    """
    import scipy.ndimage as nd

    # World → voxel (X,Y,Z → xi,yi,zi; array indexing is [zi, yi, xi])
    xi = int(round((cx - origin[0]) / spacing[0]))
    yi = int(round((cy - origin[1]) / spacing[1]))
    zi = int(round((cz - origin[2]) / spacing[2]))

    # Anisotropy-aware half-radii (spacing order is X,Y,Z)
    hrx = max(1, int(round(crop_mm / (2.0 * spacing[0]))))
    hry = max(1, int(round(crop_mm / (2.0 * spacing[1]))))
    hrz = max(1, int(round(crop_mm / (2.0 * spacing[2]))))

    # Pad so extraction never goes out of bounds
    arr_pad = np.pad(hu_arr, ((hrz, hrz), (hry, hry), (hrx, hrx)), mode="edge")

    # Shifted indices
    zi_p, yi_p, xi_p = zi + hrz, yi + hry, xi + hrx
    crop = arr_pad[zi_p - hrz: zi_p + hrz,
                   yi_p - hry: yi_p + hry,
                   xi_p - hrx: xi_p + hrx].copy()

    # Lung HU window → uint8
    crop = np.clip(crop, -1000.0, 400.0)
    crop = ((crop + 1000.0) / 1400.0 * 255.0).astype(np.uint8)

    # Resize to target cube
    zoom_factors = [out_size / s for s in crop.shape]
    crop = nd.zoom(crop, zoom_factors, order=1).astype(np.uint8)

    return crop[np.newaxis, ...]  # (1, D, H, W)


class LUNA16Dataset:
    """
    LUNA16 lung nodule dataset — genuine 3D CT patch volumes.

    Patches are extracted eagerly at construction time (one full scan load
    per scan, all patches in memory) so that training DataLoader iteration
    is fast.  Total memory usage is small: ~1950 patches × 28³ ≈ 43 MB.

    Positives  : every row in ``annotations.csv`` for the available subsets
                 (confirmed nodule ≥ 3 mm diameter, label = 1).
    Negatives  : ``neg_per_scan`` candidates sampled from ``candidates.csv``
                 class = 0 per available scan (label = 0).

    Parameters
    ----------
    root : str
        Path to the ``Luna16/`` directory (contains subset0–subset4,
        ``annotations.csv``, ``candidates.csv``).
    split : str
        One of ``"train"``, ``"val"``, ``"test"``.
    neg_per_scan : int
        Number of class-0 candidates sampled per scan.  Default 3 yields
        a ~2.2:1 neg/pos ratio across the five available subsets.
    crop_mm : float
        Physical side length (mm) of the cubic crop around each candidate.
        40 mm contains even the largest annotated nodule (32 mm diameter)
        with adequate context margin.
    out_size : int
        Output cube side length in voxels (28 to match NoduleMNIST3D).
    val_frac : float
        Fraction of all candidates reserved for validation.
    test_frac : float
        Fraction of all candidates reserved for testing.
    seed : int
        Random seed for reproducible splits and negative sampling.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        neg_per_scan: int = 3,
        crop_mm: float = 40.0,
        out_size: int = 28,
        val_frac: float = 0.10,
        test_frac: float = 0.10,
        seed: int = 42,
    ):
        import SimpleITK as sitk

        root = Path(root)
        ann_df = pd.read_csv(root / "annotations.csv")
        cand_df = pd.read_csv(root / "candidates.csv")

        # Build uid → .mhd path map from all subset*/subset*/ directories
        uid_to_mhd: Dict[str, str] = {}
        for mhd_path in sorted(root.glob("subset*/subset*/*.mhd")):
            uid_to_mhd[mhd_path.stem] = str(mhd_path)
        present = set(uid_to_mhd)

        # Positive samples from annotations.csv
        pos_df = ann_df[ann_df.seriesuid.isin(present)][
            ["seriesuid", "coordX", "coordY", "coordZ"]
        ].copy()
        pos_df["label"] = 1

        # Negative samples: sample neg_per_scan class-0 candidates per scan
        rng = np.random.default_rng(seed)
        neg_rows = []
        neg_pool = cand_df[(cand_df["class"] == 0) & (cand_df.seriesuid.isin(present))]
        for uid, grp in neg_pool.groupby("seriesuid"):
            sampled = grp.sample(
                n=min(neg_per_scan, len(grp)),
                random_state=int(rng.integers(1 << 31)),
            )
            neg_rows.append(sampled[["seriesuid", "coordX", "coordY", "coordZ"]])
        neg_df = pd.concat(neg_rows, ignore_index=True)
        neg_df["label"] = 0

        all_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(
            frac=1, random_state=seed
        ).reset_index(drop=True)

        # Stratified train / val / test split
        idx = np.arange(len(all_df))
        y = all_df["label"].values
        idx_tv, idx_test = train_test_split(
            idx, test_size=test_frac, stratify=y, random_state=seed
        )
        adj_val = val_frac / (1.0 - test_frac)
        idx_train, idx_val = train_test_split(
            idx_tv, test_size=adj_val, stratify=y[idx_tv], random_state=seed
        )
        subset_df = all_df.iloc[
            {"train": idx_train, "val": idx_val, "test": idx_test}[split]
        ].reset_index(drop=True)

        # Eagerly extract patches — load each scan once, extract all its patches
        print(
            f"LUNA16 [{split}]: extracting {len(subset_df)} patches "
            f"from {subset_df.seriesuid.nunique()} scans …"
        )
        patches: List[np.ndarray] = []
        labels_list: List[int] = []

        for uid, grp in tqdm(
            subset_df.groupby("seriesuid"),
            desc=f"LUNA16 {split}",
            leave=False,
        ):
            img = sitk.ReadImage(uid_to_mhd[uid])
            hu_arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (Z, Y, X)
            origin = np.array(img.GetOrigin())    # (X, Y, Z) mm
            spacing = np.array(img.GetSpacing())  # (X, Y, Z) mm/voxel

            for _, row in grp.iterrows():
                patch = _extract_luna16_patch(
                    hu_arr, origin, spacing,
                    row.coordX, row.coordY, row.coordZ,
                    crop_mm=crop_mm,
                    out_size=out_size,
                )
                patches.append(patch)
                labels_list.append(int(row.label))

        self.patches = patches
        self.labels = labels_list
        n_pos = sum(labels_list)
        print(f"  done — {len(labels_list)} patches  (pos={n_pos}  neg={len(labels_list)-n_pos})")

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.patches[idx], np.array([self.labels[idx]], dtype=np.int64)
