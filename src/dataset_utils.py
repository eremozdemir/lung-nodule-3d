"""
External dataset loaders for ChestXrayCancerDataset and LUNA16.

ChestXrayCancerDataset loads 2D chest X-ray images and returns
(1, 224, 224) float32 tensors with torchvision normalisation.
A helper function provides ready-made train/val/test splits with
the appropriate augmentation transforms applied per split.

LUNA16 contains genuine 3D CT volumes in MetaImage (.mhd/.raw) format.
Nodule patches are extracted at construction time using the annotated
world coordinates, windowed to the standard lung HU range, and resized
to 28×28×28 — making them directly compatible with the Deep3DCNN pipeline.

Label mapping (binary classification):
    0 — benign / normal / non-nodule candidate
    1 — malignant / confirmed nodule / cancer
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

# ChestXrayCancerDataset: Cancer/NORMAL folder names
CXR_LABEL_MAP = {
    "Cancer": 1,
    "NORMAL": 0,
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


# ── ChestXrayCancerDataset ────────────────────────────────────────────────────

class ChestXrayCancerDataset:
    """
    Chest X-ray lung cancer detection dataset for the CXRClassifier.

    Returns (tensor, label) where tensor is a (1, 224, 224) float32 image
    produced by the torchvision transform passed at construction time.
    The caller is responsible for supplying the appropriate transform
    (training vs evaluation); use ``make_cxr_splits`` for convenience.

    Expected folder structure under the root split directory:
        <split>/
          Cancer/   ← label 1
          NORMAL/   ← label 0

    Parameters
    ----------
    samples : list of (path, label) pairs
        Use ``ChestXrayCancerDataset.collect_samples(root, split)`` to build.
    transform : torchvision transform or None
        Applied to each PIL image before returning.  When None, the raw
        PIL Image is returned (useful for visualisation).
    """

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")  # grayscale
        if self.transform is not None:
            img = self.transform(img)
        return img, np.array([label], dtype=np.int64)

    @staticmethod
    def collect_samples(root: str, split: str) -> List[Tuple[str, int]]:
        """Return (path, label) list for the given split sub-folder."""
        return _collect_samples(os.path.join(root, split), CXR_LABEL_MAP)


def make_cxr_splits(
    root: str,
    val_frac: float = 0.15,
    seed: int = 42,
    img_size: int = 224,
) -> Tuple["ChestXrayCancerDataset", "ChestXrayCancerDataset", "ChestXrayCancerDataset"]:
    """
    Build train / val / test datasets for the CXR pipeline.

    The provided ``train/`` folder is split into a training portion and an
    internal validation portion (stratified, ``val_frac`` of the total).
    The original ``val/`` folder (16 samples) is discarded as too small to
    produce reliable metrics.  The ``test/`` folder (624 samples) is the
    held-out evaluation set.

    Augmentation (training split only):
        RandomHorizontalFlip, RandomRotation(±10°), RandomAffine(shear=5°),
        ColorJitter(brightness=0.15, contrast=0.15).
    Both splits are normalised to mean=0.5, std=0.5.

    Parameters
    ----------
    root : str
        Path to the ``chest_xray_lung`` directory inside ``data/``.
    val_frac : float
        Fraction of training images reserved for internal validation.
    seed : int
        Random seed for reproducibility.
    img_size : int
        Spatial resolution for resizing (default 224, matches ResNet-18).

    Returns
    -------
    (train_ds, val_ds, test_ds)
    """
    try:
        import torchvision.transforms as T
    except ImportError as exc:
        raise ImportError("torchvision is required for make_cxr_splits()") from exc

    _norm = T.Normalize(mean=[0.5], std=[0.5])

    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.RandomAffine(degrees=0, shear=5),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        _norm,
    ])
    eval_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        _norm,
    ])

    # Split train/ folder into train + internal val
    all_samples = ChestXrayCancerDataset.collect_samples(root, "train")
    paths  = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]
    p_tr, p_va, y_tr, y_va = train_test_split(
        paths, labels, test_size=val_frac, stratify=labels, random_state=seed
    )

    test_samples = ChestXrayCancerDataset.collect_samples(root, "test")

    return (
        ChestXrayCancerDataset(list(zip(p_tr, y_tr)), train_transform),
        ChestXrayCancerDataset(list(zip(p_va, y_va)), eval_transform),
        ChestXrayCancerDataset(test_samples,           eval_transform),
    )


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

    # ── Intensity normalization ──────────────────────────────────────────────
    # Trial 6: use percentile-based windowing identical to the approach used
    # for IQ-OTH:NCCD and NoduleMNIST3D, rather than the absolute HU window.
    #
    # Problem with the old approach (HU clip [-1000,400] → uint8):
    #   The raw distribution of a 40mm lung patch is dominated by air (-1000 HU)
    #   which maps to pixel 0, so the nodule occupies only the top ~30% of the
    #   [0,255] range.  NoduleMNIST3D patches use MedMNIST's own per-patch
    #   percentile rescaling, so they span the full [0,255] range.  This
    #   intensity mismatch is one of the main causes of the domain gap observed
    #   in Trials 4 and 5.
    #
    # Fix: clip to the [1st, 99th] percentile of this crop, then rescale to
    # [0, 255].  This is equivalent to what _apply_lung_window() does for the
    # pseudo-3D datasets, and empirically what MedMNIST does for NoduleMNIST3D.
    lo = np.percentile(crop, 1)
    hi = np.percentile(crop, 99)
    if hi > lo:
        crop = np.clip(crop, lo, hi)
        crop = ((crop - lo) / (hi - lo) * 255.0).astype(np.uint8)
    else:
        crop = np.zeros_like(crop, dtype=np.uint8)

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
