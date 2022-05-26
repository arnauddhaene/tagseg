import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from .dataset import TagSegDataSet
from .utils import load_nii


class AcdcDataSet(TagSegDataSet):
    def _load_except(self, filepath_raw: str, only_myo: bool) -> TensorDataset:

        # Get all patient folders from main raw downloaded ACDC directory
        patient_paths = [
            ppath for ppath in Path(filepath_raw).iterdir() if ppath.is_dir()
        ]

        # Initialize empty image and label tensors
        images: torch.Tensor = torch.Tensor()
        labels: torch.Tensor = torch.Tensor()

        skip_label: int = 0
        skip_nan: int = 0
        skip_unlabeled: int = 0

        accepted_classes: set = set([0.0, 1.0, 2.0, 3.0])

        # Iterate over all patients
        patients_pbar = tqdm(patient_paths, leave=False)
        for ppath in patients_pbar:
            patients_pbar.set_description(f"Processing {ppath.name}...")

            # Loading .nii.gz files in handled in the `Patient` class
            patient = Patient(ppath)
            assert len(patient.images) == len(patient.masks)

            # Loop through each patient's slices (usually 2 per patient)
            for slice_images, slice_labels in zip(patient.images, patient.masks):

                # Loop through each slice's list of images (around 10 per patient)
                image_pbar = tqdm(
                    zip(slice_images, slice_labels),
                    leave=False,
                    total=len(slice_images),
                )
                for image, label in image_pbar:

                    # Preprocess labels and images
                    image, label = image.astype(np.float64), label.astype(np.float64)
                    image = image / image.max()  # To [0, 1] range
                    image = self._preprocess_image(0.456, 0.224)(image).unsqueeze(0)
                    label = self._preprocess_label()(label)

                    # Exclude NaNs from dataset
                    if image.isnan().sum().item() > 0 or label.isnan().sum().item() > 0:
                        skip_nan += 1
                        continue

                    # Throw out inconsistent masks
                    classes = label.unique().numpy()
                    if len(classes) > 4 or not set(classes).issubset(accepted_classes):
                        skip_label += 1
                        continue

                    # Modify label is only looking at LV myocardium
                    if only_myo:
                        label = label == 2

                    if torch.count_nonzero(label).item() == 0:
                        skip_unlabeled += 1
                        continue

                    images = torch.cat((images, image), axis=0)
                    labels = torch.cat((labels, label), axis=0)

        log = logging.getLogger(__name__)
        log.info(f"Skipped {skip_label} image(s) due to incoherent label")
        log.info(f"Skipped {skip_nan} image(s) due to presence of NaN")
        log.info(f"Skipped {skip_unlabeled} image(s) due to absence of label")

        dataset = TensorDataset()
        dataset.tensors = (
            images,
            labels,
        )

        return dataset


class Patient:
    """Class that loads cine MR images and annotations from an ACDC patient."""

    def __init__(self, filepath: str):

        self.images, self.masks = [], []

        # Fetch list of all potential files
        files = [f for f in Path(filepath).iterdir() if f.suffixes == [".nii", ".gz"]]

        for f in files:
            # Discard 4d
            # Discard ground truth as those are fetched by their image
            if "_4d" in str(f) or "_gt" in str(f):
                continue

            # Fetch path of mask following dataset nomenclature
            f_gt = self._gt_path(f)

            if f_gt in files:
                image, mask = self.fetch_frames(f, f_gt)
                self.images.append(image)
                self.masks.append(mask)

    @staticmethod
    def _gt_path(filepath: Path) -> Path:
        """Get an image's corresponding ground truth mask

        Args:
            filepath (Path): location of image.

        Returns:
            Path: location of mask.
        """
        return filepath.parent / (filepath.stem.split(".")[0] + "_gt.nii.gz")

    @staticmethod
    def fetch_frames(
        image_path: Path, mask_path: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from image and mask locations

        Args:
            image_path (Path): location of image.
            mask_path (Path): location of mask.

        Returns:
            Tuple[ndarray, ndarray]: image (1, H, W) and mask (1, H, W)
        """

        imt, _, _ = load_nii(image_path)
        gt, _, _ = load_nii(mask_path)

        return imt.swapaxes(0, 2), gt.swapaxes(0, 2)
