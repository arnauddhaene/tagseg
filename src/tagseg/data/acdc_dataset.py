import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import nibabel as nib
import numpy as np
import torch
from kedro.io import AbstractDataSet
from torch.utils.data import TensorDataset
from torchvision import transforms
from tqdm import tqdm

from .cine2gridtag import cine2gridtag


class AcdcDataSet(AbstractDataSet):
    def __init__(self, filepath: str, tagged: bool = True, only_myo: bool = False):
        self._filepath = Path(filepath)
        self._tagged = tagged
        self._only_myo = only_myo

    def _load(self) -> TensorDataset:
        # Get all patient folders from main raw downloaded ACDC directory
        patient_paths = [ppath for ppath in self._filepath.iterdir() if ppath.is_dir()]

        # Initialize empty image and label tensors
        images: torch.Tensor = torch.Tensor()
        labels: torch.Tensor = torch.Tensor()

        skip_label: int = 0
        skip_nan: int = 0

        accepted_classes: set = set([0.0, 1.0, 2.0, 3.0])

        # Iterate over all patients
        patients_pbar = tqdm(patient_paths, leave=False)
        for ppath in patients_pbar:
            patients_pbar.set_description(f"Processing {ppath.name}...")

            # Loading .nii.gz files in handled in the `Patient` class
            patient = Patient(ppath)
            assert len(patient.images) == len(patient.masks)

            # Loop through each patient's list of images (around 10 per patient)
            image_pbar = tqdm(
                zip(patient.images, patient.masks),
                leave=False,
                total=len(patient.images),
            )
            for image, label in image_pbar:

                # Preprocess labels and images
                image, label = image.astype(np.float64), label.astype(np.float64)
                label = self._preprocess_label()(
                    label
                )  # Label first as we use the resized version
                image = image / image.max()  # To [0, 1] range
                image = self._preprocess_image(0.456, 0.224, label)(image).unsqueeze(0)

                # Exclude NaNs from dataset
                if image.isnan().sum().item() > 0 or label.isnan().sum().item() > 0:
                    skip_nan += 1
                    continue

                # Throw out inconsistent masks
                _classes = label.unique().numpy()
                if len(_classes) > 4 or not set(_classes).issubset(accepted_classes):
                    skip_label += 1
                    continue

                # Modify label is only looking at LV myocardium
                if self._only_myo:
                    label = label == 2

                images = torch.cat((images, image), axis=0)
                labels = torch.cat((labels, label), axis=0)

        log = logging.getLogger(__name__)
        log.info(f"Skipped {skip_label} image(s) due to incoherent label")
        log.info(f"Skipped {skip_nan} image(s) due to presence of NaN")

        dataset = TensorDataset()
        dataset.tensors = (
            images,
            labels,
        )

        return dataset

    def _save(self, dataset: TensorDataset) -> None:
        pass

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _describe(self) -> Dict[str, Any]:
        return dict(tagged=self._tagged, only_myo=self._only_myo)

    def _preprocess_image(
        self, mu: float, sigma: float, label: torch.Tensor = None
    ) -> transforms.Compose:
        """Preprocess image

        Args:
            mu (float): average for normalization layer
            sigma (float): standard deviation for normalization layer
            label (Tensor, ndarray, optional): ROI labels for simulatetags contrast curve

        Returns:
            transforms.Compose: transformation callback function
        """
        if self._tagged:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((256, 256)),
                    SimulateTags(label=label),
                    transforms.Normalize(mean=mu, std=sigma),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mu, std=sigma),
                    transforms.Resize((256, 256)),
                ]
            )

    def _preprocess_label(self) -> transforms.Compose:
        """Preprocess mask

        Returns:
            transforms.Compose: transformation callback function
        """
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (256, 256), interpolation=transforms.InterpolationMode.NEAREST
                ),
            ]
        )


class Patient:
    """Class that loads cine MR images and annotations from an ACDC patient."""

    def __init__(self, filepath: str):

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
                self.images, self.masks = self.fetch_frames(f, f_gt)

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


class SimulateTags(torch.nn.Module):
    """Simulates tagging.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    """

    def __init__(
        self,
        spacing: float = 5,
        contrast: float = 0.4,
        label: torch.Tensor = None,
        myo_index: int = 2,
    ):
        """
        Args:
            spacing (float, optional): spacing between tag lines, in pixels. Defaults to 5.
            contrast (float, optional): exponent to apply to the image reducing contrast.
                Defaults to 0.4.
            label (Tensor, ndarray, optional): mask of regions of interest
            myo_index (int, optional): index in the mask of the LV myocardium
        """
        super().__init__()
        self.spacing = spacing
        self.contrast = contrast
        self.label = label
        self.myo_index = myo_index

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be tagged.

        Returns:
            PIL Image or Tensor: tagged image.
        """
        return cine2gridtag(
            img, self.label, self.myo_index, self.contrast, self.spacing
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(spacing={self.spacing}, contrast={self.contrast})"
        )


def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header