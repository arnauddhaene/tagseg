import logging

from pathlib import Path
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import TensorDataset

from .dataset import TagSegDataSet
from .utils import load_nii, camel_to_snake


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


# USELESS BECAUSE THERE ARE NO LABELS

# class AcdcEvaluator(LoadableDataSet):

#     def _load_except(self, filepath_raw: str) -> pd.DataFrame:
#         # Get all patient folders from main raw downloaded ACDC directory
#         patient_paths = [
#             ppath for ppath in Path(filepath_raw).iterdir() if ppath.is_dir()
#         ]

#         # Initialize storage that will be converted to pd.DataFrame
#         storage: List[Dict[str, Any]] = []

#         # Iterate over all patients
#         patients_pbar = tqdm(patient_paths, leave=False)
#         for ppath in patients_pbar:
#             patients_pbar.set_description(f"Processing {ppath.name}...")
            
#             save_path = self._filepath.parent / self._filepath.stem
#             storage.extend(Patient(ppath, has_mask=False, save_path=save_path).storage)

#         return pd.DataFrame(storage)


class Patient:
    """Class that loads cine MR images and annotations from an ACDC patient."""

    def __init__(self, filepath: str, has_mask: bool = True, save_path: str = None):

        self.images, self.masks = [], []
        self.storage = []
        self.information = {}

        # Read patient information from .cfg file
        with open(filepath / 'Info.cfg', 'r') as f:
            pi = f.read().split('\n')

        # Remove potentially empty last line
        pi = [line for line in pi if len(line) > 0]
        # Convert each line from YAML format to dictionary element
        # Note: yaml reader does not work for some reason, probably because of file extension
        self.information = {camel_to_snake(str(k)): float(v) for k, v in map(lambda l: l.split(':'), pi)}

        # Fetch list of all potential images
        files = [f for f in Path(filepath).iterdir() if f.suffixes == [".nii", ".gz"]]

        for f in files:
            # Discard 4d
            # Discard ground truth as those are fetched by their image
            if "_4d" in str(f) or "_gt" in str(f):
                continue
            
            image = self.fetch_frames(f)

            # Store image information for training
            if has_mask:
                self.images.append(image)

                # Fetch path of mask following dataset nomenclature
                f_gt = self._gt_path(f)
                if f_gt in files:
                    self.masks.append(self.fetch_frames(f_gt))

            # Store information for evaluation (testing)
            else:
                # Loop through slices and save image in intermediate format
                directory = Path(save_path) / f.stem.split('.')[0]
                Path.mkdir(directory, parents=True, exist_ok=True)
                for i, slic in enumerate(image):
                    image_path = directory / f'slice{i:02}.npy'
                    with open(image_path, 'wb') as f:
                        np.save(f, slic)
                    self.storage.append({**self.information, 'image_path': image_path.resolve()})

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
    def fetch_frames(path: Path) -> np.ndarray:
        """Load data from image and mask locations

        Args:
            path (Path): location of image.

        Returns:
            ndarray: image (S, H, W)
        """
        imt, _, _ = load_nii(path)

        return imt.swapaxes(0, 2)
