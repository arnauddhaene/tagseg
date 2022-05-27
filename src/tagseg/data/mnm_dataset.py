import logging
from pathlib import Path, PosixPath
from typing import List, Dict, Any
from itertools import product

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from .dataset import TagSegDataSet, EvalInfoDataSet
from .utils import load_nii


class MnmDataSet(TagSegDataSet):
    def _load_except(self, filepath_raw: str, only_myo: bool) -> TensorDataset:

        patients: List[PosixPath] = list(filter(lambda p: p.is_dir(), Path(filepath_raw).iterdir()))

        # Initialize empty image and label tensors
        images: torch.Tensor = torch.Tensor()
        labels: torch.Tensor = torch.Tensor()

        skip_label: int = 0
        skip_nan: int = 0
        skip_unlabeled: int = 0

        accepted_classes: set = set([0.0, 1.0, 2.0, 3.0])

        # Iterate over each patient or 'external_code'
        for ext_code in tqdm(patients, unit='patient'):

            image_path = ext_code / f'{ext_code.name}_sa.nii.gz'
            label_path = ext_code / f'{ext_code.name}_sa_gt.nii.gz'

            assert all([image_path.exists(), label_path.exists()])

            # Load images in shape format (width, height, slice, phase)
            nii_images = load_nii(image_path)[0]
            nii_labels = load_nii(label_path)[0].astype(np.int32)

            # Find labeled slice/phase pairs
            idx_valid_imgs = np.where(nii_labels == 2)
            # Perhaps the image is completely unlabeled
            if len(idx_valid_imgs[0]) == 0:
                skip_unlabeled += 1
                continue
            slices, phases = tuple(map(np.unique, idx_valid_imgs[2:]))

            for c_slice, c_phase in product(slices, phases):

                # Preprocess labels and images
                image: np.ndarray = nii_images[..., c_slice, c_phase].astype(np.float64)
                label: np.ndarray = nii_labels[..., c_slice, c_phase].astype(np.float64)
                
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

                if torch.count_nonzero(label) == 0:
                    skip_unlabeled += 1
                    continue

                images = torch.cat((images, image), axis=0)
                labels = torch.cat((labels, label), axis=0)

        log = logging.getLogger(__name__)
        log.info(f"Skipped {skip_label} image(s) due to incoherent label")
        log.info(f"Skipped {skip_nan} image(s) due to presence of NaN")
        log.info(f"Skipped {skip_unlabeled} image(s) due to complete absence of labels")

        dataset = TensorDataset()
        dataset.tensors = (
            images,
            labels,
        )

        return dataset


class MnmEvaluator(EvalInfoDataSet):

    def _load_except(self, filepath_raw: str, patient_info: str) -> pd.DataFrame:

        # Initialize storage that will be converted to pd.DataFrame
        storage: List[Dict[str, Any]] = []

        pi = pd.read_csv(patient_info, index_col=0)

        save_directory = Path(self._filepath.parent / self._filepath.stem)
        Path.mkdir(save_directory, parents=True, exist_ok=True)

        patients: List[PosixPath] = list(filter(lambda p: p.is_dir(), Path(filepath_raw).iterdir()))

        # Iterate over each patient or 'external_code'
        for ext_code in tqdm(patients, unit='patient'):

            image_path = ext_code / f'{ext_code.name}_sa.nii.gz'
            label_path = ext_code / f'{ext_code.name}_sa_gt.nii.gz'

            assert all([image_path.exists(), label_path.exists()])

            # Load images in shape format (width, height, slice, phase)
            nii_images = load_nii(image_path)[0]
            nii_labels = load_nii(label_path)[0].astype(np.int32)

            # Find labeled slice/phase pairs
            idx_valid_imgs = np.where(nii_labels == 2)
            
            # Perhaps the image is completely unlabeled
            slices, phases = tuple(map(np.unique, idx_valid_imgs[2:]))

            for c_slice, c_phase in product(slices, phases):

                # Preprocess labels and images
                image: np.ndarray = nii_images[..., c_slice, c_phase].astype(np.float64)
                label: np.ndarray = nii_labels[..., c_slice, c_phase].astype(np.float64)

                if 2 in np.unique(label):

                    image_save_path = save_directory / f'{ext_code.name}_{c_slice}_{c_phase}_image.npy'
                    with open(image_save_path, 'wb') as f:
                        np.save(f, image)

                    label_save_path = save_directory / f'{ext_code.name}_{c_slice}_{c_phase}_label.npy'
                    with open(label_save_path, 'wb') as f:
                        np.save(f, label)

                    features = [
                        'VendorName', 'Vendor', 'Centre', 'ED', 'ES',
                        'Age', 'Pathology', 'Sex', 'Height', 'Weight'
                    ]
                    information = pi[pi['External code'] == ext_code.name][features].iloc[0].to_dict()

                    storage.append({
                        **information,
                        'image_path': image_save_path.resolve(),
                        'label_path': label_save_path.resolve()
                    })

        return pd.DataFrame(storage)
