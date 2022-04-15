import logging
from pathlib import Path
from typing import Any, Dict, Tuple
import functools

import numpy as np
import pydicom
import torch
from kedro.io import AbstractDataSet
from torch.utils.data import TensorDataset
from torchvision import transforms
from skimage.draw import polygon, polygon2mask

from .utils import SimulateTags


class ScdDataSet(AbstractDataSet):
    def __init__(self, filepath: str, tagged: bool = True):
        self._filepath = Path(filepath)
        self._tagged = tagged

    def _load(self) -> TensorDataset:

        folders = [
            (
                self._filepath / 'Sunnybrook Cardiac MR Database ContoursPart1/OnlineDataContours',
                self._filepath / 'Sunnybrook Cardiac MR Database DICOMPart1/OnlineDataDICOM'),
            (
                self._filepath / 'Sunnybrook Cardiac MR Database ContoursPart2/ValidationDataContours',
                self._filepath / 'Sunnybrook Cardiac MR Database DICOMPart2/ValidationDataDICOM',
            ),
            (
                self._filepath / 'Sunnybrook Cardiac MR Database ContoursPart3/TrainingDataContours',
                self._filepath / 'Sunnybrook Cardiac MR Database DICOMPart3/TrainingDataDICOM',
            )
        ]
        
        images: torch.Tensor = torch.Tensor()
        labels: torch.Tensor = torch.Tensor()

        skip_nan: int = 0

        for contour_folder, dicom_folder in folders:

            patients = [d for d in contour_folder.iterdir() if d.is_dir()]

            for patient in patients:
                
                if patient.name == 'file-listings':
                    continue

                contours = [f for f in (patient / 'contours-manual' / 'IRCCI-expert').iterdir()
                            if (f.is_file() and f.suffix == '.txt')]
                
                cont_ptr = {}
                for contour in contours:
                    _, _, no, _, _ = contour.stem.split('-')

                    no = f"IM-0001-{int(no):04}"

                    if no not in cont_ptr.keys():
                        cont_ptr[no] = [contour]
                    else:
                        cont_ptr[no].append(contour)

                for no, conts in cont_ptr.items():
                    # choose only inner and outer
                    conts = [cont for cont in conts if ('icontour' in str(cont) or 'ocontour' in str(cont))]
                    
                    # skip annotations that don't include endo- and epi-cardial wall
                    if len(conts) < 2:
                        continue

                    image_path = dicom_folder / patient.name / 'DICOM' / (no + '.dcm')
                    image = pydicom.dcmread(image_path).pixel_array.astype(np.float64)

                    mask_me = functools.partial(self.get_mask, image.shape)
                    # alphabetical sorting will yield inner before outer
                    inner, outer = tuple(map(mask_me, sorted(conts)))

                    label = (outer ^ inner).astype(np.float64)

                    # Preprocess labels and images
                    label = self._preprocess_label()(
                        label
                    )  # Label first as we use the resized version
                    image = image / image.max()  # To [0, 1] range
                    image = self._preprocess_image(0.456, 0.224, label)(image).unsqueeze(0)

                    # Exclude NaNs from dataset
                    if image.isnan().sum().item() > 0 or label.isnan().sum().item() > 0:
                        skip_nan += 1
                        continue

                    images = torch.cat((images, image), axis=0)
                    labels = torch.cat((labels, label), axis=0)

        log = logging.getLogger(__name__)
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
        return dict(tagged=self._tagged)

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

    @staticmethod
    def get_mask(shape: Tuple[int, int], path: Path) -> np.ndarray:

        pts = ScdDataSet.get_points(path)
        pg = polygon(pts[:, 1], pts[:, 0], shape)
        mask = polygon2mask(shape, np.array(pg).T)

        return mask

    @staticmethod
    def get_points(path: Path) -> np.ndarray:
        with open(path, 'r') as f:
            lines = f.read().split('\n')
        
        # remove last line if empty
        if lines[-1] == '':
            lines = lines[:-1]
        return np.array([x_y.split(' ') for x_y in lines]).astype(np.float64)
