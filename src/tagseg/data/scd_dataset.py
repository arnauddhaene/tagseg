import functools
import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pydicom
import torch
from skimage.draw import polygon, polygon2mask
from torch.utils.data import TensorDataset
from tqdm import tqdm

from .dataset import TagSegDataSet


class ScdDataSet(TagSegDataSet):
    def _load_except(self, filepath_raw: List[str]) -> TensorDataset:

        images: torch.Tensor = torch.Tensor()
        labels: torch.Tensor = torch.Tensor()

        skip_nan: int = 0
        skip_unlabeled: int = 0

        for filep in filepath_raw:
            
            subfolders = list(Path(filep).iterdir())

            contour_superfolder = list(
                filter(lambda s: "Contours" in s.name, subfolders)
            )[0]
            contour_folder = list(
                filter(lambda p: p.is_dir(), contour_superfolder.iterdir())
            )[0]

            dicom_superfolder = list(filter(lambda s: "DICOM" in s.name, subfolders))[0]
            dicom_folder = list(
                filter(lambda p: p.is_dir(), dicom_superfolder.iterdir())
            )[0]

            patients = [d for d in contour_folder.iterdir() if d.is_dir()]

            for patient in tqdm(patients):

                if patient.name == "file-listings":
                    continue

                contours = [
                    f
                    for f in (patient / "contours-manual" / "IRCCI-expert").iterdir()
                    if (f.is_file() and f.suffix == ".txt")
                ]

                cont_ptr = {}
                for contour in contours:
                    _, _, no, _, _ = contour.stem.split("-")

                    no = f"IM-0001-{int(no):04}"

                    if no not in cont_ptr.keys():
                        cont_ptr[no] = [contour]
                    else:
                        cont_ptr[no].append(contour)

                for no, conts in cont_ptr.items():
                    # choose only inner and outer
                    conts = [
                        cont
                        for cont in conts
                        if ("icontour" in str(cont) or "ocontour" in str(cont))
                    ]

                    # skip annotations that don't include endo- and epi-cardial wall
                    if len(conts) < 2:
                        continue

                    image_path = dicom_folder / patient.name / "DICOM" / (no + ".dcm")
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
                    image = self._preprocess_image(0.456, 0.224)(image).unsqueeze(0)

                    # Exclude NaNs from dataset
                    if image.isnan().sum().item() > 0 or label.isnan().sum().item() > 0:
                        skip_nan += 1
                        continue

                    if torch.count_nonzero(label).item() == 0:
                        skip_unlabeled += 1
                        continue

                    images = torch.cat((images, image), axis=0)
                    labels = torch.cat((labels, label), axis=0)

        log = logging.getLogger(__name__)
        log.info(f"Skipped {skip_nan} image(s) due to presence of NaN")
        log.info(f"Skipped {skip_unlabeled} image(s) due to absence of label")

        dataset = TensorDataset()
        dataset.tensors = (
            images,
            labels,
        )

        return dataset

    @staticmethod
    def get_mask(shape: Tuple[int, int], path: Path) -> np.ndarray:

        pts = ScdDataSet.get_points(path)
        pg = polygon(pts[:, 1], pts[:, 0], shape)
        mask = polygon2mask(shape, np.array(pg).T)

        return mask

    @staticmethod
    def get_points(path: Path) -> np.ndarray:
        with open(path, "r") as f:
            lines = f.read().split("\n")

        # remove last line if empty
        if lines[-1] == "":
            lines = lines[:-1]
        return np.array([x_y.split(" ") for x_y in lines]).astype(np.float64)
