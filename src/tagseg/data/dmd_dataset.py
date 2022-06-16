from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pydicom
import torch
import torchio as tio
import h5py
from kedro.io import AbstractDataSet
from skimage.draw import polygon, polygon2mask
from torch.utils.data import TensorDataset
from torchvision import transforms

from .dataset import TagSegDataSet


class DmdH5DataSet(TagSegDataSet):
    def _load_except(self, filepath_raw: str) -> TensorDataset:

        filepath_raw = Path(filepath_raw)

        images: torch.Tensor = torch.Tensor()
        labels: torch.Tensor = torch.Tensor()

        for roi_path in [path for path in filepath_raw.iterdir() if path.stem.split('_')[-1] == 'roi']:
            
            img_path = roi_path.parent / ('_'.join(roi_path.stem.split('_')[:-1]) + '.h5')

            assert img_path.is_file()

            img_hf = h5py.File(img_path, 'r')
            roi_hf = h5py.File(roi_path, 'r')

            assert 'imt' in img_hf.keys()
            assert all(map(lambda key: key in roi_hf.keys(), ['pts_interp_inner', 'pts_interp_outer']))

            imt = np.array(img_hf.get('imt')).swapaxes(0, 2)
            pts_inner = np.array(list(map(lambda i: np.array(roi_hf[roi_hf.get('pts_interp_inner')[i][0]]),
                                          range(roi_hf.get('pts_interp_inner').shape[0]))))
            pts_outer = np.array(list(map(lambda i: np.array(roi_hf[roi_hf.get('pts_interp_outer')[i][0]]),
                                          range(roi_hf.get('pts_interp_inner').shape[0]))))
            
            for t in range(imt.shape[0]):
                image = imt[t]
                image = image / image.max()
                image = self._preprocess_image(0.456, 0.224)(image).unsqueeze(0)

                inner = polygon2mask(imt.shape[1:],
                                     np.array(polygon(pts_inner[t, :, 1], pts_inner[t, :, 0])).T)
                outer = polygon2mask(imt.shape[1:],
                                     np.array(polygon(pts_outer[t, :, 1], pts_outer[t, :, 0])).T)

                label = outer ^ inner
                label = label.astype(np.float64)
                label = self._preprocess_label()(label)

                images = torch.cat((images, image), axis=0)
                labels = torch.cat((labels, label), axis=0)

                dataset = TensorDataset()
                dataset.tensors = (
                    images,
                    labels,
                )

        return dataset


class DmdDataSet(TagSegDataSet):
    def _load_except(self, filepath_raw: str) -> TensorDataset:
        
        filepath_raw = Path(filepath_raw)

        HEALTHY_DIR, DMD_DIR = (
            Path(filepath_raw) / "healthy",
            Path(filepath_raw) / "dmd",
        )

        subjects: List[tio.Subject] = []

        for directory in [HEALTHY_DIR, DMD_DIR]:
            # Iterate over all scans for each folder
            scans = [
                Scan(patient) for patient in directory.iterdir() if patient.is_dir()
            ]

            for scan in scans:
                for slic in scan.slices.values():
                    if slic.is_annotated():

                        image = slic.image[0]
                        image = image.astype(np.float64)
                        # Preprocess
                        image = image / image.max()
                        image = self._preprocess_image(0.456, 0.224)(image)

                        label = slic.mask["outer"] ^ slic.mask["inner"]
                        label = label.astype(np.float64)
                        label = self._preprocess_label()(label)

                        subjects.append(tio.Subject(
                            image=tio.ScalarImage(tensor=image[None, ...]),
                            mask=tio.LabelMap(tensor=label[None, ...])
                        ))

        return tio.SubjectsDataset(subjects)


class DmdTimeDataSet(AbstractDataSet):
    def __init__(self, filepath: str):
        self._filepath = Path(filepath)

    def _load(self) -> TensorDataset:

        HEALTHY_DIR, DMD_DIR = (
            Path(self._filepath) / "healthy",
            Path(self._filepath) / "dmd",
        )

        images: torch.Tensor = torch.Tensor()
        videos: torch.Tensor = torch.Tensor()
        labels: torch.Tensor = torch.Tensor()
        slices: List[str] = []
        conditions: List[str] = []

        for directory in [HEALTHY_DIR, DMD_DIR]:
            # Iterate over all scans for each folder
            scans = [
                Scan(patient) for patient in directory.iterdir() if patient.is_dir()
            ]

            for scan in scans:
                for slic in scan.slices.values():

                    if slic.is_annotated():

                        video = slic.image.astype(np.float64)
                        image = video[0]

                        video = self._preprocess_video()(torch.Tensor(video)).unsqueeze(
                            0
                        )

                        # Preprocess
                        image = image / image.max()
                        image = self._preprocess_image(0.456, 0.224)(image).unsqueeze(0)

                        label = slic.mask["outer"] ^ slic.mask["inner"]
                        label = label.astype(np.float64)
                        label = self._preprocess_label()(label)

                        images = torch.cat((images, image), axis=0)
                        videos = torch.cat((videos, video), axis=0)
                        labels = torch.cat((labels, label), axis=0)
                        slices.append(slic.slice_location)
                        conditions.append(str(directory.name))

        dataset = TensorDataset()
        dataset.slices = slices
        dataset.conditions = conditions
        dataset.tensors = (
            images,
            videos,
            labels,
        )

        return dataset

    def _save(self, dataset: TensorDataset) -> None:
        pass

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _describe(self) -> Dict[str, Any]:
        return "DMD Dataset"

    def _preprocess_image(
        self,
        mu: float,
        sigma: float,
    ) -> transforms.Compose:
        """Preprocess image

        Args:
            mu (float): average for normalization layer
            sigma (float): standard deviation for normalization layer

        Returns:
            transforms.Compose: transformation callback function
        """
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mu, std=sigma),
                transforms.Resize((256, 256)),
            ]
        )

    def _preprocess_video(self) -> transforms.Compose:
        """Preprocess video

        Returns:
            transforms.Compose: transformation callback function
        """
        return transforms.Compose([transforms.Resize((256, 256))])

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


class Slice:
    def __init__(self, path: Path):

        self.slice = path.name

        dcm_images = [f for f in path.iterdir() if f.is_file() and f.suffix == ".dcm"]
        if len(dcm_images) > 0:
            res = map(
                lambda ds: (ds.InstanceNumber, ds.pixel_array),
                map(pydicom.dcmread, dcm_images),
            )
            self.image = np.array(list(zip(*sorted(res, key=lambda item: item[0])))[1])

        # Extract ROI if it exists
        roi_path = path / "roi_pts.npz"
        self.slice_location = path.name
        if roi_path.is_file():
            rois = np.load(roi_path)
            self.roi = {array_key: rois[array_key] for array_key in list(rois.keys())}
            self.mask = {}
            for _roi in ["pts_interp_outer", "pts_interp_inner"]:
                if self.roi[_roi] is not None:
                    # Verify dimensions
                    assert self.roi[_roi].shape[0] == 2

                    pg = polygon(
                        self.roi[_roi][1, :], self.roi[_roi][0, :], self.image.shape[1:]
                    )
                    # Save under 'outer' or 'inner' key
                    self.mask[_roi.split("_")[-1]] = polygon2mask(
                        self.image.shape[1:], np.array(pg).T
                    )
                else:
                    self.mask[_roi.split("_")[-1]] = None

        else:
            self.roi = None

    def is_annotated(self) -> bool:
        return self.roi is not None

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"{self.slice} slice with{'out' if self.roi is None else ''} annotated ROI."
        )


class Scan:
    def __init__(self, path: Path, label: bool = False):

        self.id = path.name
        self.label = label
        self.slices = {
            s.name: Slice(s) for s in (path / "clean").iterdir() if s.is_dir()
        }
