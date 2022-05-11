import numpy as np
from scipy import ndimage
import nibabel as nib
import torch
from torch.utils.data import TensorDataset

from .cine2gridtag import cine2gridtag


def merge_tensor_datasets(a: TensorDataset, b: TensorDataset) -> TensorDataset:

    if len(a.tensors) != len(b.tensors):
        raise ValueError(
            f"TensorDatasets do not have same number of tensors: \
                         {len(a.tensors)} vs. {len(b.tensors)}"
        )

    dataset: TensorDataset = TensorDataset()
    tensors = ()

    for tensor_a, tensor_b in zip(a.tensors, b.tensors):
        if tensor_a.ndim != tensor_b.ndim:
            raise ValueError(
                f"Tensors do not have same number of dimensions: \
                             {tensor_a.ndim} vs. {tensor_b.ndim}"
            )
        tensors += (torch.cat([tensor_a, tensor_b]),)

    dataset.tensors = tensors

    return dataset


def directional_field(inp: np.ndarray, exclude_bg: bool = True) -> np.ndarray:
    """My implementation of https://arxiv.org/abs/2007.11349

    Args:
        inp (np.ndarray): Input tensor of dimensions B x C x H x W
        exclude_bg (bool, optional): Exclude background. Defaults to True.

    Returns:
        np.ndarray: Magnitude and direction of directional field in size B x 2 x H x W.
    """

    def channel_df(x: np.ndarray) -> np.ndarray:

        result = np.zeros((2, *x.shape), dtype=np.float32)

        _, ind = ndimage.distance_transform_edt(x.astype(np.uint8), return_indices=True)
        diff = np.indices(x.shape) - ind

        # Assign (x, y) distance
        result[:, x > 0] = diff[:, x > 0]

        # Cartesian to polar coordinates
        result = np.stack(
            [
                (result ** 2).sum(axis=0) ** 0.5,  # sqrt(x^2 + y^2)
                np.arctan(result[1] / (result[0] + 1e-8)),  # arctan(y/x)
            ]
        )

        return result

    offset = 1 if exclude_bg else 0

    def example_df(ex: np.ndarray):
        return np.array(list(map(channel_df, ex[offset:]))).sum(axis=0)

    return np.array(list(map(example_df, inp)))


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
