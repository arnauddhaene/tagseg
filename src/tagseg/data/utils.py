import numpy as np
from scipy import ndimage

import torch
from torch.utils.data import TensorDataset


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
        result = np.stack([
            (result ** 2).sum(axis=0) ** .5,  # sqrt(x^2 + y^2)
            np.arctan(result[1] / (result[0] + 1e-8))  # arctan(y/x)
        ])

        return result

    offset = 1 if exclude_bg else 0
    
    def example_df(ex: np.ndarray):
        return np.array(list(map(channel_df, ex[offset:]))).sum(axis=0)

    return np.array(list(map(example_df, inp)))
