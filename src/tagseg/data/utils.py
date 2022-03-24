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
