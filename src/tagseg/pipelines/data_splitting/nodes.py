import logging
from typing import Dict, List, Any

# from monai.transforms import (
#     HistogramNormalized,
#     Compose,
#     RandSpatialCropSamplesd,
#     RandAxisFlipd,
#     RandAffined,
#     Rand2DElasticd,
#     RandBiasFieldd,
#     RandGaussianNoised,
#     RandGaussianSmoothd,
#     RandGaussianSharpend,
#     RandKSpaceSpikeNoised,
#     Resized,
#     EnsureTyped
# )

# from monai.data import list_data_collate, CacheDataset

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader


def split_data(
    dataset: TensorDataset,
    data_params: Dict[str, Any]
) -> Dict[str, DataLoader]:
    train_val_split: float = data_params['train_val_split']
    batch_size: int = data_params['batch_size']
    log = logging.getLogger(__name__)

    n: int = len(dataset)
    n_train: int = round(n * train_val_split)
    split: List[int] = [n_train, n - n_train]

    train_set, val_set = random_split(
        dataset, split, generator=torch.Generator().manual_seed(42)
    )

    log.info(f"Split dataset into {split} train/val.")
    log.info(f"Dataset length is {len(train_set)}/{len(val_set)} train/val.")

    train_ds, val_ds = train_set, val_set

    loader_train = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  # collate_fn=list_data_collate,
        num_workers=64
    )

    loader_val = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=64
    )

    return dict(loader_train=loader_train, loader_val=loader_val)
