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

    # dataset = [dict(image=image, label=label) for image, label in dataset]

    train_set, val_set = random_split(
        dataset, split, generator=torch.Generator().manual_seed(42)
    )

    log.info(f"Split dataset into {split} train/val.")
    log.info(f"Dataset length is {len(train_set)}/{len(val_set)} train/val.")

    # probability: float = 0.15

    # train_transforms = Compose(
    #     [
    #         HistogramNormalized(keys=["image"]),
    #         RandSpatialCropSamplesd(
    #             keys=["image", "label"],
    #             roi_size=(192, 192),
    #             num_samples=4,
    #             random_center=True,
    #             random_size=False,
    #         ),
    #         Resized(keys=["image", "label"], spatial_size=(256, 256)),
    #         RandAxisFlipd(keys=["image", "label"], prob=probability),
    #         RandAffined(keys=["image", "label"], prob=probability),
    #         Rand2DElasticd(
    #             keys=["image", "label"],
    #             prob=probability,
    #             spacing=(16, 16),
    #             magnitude_range=(1, 2),
    #             rotate_range=0.25,
    #             padding_mode='zeros'
    #         ),
    #         RandBiasFieldd(keys=["image"], prob=probability),
    #         RandGaussianNoised(keys=["image"], prob=probability),
    #         RandGaussianSmoothd(keys=["image"], prob=probability),
    #         RandGaussianSharpend(keys=["image"], prob=probability),
    #         RandKSpaceSpikeNoised(keys=["image"], prob=probability),
    #         EnsureTyped(keys=["image", "label"]),
    #     ]
    # )

    # val_transforms = Compose(
    #     [
    #         HistogramNormalized(keys=["image"]),
    #         EnsureTyped(keys=["image", "label"]),
    #     ]
    # )

    # train_ds = CacheDataset(data=train_set, transform=train_transforms, cache_rate=1.0)
    # val_ds = CacheDataset(data=val_set, transform=val_transforms, cache_rate=1.0)

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
