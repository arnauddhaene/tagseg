import logging
from typing import Dict, List, Any

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from tagseg.data.utils import merge_tensor_datasets


def split_data(
    datasets: Dict[str, TensorDataset],
    train_val_split: float,
    batch_size: int,
    data_params: Dict[str, Any]
) -> Dict[str, DataLoader]:
    log = logging.getLogger(__name__)

    dataset: TensorDataset = TensorDataset()

    for name, data in datasets.items():
        assert name in data_params.keys()
        if data_params[name]['include']:
            dataset = merge_tensor_datasets(dataset, data)
    
    n: int = len(dataset)
    n_train: int = round(n * train_val_split)
    split: List[int] = [n_train, n - n_train]

    train_set, val_set = random_split(
        dataset, split, generator=torch.Generator().manual_seed(42)
    )
    loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    log.info(f"Split dataset into {split} train/val.")

    return dict(loader_train=loader_train, loader_val=loader_val)


def fetch_ss_data(dataset: TensorDataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size)
