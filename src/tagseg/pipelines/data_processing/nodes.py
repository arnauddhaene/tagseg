import functools
import logging
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from kedro.config import ConfigLoader
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tagseg.data.dmd_dataset import DmdDataSet, DmdTimeDataSet
from tagseg.data.utils import merge_tensor_datasets
from tagseg.models.unet_ss import UNetSS


def preprocess_dmd(params: Dict[str, Any]) -> TensorDataset:

    log = logging.getLogger(__name__)

    catalog = ConfigLoader("conf/base").get("catalog*", "catalog*/**")
    path = catalog["dmd"]["filepath"]

    log.info(f"Loading requested dataset from raw files at {path}")

    dataset = DmdDataSet(filepath=path).load()

    if params["semi_supervised"]:

        ss_loader = DataLoader(preprocess_dmd_ss(params), batch_size=16)

        model = UNetSS(n_channels=1, n_classes=2, bilinear=True).double()

        pretrained_path = params["ss_model"]
        if pretrained_path is not None:
            # Load old saved version of the model as a state dictionary
            saved_model_sd = torch.load(pretrained_path)
            # Extract UNet if saved model is parallelized
            model.load_state_dict(saved_model_sd)
            log.info(f"Weights loaded from saved model at {pretrained_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type == "cuda":
            model = nn.DataParallel(model)
            model.n_classes = model.module.n_classes
            model.to(device)
            log.info("Model parallelized on CUDA")

        images: torch.Tensor = torch.Tensor()
        labels: torch.Tensor = torch.Tensor()

        model.eval()

        for inps in ss_loader:

            inps = inps.double().to(device)
            outs = model(inps)

            images = torch.cat((images, inps.cpu()), axis=0)
            labels = torch.cat(
                (labels, F.softmax(outs["logits"], dim=1).argmax(dim=1).cpu()), axis=0
            )

        predicted_dataset = TensorDataset()
        predicted_dataset.tensors = (
            images,
            labels,
        )

        dataset = merge_tensor_datasets(dataset, predicted_dataset)

    return dataset


def preprocess_dmd_ss(params: Dict[str, Any]) -> TensorDataset:

    log = logging.getLogger(__name__)

    catalog = ConfigLoader("conf/base").get("catalog*", "catalog*/**")
    path = catalog["tdmd"]["filepath"]

    log.info(f"Loading requested dataset from raw files at {path}")

    videos: torch.Tensor = DmdTimeDataSet(filepath=path).load().tensors[1]

    # Select all images except for the first ones who are present
    # in the labelled dataset
    # FlatMap them using combination of torch.cat and unbind
    return torch.cat(videos[:, 1:].unbind()).unsqueeze(1)


def join_data(
    dataset_acdc: TensorDataset,
    dataset_scd: TensorDataset,
    dataset_mnm: TensorDataset
):
    return dict(acdc=dataset_acdc, scd=dataset_scd, mnm=dataset_mnm)


def merge_data(
    datasets: Dict[str, TensorDataset], data_params: Dict[str, Any]
) -> TensorDataset:
    log = logging.getLogger(__name__)

    dataset: List[TensorDataset] = []

    for name, data in datasets.items():
        assert name in data_params.keys()
        if data_params[name]["include"]:
            log.info(f"Including dataset {name} of length {len(data)}.")
            dataset.append(data)

    return functools.reduce(merge_tensor_datasets, dataset)
