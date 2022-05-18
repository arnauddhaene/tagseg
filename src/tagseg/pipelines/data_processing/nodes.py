import functools
import logging
from typing import Any, Dict, List
from tqdm import tqdm

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from kedro.config import ConfigLoader

from tagseg.data.dmd_dataset import DmdDataSet, DmdTimeDataSet
from tagseg.data.utils import merge_tensor_datasets
from tagseg.models.cyclegan import Generator
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


def prepare_input(
    dataset: TensorDataset, transformation_params: Dict[str, Any]
) -> TensorDataset:
    log = logging.getLogger(__name__)

    if not transformation_params.get('perform'):
        log.info('Skipping cine->tag transformation, using cine and saving to file.')
        
        output_dataset = TensorDataset()
        output_dataset.tensors = (
            dataset.tensors[0],
            dataset.tensors[1].unsqueeze(1)
        )
        
        return output_dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nc: int = 1  # no. of channels
    generator: torch.nn.Module = Generator(nc, nc)

    saved_model: str = transformation_params["generator_model"]
    generator.load_state_dict(torch.load(saved_model))
    generator.to(device)

    log.info(f"Loaded {str(generator.__class__)} from {saved_model}.")

    generator.eval()

    batch_size = transformation_params['batch_size']
    loader = DataLoader(dataset, batch_size=batch_size)

    output_B = torch.Tensor(len(dataset), 1, 256, 256).to(device)

    for i, (batch, _) in tqdm(enumerate(loader), total=len(loader)):
        span = slice(i * batch_size, i * batch_size + batch.shape[0])

        t = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        batch = Variable(t(*batch.shape).copy_(batch)).to(device)
        output_B[span] = (0.5 * generator(batch).data + 1.0).unsqueeze(0)

    output_dataset = TensorDataset()
    output_dataset.tensors = (
        output_B.cpu(),
        dataset.tensors[1].unsqueeze(1).cpu(),
    )

    log.info("Images transformed to tagged with CycleGAN and saved to file.")
    return output_dataset
