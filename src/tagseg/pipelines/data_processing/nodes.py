import functools
import logging
from typing import Any, Dict, List
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from tagseg.data.utils import merge_tensor_datasets
from tagseg.models.cyclegan import Generator


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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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
