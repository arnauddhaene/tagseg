import logging
from typing import Any, Dict

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from kedro.config import ConfigLoader
from kedro.extras.datasets.pickle import PickleDataSet

from tagseg.models.cyclegan import Generator


def cine_to_tagged(
    dataset: TensorDataset, transformation_params: Dict[str, Any]
) -> TensorDataset:
    log = logging.getLogger(__name__)

    conf_paths = ["conf/base", "conf/local"]
    conf_loader = ConfigLoader(conf_paths)
    conf_catalog = conf_loader.get("catalog*", "catalog*/**")

    saved_dataset = PickleDataSet(filepath=conf_catalog['model_input']['filepath'])

    if saved_dataset.exists():
        return saved_dataset.load()

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
