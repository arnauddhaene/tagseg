import logging
from typing import Dict, Any
from tqdm import tqdm

from kedro.config import ConfigLoader
from kedro.extras.datasets.pickle import PickleDataSet
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from tagseg.models.cyclegan import Generator


def cine_to_tagged(
    dataset: TensorDataset,
    batch_size: int,
    data_params: Dict[str, Any]
) -> TensorDataset:
    log = logging.getLogger(__name__)
    conf_cat = ConfigLoader("conf/base").get("catalog*", "catalog*/**")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nc: int = 1  # no. of channels
    generator: torch.nn.Module = Generator(nc, nc)

    saved_model: str = data_params['generator_model']
    generator.load_state_dict(torch.load(saved_model))
    generator.to(device)

    log.info(f"Loaded {str(generator.__class__)} from {saved_model}.")

    generator.eval()

    loader = DataLoader(dataset, batch_size=batch_size)

    output_B = torch.Tensor(len(dataset), 1, 256, 256).to(device)

    for i, (batch, _) in tqdm(enumerate(loader)):
        span = slice(i * batch_size, i * batch_size + batch.shape[0])

        t = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        batch = Variable(t(*batch.shape).copy_(batch)).to(device)
        output_B[span] = 0.5 * generator(batch).data + 1.0

    output_dataset = TensorDataset()
    output_dataset.tensors = (output_B, dataset.tensors[1],)

    tagged_dataset = PickleDataSet(filepath=conf_cat['model_input']['filepath'])
    tagged_dataset.save(output_dataset)

    log.info("Images transformed to tagged with CycleGAN and saved to file.")
    return tagged_dataset.load()
