import logging
from typing import Any, Dict

from kedro.config import ConfigLoader
from kedro.extras.datasets.pickle import PickleDataSet
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from tagseg.data.acdc_dataset import AcdcDataSet
from tagseg.data.scd_dataset import ScdDataSet
from tagseg.data.dmd_dataset import DmdDataSet, DmdTimeDataSet
from tagseg.models.unet_ss import UNetSS
from tagseg.data.utils import merge_tensor_datasets


def preprocess_acdc(params: Dict[str, Any]) -> TensorDataset:

    log = logging.getLogger(__name__)

    tagged, only_myo = params["tagged"], params["only_myo"]
    conf_cat = ConfigLoader("conf/base").get("catalog*", "catalog*/**")

    ds_name: str = "acdc_data"
    ds_name += "_tagged" if tagged else "_cine"
    ds_name += "_only_myo" if only_myo else ""

    dataset = PickleDataSet(filepath=conf_cat[ds_name]["filepath"])

    if dataset.exists():
        log.info(
            f"Requested dataset exists, loading from {conf_cat[ds_name]['filepath']}"
        )
        return dataset.load()
    else:
        log.info(
            f"Requested dataset not found, loading it from raw files at \
                {conf_cat['raw_acdc_data']['filepath']}"
        )
        # Specific image preprocessing occurs within AcdcDataSet loading
        acdc = AcdcDataSet(
            filepath=conf_cat["raw_acdc_data"]["filepath"],
            tagged=tagged,
            only_myo=only_myo,
        )
        # Save dataset for next time
        dataset.save(acdc.load())
        log.info("Requested dataset saved to file.")
        return dataset.load()


def preprocess_scd(params: Dict[str, Any]) -> TensorDataset:

    log = logging.getLogger(__name__)

    tagged = params["tagged"]
    conf_cat = ConfigLoader("conf/base").get("catalog*", "catalog*/**")

    ds_name: str = "scd_data"
    ds_name += "_tagged" if tagged else "_cine"

    dataset = PickleDataSet(filepath=conf_cat[ds_name]["filepath"])

    if dataset.exists():
        log.info(
            f"Requested dataset exists, loading from {conf_cat[ds_name]['filepath']}"
        )
        return dataset.load()
    else:
        log.info(
            f"Requested dataset not found, loading it from raw files at \
                {conf_cat['raw_scd_data']['filepath']}"
        )
        # Specific image preprocessing occurs within AcdcDataSet loading
        scd = ScdDataSet(
            filepath=conf_cat["raw_scd_data"]["filepath"],
            tagged=tagged,
        )
        # Save dataset for next time
        dataset.save(scd.load())
        log.info("Requested dataset saved to file.")
        return dataset.load()


def preprocess_dmd(params: Dict[str, Any]) -> TensorDataset:

    log = logging.getLogger(__name__)

    catalog = ConfigLoader("conf/base").get("catalog*", "catalog*/**")
    path = catalog['dmd_data']['filepath']

    log.info(f"Loading requested dataset from raw files at {path}")

    dataset = DmdDataSet(filepath=path).load()

    if params['semi_supervised']:

        ss_loader = DataLoader(preprocess_dmd_ss(params), batch_size=16)

        model = UNetSS(n_channels=1, n_classes=2, bilinear=True).double()

        pretrained_path = params["base_model"]
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
            labels = torch.cat((labels, F.softmax(outs['logits'], dim=1).argmax(dim=1).cpu()), axis=0)

        predicted_dataset = TensorDataset()
        predicted_dataset.tensors = (images, labels,)

        dataset = merge_tensor_datasets(dataset, predicted_dataset)

    return dataset


def preprocess_dmd_ss(params: Dict[str, Any]) -> TensorDataset:

    log = logging.getLogger(__name__)

    catalog = ConfigLoader("conf/base").get("catalog*", "catalog*/**")
    path = catalog['dmd_time_data']['filepath']

    log.info(f"Loading requested dataset from raw files at {path}")

    videos: torch.Tensor = DmdTimeDataSet(filepath=path).load().tensors[1]

    # Select all images except for the first ones who are present
    # in the labelled dataset
    # FlatMap them using combination of torch.cat and unbind
    return torch.cat(videos[:, 1:].unbind()).unsqueeze(1)


def join_data(
    dataset_acdc: TensorDataset,
    dataset_dmd: TensorDataset,
    dataset_scd: TensorDataset,
):
    return dict(
        acdc=dataset_acdc,
        dmd=dataset_dmd,
        scd=dataset_scd
    )
