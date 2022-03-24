import logging
from typing import Any, Dict

from kedro.config import ConfigLoader
from kedro.extras.datasets.pickle import PickleDataSet
from torch.utils.data import TensorDataset

from tagseg.data.acdc_dataset import AcdcDataSet
from tagseg.data.dmd_dataset import DmdDataSet


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


def preprocess_dmd(params: Dict[str, Any]) -> TensorDataset:

    log = logging.getLogger(__name__)

    catalog = ConfigLoader("conf/base").get("catalog*", "catalog*/**")
    path = catalog['dmd_data']['filepath']

    log.info(f"Loading requested dataset from raw files at {path}")

    return DmdDataSet(filepath=path).load()
