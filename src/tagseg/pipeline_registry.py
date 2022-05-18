"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from tagseg.pipelines import data_processing as dp
from tagseg.pipelines import data_splitting as ds
from tagseg.pipelines import model_training as mt
from tagseg.pipelines import model_tuning as tu


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    preprocess = dp.create_pipeline()
    split = ds.create_pipeline()
    train = mt.create_pipeline()
    tune = tu.create_pipeline()

    return {
        "preprocess": preprocess,
        "train": split + train,
        "find_lr": split + tune,
        "__default__": preprocess + split + train,
    }
