"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from tagseg.pipelines import data_processing as dp
from tagseg.pipelines import data_splitting as ds
from tagseg.pipelines import model_training as mt
from tagseg.pipelines import data_transforming as dt


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    preprocess = dp.create_pipeline()
    split = ds.create_pipeline()
    train = mt.create_pipeline()
    cine_to_tagged = dt.create_pipeline()

    return {
        "preprocess": preprocess,
        "cine2tag": preprocess + cine_to_tagged,
        "__default__": preprocess + cine_to_tagged + split + train
    }
