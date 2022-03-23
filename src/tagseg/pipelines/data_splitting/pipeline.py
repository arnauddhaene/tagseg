"""
This is a boilerplate pipeline 'data_splitting'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                split_data,
                ["dataset", "params:train_val_split", "params:batch_size"],
                dict(loader_train="loader_train", loader_val="loader_val"),
            )
        ]
    )
