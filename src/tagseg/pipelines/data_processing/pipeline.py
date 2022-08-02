"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    join_data,
    merge_data,
    prepare_input
)


def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
            node(join_data, ["acdc_train", "scd_train", "mnm_train"], "datasets", name="join_data"),
            node(
                merge_data,
                ["datasets", "params:data"],
                "dataset",
            ),
            node(
                prepare_input,
                ["dataset", "params:transformation"],
                "model_input",
                name="prepare_input",
            ),
        ]
    )
