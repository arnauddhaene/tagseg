"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import join_data, merge_data, preprocess_dmd, preprocess_dmd_ss


def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
            node(
                preprocess_dmd,
                ["params:training"],
                "dataset_dmd",
                name="preprocess_dmd",
            ),
            node(
                preprocess_dmd_ss,
                ["params:training"],
                "dataset_dmd_ss",
                name="preprocess_ss",
            ),
            node(join_data, ["acdc_train", "scd_train"], "datasets", name="join_data"),
            node(
                merge_data,
                ["datasets", "params:data"],
                "dataset",
            ),
        ]
    )
