"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_dmd, preprocess_dmd_ss, preprocess_acdc, preprocess_scd, join_data


def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
            node(preprocess_dmd, ["params:data_params"], "dataset_dmd", name="preprocess_dmd"),
            node(preprocess_acdc, ["params:data_params"], "dataset_acdc", name="preprocess_acdc"),
            node(preprocess_scd, ["params:data_params"], "dataset_scd", name="preprocess_scd"),
            node(join_data, [
                "dataset_dmd", "dataset_acdc", "dataset_scd"
            ], "datasets", name="join_data"),
            node(preprocess_dmd_ss, ["params:data_params"], "dataset_dmd_ss", name="preprocess_ss"),
        ]
    )
