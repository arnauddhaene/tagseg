"""
This is a boilerplate pipeline 'data_splitting'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import fetch_ss_data, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                split_data,
                [
                    "model_input",
                    "params:train_val_split",
                    "params:batch_size",
                    "params:data_params",
                ],
                dict(loader_train="loader_train", loader_val="loader_val"),
            ),
            node(
                fetch_ss_data,
                ["dataset_dmd_ss", "params:batch_size"],
                "loader_train_ss",
            ),
        ]
    )
