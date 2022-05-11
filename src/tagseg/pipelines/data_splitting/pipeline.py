from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                split_data,
                [
                    "model_input",
                    "params:data",
                ],
                dict(loader_train="loader_train", loader_val="loader_val"),
            ),
        ]
    )
