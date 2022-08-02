from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_model, save_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                load_model,
                ["params:training"],
                "model",
                name="load_model",
            ),
            node(
                train_model,
                [
                    "model",
                    "loader_train",
                    "loader_val",
                ],
                "trained_model",
                name="train_model",
            ),
            node(
                save_model,
                ["trained_model", "params:training"],
                None,
                name="save_model",
            ),
        ]
    )
