from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_model, save_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                load_model,
                ["params:training"],
                dict(model="model", device="device"),
                name="load_model",
            ),
            node(
                train_model,
                [
                    "model",
                    "loader_train",
                    "loader_train_ss",
                    "loader_val",
                    "device",
                ],
                "trained_model",
                name="train_model",
            ),
            node(
                save_model,
                ["trained_model", "params:training:checkpoint"],
                None,
                name="save_model",
            ),
        ]
    )
