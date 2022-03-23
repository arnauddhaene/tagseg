from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_model, save_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                load_model,
                ["params:data_params"],
                dict(model="model", device="device"),
                name="load_model",
            ),
            node(
                train_model,
                [
                    "model",
                    "loader_train",
                    "loader_val",
                    "device",
                    "params:train_params",
                    "params:experiment_name"
                ],
                "trained_model",
                name="train_model",
            ),
            node(
                save_model, ["trained_model", "params:trained_model"], None, name="save_model"
            ),
        ]
    )
