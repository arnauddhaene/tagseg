from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_aim_run, load_model, save_model, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                create_aim_run, ["params:experiment_name"], "run", name="create_aim_run"
            ),
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
                    "device" "params:train_params",
                    "run",
                ],
                "model",
                name="train_model",
            ),
            node(
                save_model, ["model", "params:trained_model"], None, name="save_model"
            ),
        ]
    )
