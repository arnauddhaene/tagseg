from kedro.pipeline import Pipeline, node, pipeline

from .nodes import find_best_lr
from ..model_training.nodes import load_model


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
                find_best_lr,
                [
                    "model",
                    "loader_train",
                    "loader_val",
                ],
                "suggestion",
                name="find_best_lr",
            )
        ]
    )
