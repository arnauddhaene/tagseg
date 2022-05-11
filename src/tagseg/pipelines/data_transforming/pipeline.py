from kedro.pipeline import Pipeline, node, pipeline

from .nodes import cine_to_tagged


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                cine_to_tagged,
                ["dataset", "params:transformation"],
                "model_input",
                name="cine_to_tag",
            ),
        ]
    )
