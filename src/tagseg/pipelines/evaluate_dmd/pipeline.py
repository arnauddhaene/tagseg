from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_dmd


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            evaluate_dmd,
            ["dmd_test", "params:evaluation"],
            "output",
            name="evaluate_dmd"
        ),
        node(
            evaluate_dmd,
            ["dmd_test_train", "params:evaluation"],
            "output_train",
            name="evaluate_dmd_train"
        )
    ])
