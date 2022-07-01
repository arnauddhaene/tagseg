from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_dmd


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            evaluate_dmd,
            ["dmd_test", "params:evaluation"],
            "dmd_results",
            name="evaluate_dmd"
        )
    ])
