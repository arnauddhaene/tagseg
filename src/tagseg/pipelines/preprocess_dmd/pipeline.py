"""
This is a boilerplate pipeline 'preprocess_dmd'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # This is needed to save dmd as model_input
        node(
            lambda x: x, "dmd_train", "model_input",
            name="preprocess_dmd"
        )
    ])
