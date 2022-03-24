"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.17.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_acdc, preprocess_dmd


def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [node(preprocess_dmd, ["params:data_params"], "dataset", name="preprocess")]
    )
