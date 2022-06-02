from kedro.pipeline import Pipeline, node, pipeline

from .nodes import tag_subjects, evaluate


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            tag_subjects,
            ["mnm_test", "params:transformation"],
            "mnm_test_tagged",
            name="tag_mnm"
        ),
        node(
            tag_subjects,
            ["scd_test", "params:transformation"],
            "scd_test_tagged",
            name="tag_scd"
        ),
        node(
            evaluate,
            ["mnm_test_tagged", "scd_test_tagged", "dmd", "params:evaluation"],
            dict(
                mnm_results="mnm_results",
                scd_results="scd_results",
                dmd_results="dmd_results"
            ),
            name="evaluate"
        )
    ])
