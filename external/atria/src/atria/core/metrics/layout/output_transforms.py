from atria.core.models.model_outputs import LayoutTokenClassificationModelOutput


def _layout_token_classification_metrics_output_transform(
    output: LayoutTokenClassificationModelOutput,
):
    assert isinstance(
        output, LayoutTokenClassificationModelOutput
    ), f"Expected {LayoutTokenClassificationModelOutput}, got {type(output)}"
    return output.logits, output.label, output.token_bboxes
