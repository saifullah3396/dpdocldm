from atria.core.models.model_outputs import ClassificationModelOutput


def _classification_metrics_output_transform(output: ClassificationModelOutput):
    assert isinstance(
        output, ClassificationModelOutput
    ), f"Expected {ClassificationModelOutput}, got {type(output)}"
    return output.logits, output.label
