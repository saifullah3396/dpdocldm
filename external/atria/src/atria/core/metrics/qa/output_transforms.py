from typing import List, Tuple

import torch

from atria.core.models.model_outputs import QAModelOutput, SequenceQAModelOutput


def anls_output_transform(output: QAModelOutput) -> Tuple[List[str], List[str]]:
    assert isinstance(
        output, QAModelOutput
    ), f"Expected {QAModelOutput}, got {type(output)}"
    return output.pred_answers, output.target_answers


def sequence_anls_output_transform(
    output: SequenceQAModelOutput,
) -> Tuple[List[str], List[int], List[int], str, torch.Tensor, torch.Tensor, List[str]]:
    assert isinstance(
        output, SequenceQAModelOutput
    ), f"Expected {SequenceQAModelOutput}, got {type(output)}"
    return (
        output.words,
        output.word_ids,
        output.sequence_ids,
        output.question_id,
        output.start_logits,
        output.end_logits,
        output.gold_answers,
    )
