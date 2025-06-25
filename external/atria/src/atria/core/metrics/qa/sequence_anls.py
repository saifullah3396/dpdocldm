import collections
import json
from typing import Callable, List, Union

import numpy as np
import textdistance as td
import torch
from atria.core.metrics.common.epoch_dict_metric import EpochDictMetric
from atria.core.utilities.logging import get_logger
from ignite.metrics import Metric

from anls import anls_score

logger = get_logger(__name__)


def convert_to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    return x


def postprocess_qa_predictions(
    words,
    word_ids,
    sequence_ids,
    question_ids,
    start_logits,
    end_logits,
    n_best_size: int = 20,
    max_answer_length: int = 100,
):
    # words = words
    word_ids = convert_to_list(word_ids)
    sequence_ids = convert_to_list(sequence_ids)
    question_ids = convert_to_list(question_ids)

    features_per_example = collections.defaultdict(list)
    for feature_id, question_id in enumerate(
        question_ids
    ):  # each example has a unique question id
        features_per_example[question_id].append(feature_id)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_predictions_list = []
    all_nbest_json = collections.OrderedDict()

    # Let's loop over all the examples!
    for question_id, feature_indices in features_per_example.items():
        # print("question_id", question_id)
        # print("feature_indices", feature_indices)
        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # Looping through all the features associated to the current example.
            # We grab the predictions of the model for this feature.
            feature_start_logits = start_logits[feature_index].numpy()
            feature_end_logits = end_logits[feature_index].numpy()

            # print("start_logits", feature_start_logits.shape)
            # print("end_logits", feature_end_logits.shape)

            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            feature_word_ids = word_ids[feature_index]
            feature_sequence_ids = sequence_ids[feature_index]

            # print("subword_idx2word_idx", feature_word_ids)
            # print("sequence_ids", sequence_ids)

            num_question_tokens = 0
            while feature_sequence_ids[num_question_tokens] != 1:
                num_question_tokens += 1
            # print("num_question_tokens", num_question_tokens)

            feature_null_score = feature_start_logits[0] + feature_end_logits[0]
            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score
            ):
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": feature_start_logits[0],
                    "end_logit": feature_end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(feature_start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(feature_end_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            # print("start_indexes", start_indexes)
            # print("end_indexes", end_indexes)

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index
                        < num_question_tokens  ## because we do not want to consider answers that are
                        or end_index < num_question_tokens  ## part of the question.
                        or start_index >= len(feature_word_ids)
                        or end_index >= len(feature_word_ids)
                        or feature_word_ids[start_index] is None
                        or feature_word_ids[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    prelim_predictions.append(
                        {
                            "word_ids": (
                                feature_word_ids[start_index],
                                feature_word_ids[end_index],
                            ),
                            "score": feature_start_logits[start_index]
                            + feature_end_logits[end_index],
                            "start_logit": feature_start_logits[start_index],
                            "end_logit": feature_end_logits[end_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        # print("predictions", predictions)
        # Use the offsets to gather the answer text in the original context.
        first_feature_id = features_per_example[question_id][0]
        context = words[first_feature_id]

        for pred in predictions:
            offsets = pred.pop("word_ids")
            # print("context", context, context[offsets[0] : offsets[1] + 1])
            pred["text"] = " ".join(
                [x.strip() for x in context[offsets[0] : offsets[1] + 1]]
            )

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):
            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        all_predictions[question_ids[feature_index]] = predictions[0]["text"]
        all_predictions_list.append(
            {
                "questionId": question_ids[feature_index],
                "answer": predictions[0]["text"],
            }
        )
        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[question_ids[feature_index]] = [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.float16, np.float32, np.float64))
                    else v
                )
                for k, v in pred.items()
            }
            for pred in predictions
        ]

    with open("results.json", "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    return all_predictions, all_predictions_list


def anls_metric_str(
    predictions: List[List[str]], gold_labels: List[List[str]], tau=0.5, rank=0
):
    res = []
    """
    predictions: List[List[int]]
    gold_labels: List[List[List[int]]]: each instances probably have multiple gold labels.
    """
    for i, (preds, golds) in enumerate(zip(predictions, gold_labels)):
        max_s = 0
        for pred in preds:
            for gold in golds:
                dis = td.levenshtein.distance(pred.lower(), gold.lower())
                print(pred, gold, dis)
                max_len = max(len(pred), len(gold))
                if max_len == 0:
                    s = 0
                else:
                    nl = dis / max_len
                    s = 1 - nl if nl < tau else 0
                max_s = max(s, max_s)
        res.append(max_s)
    return res, sum(res) / len(res)


def sequence_anls(
    output_transform: Callable, device: Union[str, torch.device], threshold: float = 0.5
) -> Metric:
    def wrap(
        words,
        word_ids,
        sequence_ids,
        question_ids,
        start_logits,
        end_logits,
        answers,
    ):  # reversed targets and preds
        all_predictions, all_predictions_list = postprocess_qa_predictions(
            words=words,
            word_ids=word_ids.detach().cpu(),
            sequence_ids=sequence_ids.detach().cpu(),
            question_ids=question_ids.detach().cpu(),
            start_logits=start_logits.detach().cpu(),
            end_logits=end_logits.detach().cpu(),
        )
        logger.info(f"Total predictions: {len(all_predictions)}")

        # get all associated answers per question
        true_answers_per_example = collections.defaultdict(list)
        question_ids = convert_to_list(question_ids)
        for question_id, answer in zip(question_ids, answers):
            true_answers_per_example[question_id].append(answer)
        true_answers_per_example = {
            k: v[0] for k, v in true_answers_per_example.items()
        }
        true_answers_per_example = list(true_answers_per_example.values())

        all_pred_answers = [prediction["answer"] for prediction in all_predictions_list]
        logger.info(f"prediction: {all_pred_answers[:20]}")
        logger.info(f"gold_answers: {true_answers_per_example[:20]}")
        anls_scores = [
            anls_score(pred, target, threshold=threshold)
            for pred, target in zip(all_pred_answers, true_answers_per_example)
        ]
        anls = sum(anls_scores) / len(anls_scores)
        return anls

    return EpochDictMetric(wrap, output_transform=output_transform, device=device)
