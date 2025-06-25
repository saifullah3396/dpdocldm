from anls import anls_score
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


class ANLS(Metric):
    def __init__(
        self, threshold: float = 0.5, output_transform=lambda x: x, device="cpu"
    ):
        self.threshold = threshold
        self._sum_anls = None
        self._num_examples = None
        super(ANLS, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._sum_anls = 0
        self._num_examples = 0
        super(ANLS, self).reset()

    @reinit__is_reduced
    def update(self, output):
        pred_answers_batch, target_answers_batch = output[0], output[1]

        # compute anls scores of answeres
        for prediction, gold_labels in zip(pred_answers_batch, target_answers_batch):
            print(prediction, gold_labels)
            anls = anls_score(
                prediction=prediction,
                gold_labels=gold_labels,  # this takes a list of targets
                threshold=self.threshold,
            )
            self._sum_anls += anls
        self._num_examples += len(pred_answers_batch)

    @sync_all_reduce("_num_examples", "_sum_anls:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "ANLS must have at least one example before it can be computed."
            )
        return self._sum_anls / self._num_examples
