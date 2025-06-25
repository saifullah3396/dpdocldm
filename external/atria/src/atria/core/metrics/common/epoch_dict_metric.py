import itertools
import warnings
from typing import Callable, List, Mapping, Optional, Union, cast

import ignite.distributed as idist
import torch
from ignite.metrics.metric import Metric, reinit__is_reduced


class EpochDictMetric(Metric):
    def __init__(
        self,
        compute_fn: Callable[[torch.Tensor, torch.Tensor], float],
        output_transform: Callable = lambda x: x,
        check_compute_fn: bool = True,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        if not callable(compute_fn):
            raise TypeError("Argument compute_fn should be callable.")

        self.compute_fn = compute_fn
        self._check_compute_fn = check_compute_fn

        super(EpochDictMetric, self).__init__(
            output_transform=output_transform, device=device
        )

    @reinit__is_reduced
    def reset(self) -> None:
        self._fn_inputs: List[List[torch.Tensor]] = []
        self._result: Optional[float] = None

    @reinit__is_reduced
    def update(self, output: Mapping[str, torch.Tensor]) -> None:
        output = [
            (
                value.detach().clone().to(self._device)
                if isinstance(value, torch.Tensor)
                else value
            )
            for value in output
        ]
        self._fn_inputs.append(output)

        # Check once the signature and execution of compute_fn
        if len(self._fn_inputs) == 1 and self._check_compute_fn:
            try:
                self.compute_fn(*self._fn_inputs[0])
            except Exception as e:
                warnings.warn(
                    f"Probably, there can be a problem with `compute_fn`:\n {e}.",
                    EpochDictMetricWarning,
                )

    def compute(self) -> float:
        if self._result is None:
            _gathered_fn_inputs = []
            for item_idx in range(len(self._fn_inputs[0])):
                gathered_fn_input = [
                    self._fn_inputs[batch_idx][item_idx]
                    for batch_idx in range(len(self._fn_inputs))
                ]
                if isinstance(gathered_fn_input[0], torch.Tensor):
                    if len(gathered_fn_input[0].shape) == 0:
                        gathered_fn_input = torch.tensor(gathered_fn_input)
                    else:
                        gathered_fn_input = torch.cat(gathered_fn_input, dim=0)
                elif isinstance(gathered_fn_input[0], list):
                    gathered_fn_input = list(
                        itertools.chain.from_iterable(gathered_fn_input)
                    )

                ws = idist.get_world_size()
                if ws > 1:
                    # All gather across all processes
                    gathered_fn_input = idist.all_gather(gathered_fn_input)
                _gathered_fn_inputs.append(gathered_fn_input)

            self._result = 0.0
            if idist.get_rank() == 0:
                # Run compute_fn on zero rank only
                self._result = self.compute_fn(*_gathered_fn_inputs)

            if ws > 1:
                # broadcast result to all processes
                self._result = cast(float, idist.broadcast(self._result, src=0))

            return self._result


class EpochDictMetricWarning(UserWarning):
    pass
