# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Tuple

import torch
from ignite.engine import Engine
from ignite.metrics import Metric
from opacus import PrivacyEngine



class PrivacyLossMetric(Metric):
    def __init__(self, privacy_engine: PrivacyEngine, delta: float) -> None:
        super(PrivacyLossMetric, self).__init__()
        self._privacy_engine = privacy_engine
        self._delta = delta

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        try:
            epsilon = self._privacy_engine.accountant.get_epsilon(self._delta)
            engine.state.metrics["eps"] = epsilon
            engine.state.metrics["delta"] = self._delta
        except ValueError:
            pass

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]) -> None:
        return 0.0

    def compute(self) -> List[Dict[str, Any]]:
        return 0.0
