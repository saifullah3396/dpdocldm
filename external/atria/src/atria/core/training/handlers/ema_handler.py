import warnings
from typing import Union

import torch
from ignite.engine import CallableEventWithFilter, Engine, Events, EventsList
from ignite.handlers import EMAHandler
from ignite.handlers.state_param_scheduler import LambdaStateScheduler


class AtriaEMAHandler(EMAHandler):
    def attach(
        self,
        engine: Engine,
        name: str = "ema_momentum",
        warn_if_exists: bool = True,
        event: Union[
            str, Events, CallableEventWithFilter, EventsList
        ] = Events.ITERATION_COMPLETED,
    ) -> None:
        """Attach the handler to engine. After the handler is attached, the ``Engine.state`` will add an new attribute
        with name ``name`` if the attribute does not exist. Then, the current momentum can be retrieved from
        ``Engine.state`` when the engine runs.


        Note:
            There are two cases where a momentum with name ``name`` already exists: 1. the engine has loaded its
            state dict after resuming. In this case, there is no need to initialize the momentum again, and users
            can set ``warn_if_exists`` to False to suppress the warning message; 2. another handler has created
            a state attribute with the same name. In this case, users should choose another name for the ema momentum.


        Args:
            engine: trainer to which the handler will be attached.
            name: attribute name for retrieving EMA momentum from ``Engine.state``. It should be a unique name since a
                trainer can have multiple EMA handlers.
            warn_if_exists: if True, a warning will be thrown if the momentum with name ``name`` already exists.
            event: event when the EMA momentum and EMA model are updated.

        """
        if hasattr(engine.state, name):
            if warn_if_exists:
                warnings.warn(
                    f"Attribute '{name}' already exists in Engine.state. It might because 1. the engine has loaded its "
                    f"state dict or 2. {name} is already created by other handlers. Turn off this warning by setting"
                    f"warn_if_exists to False.",
                    category=UserWarning,
                )
        else:
            setattr(engine.state, name, self.momentum)

        if self._momentum_lambda_obj is not None:
            self.momentum_scheduler = LambdaStateScheduler(
                self._momentum_lambda_obj, param_name=name
            )

            # first update the momentum and then update the EMA model
            self.momentum_scheduler.attach(engine, event)
        engine.add_event_handler(event, self._update_ema_model, name)

    @torch.no_grad()
    def swap_params(self) -> None:
        for ema_v, model_v in zip(
            self.ema_model.state_dict().values(), self.model.state_dict().values()
        ):
            tmp = model_v.data.clone()
            model_v.data.copy_(ema_v.data)
            ema_v.data.copy_(tmp)
