import dataclasses
from typing import Dict, List, Optional, Union

import torch
from atria.core.constants import DataKeys
from atria.core.data.datasets.dataset_metadata import DatasetMetadata
from atria.core.models.model_outputs import ModelOutput
from atria.core.models.task_modules.atria_task_module import (
    AtriaTaskModule,
    CheckpointConfig,
)
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict
from ignite.engine import Engine
from ignite.handlers import TensorboardLogger

logger = get_logger(__name__)


@dataclasses.dataclass
class ObjectDetectionModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None


class ObjectDetectionModule(AtriaTaskModule):
    _REQUIRES_BUILDER_DICT = False
    _SUPPORTED_BUILDERS = [
        "Detectron2ModelBuilder",
    ]

    def __init__(
        self,
        torch_model_builder: Union[
            TorchModelBuilderBase, Dict[str, TorchModelBuilderBase]
        ],
        checkpoint_configs: Optional[List[CheckpointConfig]] = None,
        dataset_metadata: Optional[DatasetMetadata] = None,
        tb_logger: Optional[TensorboardLogger] = None,
    ):
        super().__init__(
            torch_model_builder=torch_model_builder,
            checkpoint_configs=checkpoint_configs,
            dataset_metadata=dataset_metadata,
            tb_logger=tb_logger,
        )

    def required_keys_in_batch(self, stage: TrainingStage) -> torch.List[str]:
        required_keys = [
            DataKeys.IMAGE_HEIGHT,
            DataKeys.IMAGE_WIDTH,
            DataKeys.IMAGE,
        ]
        if stage == TrainingStage.train:
            return required_keys
        elif stage == TrainingStage.validation:
            return required_keys
        elif stage == TrainingStage.test:
            return required_keys
        elif stage == TrainingStage.predict:
            return required_keys
        else:
            raise ValueError(f"Stage {stage} not supported")

    def _model_forward(self, batch: BatchDict) -> dict:
        return self._torch_model(batch)

    def training_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        assert self._torch_model.training
        losses = self._model_forward(batch)
        return ObjectDetectionModelOutput(loss=sum(losses.values()), **losses)

    def evaluation_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        assert not self._torch_model.training
        outputs = self._model_forward(batch)
        outputs = {k: [dic[k] for dic in outputs] for k in outputs[0]}
        return outputs

    def predict_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        assert not self._torch_model.training
        outputs = self._model_forward(batch)
        outputs = {k: [dic[k] for dic in outputs] for k in outputs[0]}
        return outputs

    def visualization_step(
        self,
        batch: BatchDict,
        training_engine: Engine,
        **kwargs,
    ) -> None:
        import math

        import ignite.distributed as idist
        import torchvision

        assert not self._torch_model.training

        # detectr2on ensures the bbox always remap to original image size,
        # but we don't want that during visualization so we manually update it to current image size
        for sample in batch:
            sample[DataKeys.IMAGE_HEIGHT] = sample[DataKeys.IMAGE].shape[2]
            sample[DataKeys.IMAGE_WIDTH] = sample[DataKeys.IMAGE].shape[3]

        outputs = self._model_forward(batch)

        if idist.get_rank() == 0:
            import torch
            from detectron2.utils.visualizer import ColorMode, Visualizer
            from torchvision.transforms.functional import resize

            image_batch = []
            for sample, output in zip(batch, outputs):
                image = sample[DataKeys.IMAGE].detach().cpu().permute(1, 2, 0).numpy()
                v = Visualizer(image, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
                image_output = v.draw_instance_predictions(
                    output["instances"].to("cpu")
                ).get_image()
                image_batch.append(
                    resize(
                        torch.from_numpy(image_output.transpose(2, 0, 1)), (512, 512)
                    )
                )

            logger.info(
                f"Saving image batch {training_engine.state.iteration} to tensorboard."
            )
            # step
            global_step = (
                training_engine.state.iteration if training_engine is not None else 1
            )
            batch_step = training_engine.state.iteration
            # save a single image to tensorboard
            num_samples = len(image_batch)
            self._tb_logger.writer.add_image(
                f"visualization/pred_instances_{batch_step}",  # engine .iteration refers to the batch id
                torchvision.utils.make_grid(
                    image_batch, nrow=int(math.sqrt(num_samples))
                ),
                global_step,
            )
