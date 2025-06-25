from typing import List, Optional, Type

import hydra_zen
import torch
from atria.core.models.tasks import ModelTasks
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.utilities.logging import get_logger
from atria.core.utilities.nn_modules import (
    _find_layer_in_model,
    _freeze_layers,
    _freeze_layers_by_name,
)
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
)

logger = get_logger(__name__)


class TransformersModelBuilder(TorchModelBuilderBase):
    """
    Constructs a transformer model for various tasks such as sequence classification,
    token classification, image classification, and layout token classification.
    """

    def __init__(
        self,
        model_name: str = hydra_zen.MISSING,
        model_task: ModelTasks = ModelTasks.sequence_classification,
        model_cache_dir: Optional[str] = None,
        pretrained: bool = True,
        convert_bn_to_gn: bool = False,
        is_frozen: bool = False,
        required_in_training_step: bool = True,
        pretrained_checkpoint: Optional[str] = None,
        strict: bool = True,
        frozen_keys_patterns: Optional[List[str]] = None,
        unfrozen_keys_patterns: Optional[List[str]] = None,
        n_frozen_encoder_layers: int = 0,
        encoder_layer_name: Optional[str] = None,
        frozen_layers: Optional[List[str]] = None,
        **model_config,
    ):
        """
        Initialize the TransformersModelBuilder with the given parameters.

        Args:
            n_frozen_encoder_layers (int): Number of frozen layers in the encoder.
            encoder_layer_name (Optional[str]): Name of the encoder layer to freeze.
            frozen_layers (Optional[List[str]]): List of layer names to freeze.
            **kwargs: Additional keyword arguments for the base class.
        """
        super().__init__(
            model_name=model_name,
            model_task=model_task,
            model_cache_dir=model_cache_dir,
            pretrained=pretrained,
            convert_bn_to_gn=convert_bn_to_gn,
            is_frozen=is_frozen,
            required_in_training_step=required_in_training_step,
            pretrained_checkpoint=pretrained_checkpoint,
            strict=strict,
            frozen_keys_patterns=frozen_keys_patterns,
            unfrozen_keys_patterns=unfrozen_keys_patterns,
            **model_config,
        )
        self.n_frozen_encoder_layers = n_frozen_encoder_layers
        self.encoder_layer_name = encoder_layer_name
        self.frozen_layers = frozen_layers if frozen_layers is not None else []

        if self.n_frozen_encoder_layers > 0:
            assert (
                self.encoder_layer_name is not None
            ), "Please provide the encoder_layer_name to unfreeze last N layers of."

    def _build(self) -> PreTrainedModel:
        """
        Initializes the transformer model based on the specified task and configuration.

        Returns:
            PreTrainedModel: The initialized transformer model.
        """
        supported_tasks = [
            ModelTasks.sequence_classification,
            ModelTasks.token_classification,
            ModelTasks.image_classification,
            ModelTasks.layout_token_classification,
            ModelTasks.visual_question_answering,
            ModelTasks.question_answering,
        ]
        assert (
            self._model_task in supported_tasks
        ), f"Task {self._model_task} not supported for {self.__class__.__name__}."

        initializer_class: Type[PreTrainedModel]

        if self._model_task == ModelTasks.sequence_classification:
            initializer_class = AutoModelForSequenceClassification
        elif self._model_task in [
            ModelTasks.token_classification,
            ModelTasks.layout_token_classification,
        ]:
            initializer_class = AutoModelForTokenClassification
        elif self._model_task == ModelTasks.image_classification:
            initializer_class = AutoModelForImageClassification
        elif self._model_task in [
            ModelTasks.visual_question_answering,
            ModelTasks.question_answering,
        ]:
            initializer_class = AutoModelForQuestionAnswering
        else:
            raise ValueError(f"Task {self._model_task} not supported.")

        model_config = {**self._model_config}

        if initializer_class in [
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
        ]:
            assert (
                "num_labels" in model_config
            ), "num_labels must be provided for (AutoModelForSequenceClassification, AutoModelForTokenClassification)."
            model_config["return_dict"] = True
        elif initializer_class == AutoModelForImageClassification:
            num_labels = model_config.pop("num_labels", None)

        hf_config: PretrainedConfig
        if self._pretrained:
            hf_config = AutoConfig.from_pretrained(
                self._model_name,
                cache_dir=self._model_cache_dir,
                **model_config,
            )

            logger.debug(
                f"Initializing the model with the following config:\n {hf_config}"
            )
            model = initializer_class.from_pretrained(
                self._model_name,
                config=hf_config,
                cache_dir=self._model_cache_dir,
            )
        else:
            hf_config = AutoConfig(
                self._model_name,
                cache_dir=self._model_cache_dir,
                **model_config,
            )
            model = initializer_class(
                self._model_name,
                config=hf_config,
                cache_dir=self._model_cache_dir,
            )

        if (
            initializer_class == AutoModelForImageClassification
            and num_labels is not None
        ):
            model.classifier = torch.nn.Linear(
                model.classifier.in_features,
                num_labels,
            )
            logger.info(
                f"Initializing the classifier head for image classification with {model.classifier}"
            )

        if self.n_frozen_encoder_layers > 0:
            encoder_layer = _find_layer_in_model(model, self.encoder_layer_name)
            _freeze_layers(encoder_layer[: self.n_frozen_encoder_layers])

        if self.frozen_layers:
            _freeze_layers_by_name(model, self.frozen_layers)

        return model
