""" PyTorch module that defines the base model for training/testing etc. """

from __future__ import annotations

from atria.core.models.task_modules.atria_task_module import AtriaTaskModule


class HuggingfaceTaskModule(AtriaTaskModule):
    _SUPPORTS_MULTIPLE_BUILDERS = False
    _SUPPORTED_BUILDERS = ["TransformersModelBuilder"]
