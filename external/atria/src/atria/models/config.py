from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

# register task modules type=[HuggingfaceTaskModule]
AtriaModuleRegistry.register_task_module(
    module=_get_parent_module(__name__) + ".task_modules.huggingface",
    registered_class_or_func="HuggingfaceTaskModule",
    name="huggingface",
    hydra_defaults=[
        "_self_",
        {"/torch_model_builder@torch_model_builder": "transformers"},
    ],
)

# register classification task modules type=[ImageClassificationModule, SequenceClassificationModule, TokenClassificationModule, TokenLayoutClassificationModule]
AtriaModuleRegistry.register_task_module(
    module=_get_parent_module(__name__) + ".task_modules.classification.image",
    registered_class_or_func="ImageClassificationModule",
    name="image_classification",
    hydra_defaults=[
        "_self_",
        {"/torch_model_builder@torch_model_builder": "timm"},
    ],
)
AtriaModuleRegistry.register_task_module(
    module=_get_parent_module(__name__) + ".task_modules.classification.sequence",
    registered_class_or_func="SequenceClassificationModule",
    name="sequence_classification",
    hydra_defaults=[
        "_self_",
        {"/torch_model_builder@torch_model_builder": "transformers"},
    ],
)
AtriaModuleRegistry.register_task_module(
    module=_get_parent_module(__name__) + ".task_modules.classification.token",
    registered_class_or_func="TokenClassificationModule",
    name="token_classification",
    hydra_defaults=[
        "_self_",
        {"/torch_model_builder@torch_model_builder": "transformers"},
    ],
)
AtriaModuleRegistry.register_task_module(
    module=_get_parent_module(__name__) + ".task_modules.classification.layout_token",
    registered_class_or_func="LayoutTokenClassificationModule",
    name="layout_token_classification",
    hydra_defaults=[
        "_self_",
        {"/torch_model_builder@torch_model_builder": "transformers"},
    ],
)

# register detection task modules type=[ObjectDetectionModule]
AtriaModuleRegistry.register_task_module(
    module=_get_parent_module(__name__) + ".task_modules.detection.image",
    name="image_object_detection",
    registered_class_or_func="ImageObjectDetectionModule",
    hydra_defaults=[
        "_self_",
        {"/torch_model_builder@torch_model_builder": "transformers"},
    ],
)

# register question answering task modules type=[QuestionAnsweringModule]
AtriaModuleRegistry.register_task_module(
    module=_get_parent_module(__name__) + ".task_modules.qa.base",
    name="question_answering",
    registered_class_or_func="QuestionAnsweringModule",
    hydra_defaults=[
        "_self_",
        {"/torch_model_builder@torch_model_builder": "transformers"},
    ],
)

# register autoencoding task modules type=[ImageAutoEncodingModule]
AtriaModuleRegistry.register_task_module(
    module=_get_parent_module(__name__) + ".task_modules.autoencoding.image",
    name="image_autoencoding",
    registered_class_or_func="ImageAutoEncodingModule",
    hydra_defaults=[
        "_self_",
        {"/torch_model_builder@torch_model_builder": "local"},
    ],
)


# register autoencoding task modules type=[ImageVarAutoEncodingModule]
AtriaModuleRegistry.register_task_module(
    module=_get_parent_module(__name__) + ".task_modules.var_autoencoding.image",
    name="image_var_autoencoding",
    registered_class_or_func="ImageVarAutoEncodingModule",
    hydra_defaults=[
        "_self_",
        {"/torch_model_builder@torch_model_builder": "diffusers"},
    ],
)

# register autoencoding task modules type=[ImageVarAutoEncodingGANModule]
AtriaModuleRegistry.register_task_module(
    module=_get_parent_module(__name__) + ".task_modules.var_autoencoding.image_gan",
    name="image_var_autoencoding_gan",
    registered_class_or_func="ImageVarAutoEncodingGANModule",
    hydra_defaults=[
        "_self_",
        {"/torch_model_builder@torch_model_builder": "diffusers"},
    ],
)

# register diffusers task modules type=[DiffusionModule]
AtriaModuleRegistry.register_task_module(
    module=_get_parent_module(__name__) + ".task_modules.diffusion.diffusion",
    name="diffusion",
    registered_class_or_func="DiffusionModule",
    hydra_defaults=[
        "_self_",
        {
            "/torch_model_builder@torch_model_builder.reverse_diffusion_model": "diffusers"
        },
    ],
)

# register diffusers task modules type=[LatentDiffusionModule]
AtriaModuleRegistry.register_task_module(
    module=_get_parent_module(__name__) + ".task_modules.diffusion.latent_diffusion",
    name="latent_diffusion",
    registered_class_or_func="LatentDiffusionModule",
    hydra_defaults=[
        "_self_",
        {
            "/torch_model_builder@torch_model_builder.reverse_diffusion_model": "diffusers"
        },
        {"/torch_model_builder@torch_model_builder.vae": "diffusers"},
    ],
)
