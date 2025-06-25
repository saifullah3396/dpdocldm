from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

# register AutoEncoderKL model from diffusers library
AtriaModuleRegistry.register_model_config(
    module="diffusers",
    name="diffusers/autoencoder_kl",
    registered_class_or_func="AutoencoderKL",
)

# register UNet2DModel model from diffusers library
AtriaModuleRegistry.register_model_config(
    module="diffusers",
    name="diffusers/unet_2d_model",
    registered_class_or_func="UNet2DModel",
)

# register UNet2DModel model from diffusers library
AtriaModuleRegistry.register_model_config(
    module="diffusers.models.unets.unet_2d_condition",
    name="diffusers/unet_2d_conditional_model",
    registered_class_or_func="UNet2DConditionModel",
)

# register CompvisAutoencoderKL model from compvis library
AtriaModuleRegistry.register_model_config(
    module=_get_parent_module(__name__) + ".autoencoding.compvis_vae",
    name="compvis_vae",
    registered_class_or_func="CompvisAutoencoderKL",
)
