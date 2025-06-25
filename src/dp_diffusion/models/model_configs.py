from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

AtriaModuleRegistry.register_model_config(
    module=_get_parent_module(__name__) + ".openai_models.dp_promise_unet.models",
    name="dp_promise_openai_unet_model_32x32",
    registered_class_or_func="UNetModel",
    image_size=32,
    in_channels=4,
    model_channels=128,
    out_channels=4,
    num_res_blocks=2,
    attention_resolutions=[2],  # maps to attention resolutions 16
    dropout=0.0,
    channel_mult=[
        1,
        2,
        2,
        2,
    ],
)

AtriaModuleRegistry.register_model_config(
    module=_get_parent_module(__name__) + ".openai_models.dp_promise_unet.models",
    name="dp_promise_openai_unet_model_64x64",
    registered_class_or_func="UNetModel",
    image_size=64,
    in_channels=3,
    model_channels=128,
    out_channels=3,
    num_res_blocks=2,
    attention_resolutions=[4, 8],  # maps to attention resolutions 16,8
    dropout=0.0,
    channel_mult=[
        1,
        2,
        3,
        4,
    ],
)
