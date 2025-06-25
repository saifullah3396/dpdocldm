from hydra_zen import builds

from atria.core.data.batch_samplers import BatchSamplersDict
from atria.core.data.data_transforms import DataTransformsDict
from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

DEFAULT_DATA_MODULE_KWARGS = dict(
    runtime_data_transforms=builds(DataTransformsDict, populate_full_signature=True),
    batch_samplers=builds(BatchSamplersDict, populate_full_signature=True),
    hydra_defaults=[
        "_self_",
        {
            "/dataset_cacher@dataset_cacher": "msgpack"
        },  # this sets dataset_cacher to /dataset_cacher/msgpack_dataset_cacher
        {
            "/dataloader_builder@train_dataloader_builder": "torch"
        },  # this sets train_dataloader_builder to /data_loader_builder/torch
        {
            "/dataloader_builder@evaluation_dataloader_builder": "torch"
        },  # this sets evaluation_dataloader_builder to /data_loader_builder/torch
        {"/train_validation_splitter@train_validation_splitter": None},
    ],
)
DEFAULT_STREAMABLE_DATA_MODULE_KWARGS = dict(
    streaming_mode=True,
    runtime_data_transforms=builds(DataTransformsDict, populate_full_signature=True),
    batch_samplers=builds(BatchSamplersDict, populate_full_signature=True),
    hydra_defaults=[
        "_self_",
        {
            "/dataset_cacher@dataset_cacher": "webdataset"
        },  # this sets dataset_cacher to /dataset_cacher/msgpack_dataset_cacher
        {
            "/dataloader_builder@train_dataloader_builder": "webdataset"
        },  # this sets train_dataloader_builder to /data_loader_builder/torch
        {
            "/dataloader_builder@evaluation_dataloader_builder": "webdataset"
        },  # this sets evaluation_dataloader_builder to /data_loader_builder/torch
        {"/train_validation_splitter@train_validation_splitter": None},
    ],
)

# register data_module, types=[huggingface, huggingface_streamable, torch, torch_streamable]
for name, kwargs in zip(
    ["huggingface", "huggingface_streamable"],
    [DEFAULT_DATA_MODULE_KWARGS, DEFAULT_STREAMABLE_DATA_MODULE_KWARGS],
):
    AtriaModuleRegistry.register_data_module(
        module=_get_parent_module(__name__) + ".data_modules.huggingface_data_module",
        name=name,
        registered_class_or_func="HuggingfaceDataModule",
        **kwargs,
    )
for name, kwargs in zip(
    ["torch", "torch_streamable"],
    [DEFAULT_DATA_MODULE_KWARGS, DEFAULT_STREAMABLE_DATA_MODULE_KWARGS],
):
    AtriaModuleRegistry.register_data_module(
        module=_get_parent_module(__name__) + ".data_modules.torch_data_module",
        name=name,
        registered_class_or_func="TorchDataModule",
        **kwargs,
    )

# register data_module/data_cacher, data_cacher is a child node of data_module, types=[webdataset, msgpack]
for cache_type in ["webdataset", "msgpack"]:
    AtriaModuleRegistry.register_dataset_cacher(
        module=_get_parent_module(__name__)
        + ".data_modules.dataset_cacher.dataset_cacher",
        name=cache_type,
        registered_class_or_func="DatasetCacher",
        cache_type=cache_type,
        preprocess_data_transforms=builds(
            DataTransformsDict, populate_full_signature=True
        ),
    )

# register data_module/train_val_samplers, types=[train_val_sampler]
AtriaModuleRegistry.register_train_validation_splitter(
    module=_get_parent_module(__name__) + ".train_validation_splitter",
    name="default",
    registered_class_or_func="DefaultTrainValidationSplitter",
)

# register data_module/dataloader_builder, types=[torch, webdataset]
DEFAULT_DATA_LOADER_BUILDER_KWARGS = dict(
    zen_partial=True,
    batch_size=64,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    collate_fn=None,
    hydra_defaults=[
        "_self_",
        {"/data_collator@collate_fn": "batch_to_tensor"},
    ],
)
AtriaModuleRegistry.register_dataloader_builder(
    module=_get_parent_module(__name__) + ".data_modules.utilities",
    name="torch",
    registered_class_or_func="auto_dataloader",
    **DEFAULT_DATA_LOADER_BUILDER_KWARGS,
)
AtriaModuleRegistry.register_dataloader_builder(
    module="webdataset",
    name="webdataset",
    registered_class_or_func="WebLoader",
    **DEFAULT_DATA_LOADER_BUILDER_KWARGS,
)

# register collate_fn, collate_fns are child nodes of dataloader_builders types=[simple, batch_to_tensor]
AtriaModuleRegistry.register_data_collator(
    module=_get_parent_module(__name__) + ".data_collators",
    name="simple_batch",
    registered_class_or_func="SimpleBatchDataCollator",
)
AtriaModuleRegistry.register_data_collator(
    module=_get_parent_module(__name__) + ".data_collators",
    name="batch_to_tensor",
    registered_class_or_func="BatchToTensorDataCollator",
)
