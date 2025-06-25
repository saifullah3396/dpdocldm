from atria.core.registry.module_registry import AtriaModuleRegistry


AtriaModuleRegistry.register_batch_sampler(
    module=f"atria.data.batch_samplers.group_batch_sampler",
    registered_class_or_func="GroupBatchSampler",
)
AtriaModuleRegistry.register_batch_sampler(
    module=f"atria.data.batch_samplers.ar_group_batch_sampler",
    registered_class_or_func="AspectRatioGroupBatchSampler",
)
