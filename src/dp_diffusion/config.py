from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

# register dp training engine step
AtriaModuleRegistry.register_engine_step(
    module=".".join((_get_parent_module(__name__), f"engines.dp_training_step")),
    registered_class_or_func=[
        "DPTrainingStep",
    ],
    name=[
        "dp_training_step",
    ],
)

# register training engine with gan steps
AtriaModuleRegistry.register_engine(
    module=".".join((_get_parent_module(__name__), f"engines.dp_training")),
    registered_class_or_func="DPTrainingEngine",
    name="dp_training_engine",
    hydra_defaults=[
        "_self_",
        {"/engine_step@engine_step": "dp_training_step"},
    ],
)

# register training engine with gan steps
AtriaModuleRegistry.register_engine(
    module=".".join((_get_parent_module(__name__), f"engines.dp_promise_training")),
    registered_class_or_func="DPPromiseTrainingEngine",
    name="dp_promise_training_engine",
    hydra_defaults=[
        "_self_",
        {"/engine_step@engine_step": "dp_training_step"},
    ],
)

# register the diffusion test engine useful for testing over multiple sampling configs such as guidance scales
AtriaModuleRegistry.register_engine(
    module=".".join((_get_parent_module(__name__), f"engines.diffusion_test")),
    registered_class_or_func="DiffusionTestEngine",
    name="diffusion_test_engine",
    hydra_defaults=[
        "_self_",
        {"/engine_step@engine_step": "default_test_step"},
        {"/metric@metrics.fid_score": "fid_score"},
    ],
)

TRAINER_DIR = "${output_dir}/dp_promise_trainer/"
TRAINER_DIR += "${resolve_dir_name:${data_module.dataset_name}}/"
TRAINER_DIR += (
    "${resolve_dir_name:${task_module._target_}}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
)

AtriaModuleRegistry.register_task_runner(
    module=".".join(
        (_get_parent_module(__name__), f"task_runners._dp_promise_trainer")
    ),
    registered_class_or_func="DPPromiseTrainer",
    name="dp_promise_trainer",
    hydra_defaults=[
        "_self_",
        {"/data_module@data_module": "huggingface"},
        {"/task_module@task_module": "diffusion"},
        {"/engine@non_private_training_engine": "default_training_engine"},
        {"/engine@private_training_engine": "dp_promise_training_engine"},
        {"/engine@validation_engine": "default_validation_engine"},
        {"/engine@visualization_engine": "default_visualization_engine"},
        {"/engine@test_engine": "default_test_engine"},
        {"/optimizer@non_private_training_engine.optimizers": "adam"},
        {
            "/lr_scheduler@non_private_training_engine.lr_schedulers": "cosine_annealing_lr"
        },
        {"/optimizer@private_training_engine.optimizers": "adam"},
        {"/lr_scheduler@private_training_engine.lr_schedulers": None},
    ],
    zen_meta=dict(
        hydra={
            "run": {"dir": TRAINER_DIR},
            "output_subdir": "hydra",
            "job": {"chdir": False},
        },
    ),
)

AtriaModuleRegistry.register_task_module(
    module=".".join(
        (
            _get_parent_module(__name__),
            "models.task_modules.layout_conditioned_diffusion",
        )
    ),
    registered_class_or_func="LayoutConditionedDiffusionModule",
    name="layout_conditioned_diffusion",
)

AtriaModuleRegistry.register_task_module(
    module=".".join(
        (
            _get_parent_module(__name__),
            "models.task_modules.layout_conditioned_latent_diffusion",
        )
    ),
    registered_class_or_func="LayoutConditionedLatentDiffusionModule",
    name="layout_conditioned_latent_diffusion_diffusers_vae",
    hydra_defaults=[
        "_self_",
        {
            "/torch_model_builder@torch_model_builder.reverse_diffusion_model": "diffusers"
        },
        {"/torch_model_builder@torch_model_builder.vae": "diffusers"},
    ],
)

AtriaModuleRegistry.register_task_module(
    module=".".join(
        (
            _get_parent_module(__name__),
            "models.task_modules.layout_conditioned_latent_diffusion",
        )
    ),
    registered_class_or_func="LayoutConditionedLatentDiffusionModule",
    name="layout_conditioned_latent_diffusion_local_vae",
    hydra_defaults=[
        "_self_",
        {
            "/torch_model_builder@torch_model_builder.reverse_diffusion_model": "diffusers"
        },
        {"/torch_model_builder@torch_model_builder.vae": "local"},
    ],
)

AtriaModuleRegistry.register_data_transform(
    module=".".join((_get_parent_module(__name__), "transforms.hocr_to_layout_mask")),
    registered_class_or_func="HocrToLayoutMask",
    name="hocr_to_layout_mask",
)


AtriaModuleRegistry.register_data_transform(
    module=".".join(
        (_get_parent_module(__name__), "transforms.ccpdf_preprocess_transform")
    ),
    registered_class_or_func="CCPdfPreprocessTransform",
    name="ccpdf_preprocess_transform",
)
