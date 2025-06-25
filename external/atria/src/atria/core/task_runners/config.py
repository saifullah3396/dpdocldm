from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

# register data modules as base node, types=[huggingface, torch]
AtriaModuleRegistry.register_task_runner(
    module=".".join((_get_parent_module(__name__), f"atria_data_processor")),
    registered_class_or_func="AtriaDataProcessor",
    hydra_defaults=[
        "_self_",
        {"/data_module@data_module": "huggingface"},
    ],
    zen_meta=dict(
        hydra={
            "run": {"dir": "/tmp/atria_data_processor"},
            "output_subdir": "hydra",
            "job": {"chdir": False},
        },
    ),
)

MODEL_EVALUATOR_DIR = "${output_dir}/atria_model_evaluator/"
MODEL_EVALUATOR_DIR += "${resolve_dir_name:${data_module.dataset_name}}/"
MODEL_EVALUATOR_DIR += (
    "${resolve_dir_name:${task_module._target_}}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
)

AtriaModuleRegistry.register_task_runner(
    module=".".join((_get_parent_module(__name__), f"atria_model_evaluator")),
    registered_class_or_func="AtriaModelEvaluator",
    hydra_defaults=[
        "_self_",
        {"/data_module@data_module": "huggingface"},
        {"/task_module@task_module": "image_classification"},
        {"/engine@test_engine": "default_test_engine"},
    ],
    zen_meta=dict(
        hydra={
            "run": {"dir": MODEL_EVALUATOR_DIR},
            "output_subdir": "hydra",
            "job": {"chdir": False},
        },
    ),
)

TRAINER_DIR = "${output_dir}/atria_trainer/"
TRAINER_DIR += "${resolve_dir_name:${data_module.dataset_name}}/"
TRAINER_DIR += (
    "${resolve_dir_name:${task_module._target_}}/${now:%Y-%m-%d}/${now:%H-%M-%S}"
)

AtriaModuleRegistry.register_task_runner(
    module=".".join((_get_parent_module(__name__), f"_atria_trainer")),
    registered_class_or_func="AtriaTrainer",
    hydra_defaults=[
        "_self_",
        {"/data_module@data_module": "huggingface"},
        {"/task_module@task_module": "image_classification"},
        {"/engine@training_engine": "default_training_engine"},
        {"/engine@validation_engine": "default_validation_engine"},
        {"/engine@visualization_engine": "default_visualization_engine"},
        {"/engine@test_engine": "default_test_engine"},
        {"/optimizer@training_engine.optimizers": "adam"},
        {"/lr_scheduler@training_engine.lr_schedulers": "cosine_annealing_lr"},
    ],
    zen_meta=dict(
        hydra={
            "run": {"dir": TRAINER_DIR},
            "output_subdir": "hydra",
            "job": {"chdir": False},
        },
    ),
)
