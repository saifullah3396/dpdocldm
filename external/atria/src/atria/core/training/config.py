
from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

# register evaluation engine steps
AtriaModuleRegistry.register_engine_step(
    module=".".join((_get_parent_module(__name__), f"engines.engine_steps.evaluation")),
    registered_class_or_func=[
        "ValidationStep",
        "VisualizationStep",
        "TestStep",
        "PredictionStep",
        "FeatureExtractorStep",
    ],
    name=[
        "default_validation_step",
        "default_visualization_step",
        "default_test_step",
        "default_prediction_step",
        "default_feature_extractor_step",
    ],
)

# register evaluation engines
for name, engine_type in zip(
    [
        "default_validation_engine",
        "default_visualization_engine",
        "default_test_engine",
        "default_prediction_engine",
        "default_feature_extractor_engine",
    ],
    [
        "Validation",
        "Visualization",
        "Test",
        "Prediction",
        "FeatureExtractor",
    ],
):
    AtriaModuleRegistry.register_engine(
        module=".".join((_get_parent_module(__name__), f"engines.evaluation")),
        registered_class_or_func=f"{engine_type}Engine",
        name=name,
        hydra_defaults=[
            "_self_",
            {"/engine_step@engine_step": f"{name.replace('_engine', '_step')}"},
        ],
    )

# register evaluation engine steps
AtriaModuleRegistry.register_engine_step(
    module=".".join((_get_parent_module(__name__), f"engines.engine_steps.training")),
    registered_class_or_func=[
        "TrainingStep",
    ],
    name=[
        "default_training_step",
    ],
)
AtriaModuleRegistry.register_engine_step(
    module=".".join((_get_parent_module(__name__), f"engines.engine_steps.training")),
    registered_class_or_func=[
        "GANTrainingStep",
    ],
    name=[
        "gan_training_step",
    ],
)

# register training engine steps
AtriaModuleRegistry.register_engine(
    module=".".join((_get_parent_module(__name__), f"engines.training")),
    registered_class_or_func="TrainingEngine",
    name="default_training_engine",
    hydra_defaults=[
        "_self_",
        {"/engine_step@engine_step": "default_training_step"},
    ],
)

# register training engine with gan steps
AtriaModuleRegistry.register_engine(
    module=".".join((_get_parent_module(__name__), f"engines.training")),
    registered_class_or_func="TrainingEngine",
    name="gan_training_engine",
    hydra_defaults=[
        "_self_",
        {"/engine_step@engine_step": "gan_training_step"},
    ],
    outputs_to_running_avg=[
        "gen_loss",
        "disc_loss",
    ],
)

# register more task specific engines for testing
for engine_type in ["Validation", "Test"]:
    AtriaModuleRegistry.register_engine(
        module=".".join((_get_parent_module(__name__), f"engines.evaluation")),
        registered_class_or_func=f"{engine_type}Engine",
        name=f"image_classification_{engine_type.lower()}_engine",
        hydra_defaults=[
            "_self_",
            {"/engine_step@engine_step": f"default_{engine_type.lower()}_step"},
            {"/metric@metrics.accuracy": "accuracy"},
            {"/metric@metrics.precision": "precision"},
            {"/metric@metrics.recall": "recall"},
            {"/metric@metrics.f1": "f1"},
        ],
    )

for engine_type in ["Validation", "Test"]:
    AtriaModuleRegistry.register_engine(
        module=".".join((_get_parent_module(__name__), f"engines.evaluation")),
        registered_class_or_func=f"{engine_type}Engine",
        name=f"sequence_classification_{engine_type.lower()}_engine",
        hydra_defaults=[
            "_self_",
            {"/engine_step@engine_step": f"default_{engine_type.lower()}_step"},
            {"/metric@metrics.accuracy": "accuracy"},
            {"/metric@metrics.precision": "precision"},
            {"/metric@metrics.recall": "recall"},
            {"/metric@metrics.f1": "f1"},
        ],
    )

for engine_type in ["Validation", "Test"]:
    AtriaModuleRegistry.register_engine(
        module=".".join((_get_parent_module(__name__), f"engines.evaluation")),
        registered_class_or_func=f"{engine_type}Engine",
        name=f"token_classification_{engine_type.lower()}_engine",
        hydra_defaults=[
            "_self_",
            {"/engine_step@engine_step": f"default_{engine_type.lower()}_step"},
            {"/metric@metrics.seqeval_accuracy": "seqeval_accuracy_score"},
            {"/metric@metrics.seqeval_precision": "seqeval_precision_score"},
            {"/metric@metrics.seqeval_recall": "seqeval_recall_score"},
            {"/metric@metrics.seqeval_f1": "seqeval_f1_score"},
            {
                "/metric@metrics.seqeval_classification_report": "seqeval_classification_report"
            },
        ],
    )

for engine_type in ["Validation", "Test"]:
    AtriaModuleRegistry.register_engine(
        module=".".join((_get_parent_module(__name__), f"engines.evaluation")),
        registered_class_or_func=f"{engine_type}Engine",
        name=f"layout_token_classification_{engine_type.lower()}_engine",
        hydra_defaults=[
            "_self_",
            {"/engine_step@engine_step": f"default_{engine_type.lower()}_step"},
            {"/metric@metrics.layout_precision": "layout_precision"},
            {"/metric@metrics.layout_recall": "layout_recall"},
            {"/metric@metrics.layout_f1": "layout_f1"},
        ],
    )

for engine_type in ["Validation", "Test"]:
    AtriaModuleRegistry.register_engine(
        module=".".join((_get_parent_module(__name__), f"engines.evaluation")),
        registered_class_or_func=f"{engine_type}Engine",
        name=f"question_answering_{engine_type.lower()}_engine",
        hydra_defaults=[
            "_self_",
            {"/engine_step@engine_step": f"default_{engine_type.lower()}_step"},
            {"/metric@metrics.sequence_anls": "sequence_anls"},
        ],
    )

for engine_type in ["Validation", "Test"]:
    AtriaModuleRegistry.register_engine(
        module=".".join((_get_parent_module(__name__), f"engines.evaluation")),
        registered_class_or_func=f"{engine_type}Engine",
        name=f"generative_modeling_{engine_type.lower()}_engine",
        hydra_defaults=[
            "_self_",
            {"/engine_step@engine_step": f"default_{engine_type.lower()}_step"},
            {"/metric@metrics.fid_score": "fid_score"},
        ],
    )
