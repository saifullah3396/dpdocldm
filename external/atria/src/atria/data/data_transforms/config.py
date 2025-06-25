from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

# register torch model torch_model_builders as child node of task_module
AtriaModuleRegistry.register_data_transform(
    module=_get_parent_module(__name__) + ".general",
    registered_class_or_func=[
        "NumpyToTensor",
        "NumpyBrightness",
        "NumpyContrast",
        "TensorGrayToRgb",
        "TensorRgbToBgr",
        "PilGrayToRgb",
        "PilRgbToGray",
    ],
)
AtriaModuleRegistry.register_data_transform(
    module=_get_parent_module(__name__) + ".noise",
    registered_class_or_func=[
        "TensorGaussianNoiseRgb",
        "NumpyShotNoiseRgb",
        "NumpyFibrousNoise",
        "NumpyMultiscaleNoise",
    ],
)
AtriaModuleRegistry.register_data_transform(
    module=_get_parent_module(__name__) + ".geometric",
    registered_class_or_func=[
        "NumpyTranslation",
        "NumpyScale",
        "NumpyRotation",
        "NumpyRandomChoiceAffine",
        "NumpyElastic",
        "TensorResizeOneDim",
        "TensorResizeWithAspectAndPad",
        "TensorRandomResize",
        "TensorRandomResizedCrop",
        "TensorRandomCrop",
        "NormalizeAndRescaleBoundingBoxes",
        "RescaleBoundingBoxes",
    ],
)
AtriaModuleRegistry.register_data_transform(
    module=_get_parent_module(__name__) + ".advanced",
    registered_class_or_func=[
        "ImageLoader",
        "ImagePreprocess",
        "BasicImageAug",
        "RandAug",
        "Moco",
        "BarlowTwins",
        "MultiCrop",
        "BinarizationAug",
        "Cifar10Aug",
        # "TwinDocs",
    ],
)
AtriaModuleRegistry.register_data_transform(
    module=_get_parent_module(__name__) + ".blur",
    registered_class_or_func=[
        "NumpyGaussianBlur",
        "NumpyBinaryBlur",
        "NumpyNoisyBinaryBlur",
        "NumpyDefocusBlur",
        "NumpyMotionBlur",
        "NumpyZoomBlur",
        "PilGaussianBlur",
    ],
)
AtriaModuleRegistry.register_data_transform(
    module=_get_parent_module(__name__) + ".distortions",
    registered_class_or_func=[
        "NumpyRandomDistortion",
        "NumpyRandomBlotches",
        "NumpySurfaceDistortion",
        "NumpyThreshold",
        "NumpyPixelate",
        "NumpyJpegCompression",
        "PilSolarization",
    ],
)

AtriaModuleRegistry.register_data_transform(
    module=_get_parent_module(__name__) + ".detectron2",
    registered_class_or_func=[
        "ObjectDetectionImagePreprocess",
        "ObjectDetectionImageAug",
    ],
)


AtriaModuleRegistry.register_data_transform(
    module=_get_parent_module(__name__) + ".huggingface_processors",
    registered_class_or_func=[
        "HuggingfaceProcessor",
        "QuestionAnsweringHuggingfaceProcessor",
        "SequenceSanitizer",
        "AnswerWordIndicesToAnswerTokenIndices",
    ],
)
AtriaModuleRegistry.register_data_transform(
    module=_get_parent_module(__name__) + ".cifar10_toy_aug",
    registered_class_or_func=[
        "Cifar10ToyAug",
    ],
)
