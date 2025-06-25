from enum import Enum


class ModelTasks(str, Enum):
    image_classification = "image_classification"
    sequence_classification = "sequence_classification"
    token_classification = "token_classification"
    question_answering = "question_answering"
    visual_question_answering = "visual_question_answering"
    layout_token_classification = "layout_token_classification"
    image_generation = "image_generation"
    object_detection = "object_detection"
    embedding = "embedding"
    autoencoding = "autoencoding"
    gan = "gan"
    diffusion = "diffusion"
    custom = "custom"
