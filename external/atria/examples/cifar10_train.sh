#!/bin/bash

atria.train \
    output_dir=./output/cifar10 \
    data_module=huggingface \
    data_module.dataset_name=atria.data.datasets.cifar10.cifar10_huggingface_dataset.Cifar10HFDataset \
    task_module.torch_model_builder.model_name=resnet50 test_engine.test_run=False \
    train_validation_splitter@data_module.train_validation_splitter=default \
    +data_transform@data_module.runtime_data_transforms.train=basic_image_aug \
    +data_transform@data_module.runtime_data_transforms.evaluation=basic_image_aug \
    training_engine.max_epochs=1 \
    engine@test_engine=image_classification_test_engine \
    engine@validation_engine=image_classification_validation_engine \
    data_module.dataset_output_key_map='{image: img, label: label}' \
    data_collator@data_module.train_dataloader_builder.collate_fn=batch_to_tensor \
    data_collator@data_module.evaluation_dataloader_builder.collate_fn=batch_to_tensor \
    data_module.train_dataloader_builder.collate_fn.batch_filter_key_map='{image: image, label: label}' \
    data_module.evaluation_dataloader_builder.collate_fn.batch_filter_key_map='{image: image, label: label}' \
    \
    $@ # task_module.checkpoint=output/cifar10/atria_trainer/Cifar10AtriaDataset/resnet50/checkpoints/checkpoint_1.pt \

# atria.train \
#     output_dir=./output/cifar10 \
#     data_module=torch \
#     data_module.dataset_name=atria.data.datasets.cifar10.cifar10_torch_dataset.Cifar10TorchDataset \
#     task_module.torch_model_builder.model_name=resnet50 test_engine.test_run=False \
#     +data_transform@data_module.runtime_data_transforms.train=basic_image_aug \
#     +data_transform@data_module.runtime_data_transforms.evaluation=basic_image_aug \
#     training_engine.max_epochs=10 \
#     data_module.dataset_output_key_map='{image: img, label: label}' \
#     data_collator@data_module.train_dataloader_builder.collate_fn=batch_to_tensor \
#     data_collator@data_module.evaluation_dataloader_builder.collate_fn=batch_to_tensor \
#     # task_module.checkpoint=output/cifar10/atria_trainer/Cifar10AtriaDataset/resnet50/checkpoints/checkpoint_1.pt \
#     $@
