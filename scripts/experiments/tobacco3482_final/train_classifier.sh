#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_PATH=$SCRIPT_DIR/../../../
PYTHONPATH=$BASE_PATH/src:$BASE_PATH/external/atria/src:$BASE_PATH/external/docsets/src:$PYTHONPATH

generate_experiment_names() {
    local -n paths=$1

    local experiment_names=()

    for path in "${paths[@]}"; do
        # Extract key components from the path
        local name=$(echo "$path" | \
            sed -E 's|.*/atria_trainer/||' | \
            sed -E 's|/samples/.*||' | \
            tr '/' '_' | \
            sed 's|\*||g')

        # Ensure uniqueness by appending a hash if necessary
        local unique_name="${name}_$(echo -n "$path" "-val" | md5sum | cut -c1-8)"

        experiment_names+=("$unique_name")
    done

    echo "${experiment_names[@]}"
}

declare -a train_dataset_override_path=(
    "output/atria_trainer/Tobacco3482/dp_unaug_layout_cond_klf4_cfg_per_label_eps_1/**/**/**/*samples.msgpack"
    "output/atria_trainer/Tobacco3482/dp_unaug_layout_cond_sbv1.4_cfg_per_label_eps_1/**/**/**/*samples.msgpack"
    # "output/atria_trainer/Tobacco3482/dp_unaug_layout_cond_klf4_cfg_per_label_eps_5/**/**/**/*samples.msgpack"
    # "output/atria_trainer/Tobacco3482/dp_unaug_layout_cond_sbv1.4_cfg_per_label_eps_5/**/**/**/*samples.msgpack"
    # "output/atria_trainer/Tobacco3482/dp_unaug_layout_cond_sbv1.4_cfg_per_label_eps_10/**/**/**/*samples.msgpack"
    # "output/atria_trainer/Tobacco3482/dp_unaug_layout_cond_klf4_cfg_per_label_eps_10/**/**/**/*samples.msgpack"
)

# Generate and store unique experiment names
experiment_names=($(generate_experiment_names train_dataset_override_path))

# Use the train_dataset_override_path in the script
for idx in ${!train_dataset_override_path[@]}; do
    path=${train_dataset_override_path[$idx]}
    experiment_name=${experiment_names[$idx]}
    if compgen -G "$path" > /dev/null; then
        echo "processing experiment: $experiment_name"
        MODEL=convnext_base
        PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=tobacco3482 python $BASE_PATH/external/atria/src/atria/core/task_runners/atria_trainer.py \
            hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://dp_diffusion/conf] \
            +trainer_configs=image_classification/train \
            torch_model_builder@task_module.torch_model_builder=timm \
            +task_module.torch_model_builder.model_name=$MODEL \
            +task_module.torch_model_builder.drop_rate=0.5 \
            validation_engine.validate_every_n_epochs=1 \
            image_size=256 \
            experiment_name=$experiment_name \
            +data_module.train_dataset_override_path=$path \
            data_module.max_train_samples=50000

        MODEL=resnet50
        PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=tobacco3482 python $BASE_PATH/external/atria/src/atria/core/task_runners/atria_trainer.py \
            hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://dp_diffusion/conf] \
            +trainer_configs=image_classification/train \
            torch_model_builder@task_module.torch_model_builder=timm \
            +task_module.torch_model_builder.model_name=$MODEL \
            +task_module.torch_model_builder.drop_rate=0.5 \
            validation_engine.validate_every_n_epochs=1 \
            image_size=256 \
            experiment_name=$experiment_name \
            +data_module.train_dataset_override_path=$path \
            data_module.max_train_samples=50000

        MODEL=dp_diffusion.models.dit_model.DitModel
        PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=tobacco3482 python $BASE_PATH/external/atria/src/atria/core/task_runners/atria_trainer.py \
            hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://dp_diffusion/conf] \
            +trainer_configs=image_classification/train \
            torch_model_builder@task_module.torch_model_builder=local \
            +task_module.torch_model_builder.model_name=$MODEL \
            +task_module.torch_model_builder.drop_rate=0.5 \
            validation_engine.validate_every_n_epochs=1 \
            image_size=256 \
            experiment_name=$experiment_name \
            +data_module.train_dataset_override_path=$path \
            data_module.max_train_samples=50000
    else
        echo "$experiment_name does not exist"
    fi
done
