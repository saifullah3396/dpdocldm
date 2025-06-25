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
        local unique_name="${name}_$(echo -n "$path" | md5sum | cut -c1-8)"

        experiment_names+=("$unique_name")
    done

    echo "${experiment_names[@]}"
}

declare -a train_dataset_override_path=(
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_layout_cond_klf4_cfg_bs_4096/None/samples/DDPMScheduler/cfg-3.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_layout_cond_klf4_cfg_per_label/**/**/**/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_layout_cond_klf4_cfg_bs_4096/None/samples/DDPMScheduler/cfg-1.0/*samples.msgpack"
    # # dp unaug layout
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_layout_cond_sbv1.4_cfg_bs_4096/None/samples/DDPMScheduler/cfg-1.0/*samples.msgpack"
    # # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_layout_cond_sbv1.4_cfg_bs_4096/None/samples/DDPMScheduler/cfg-3.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_layout_cond_sbv1.4_cfg_per_label/**/**/**/*samples.msgpack"

    # dp promise unaug layout
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_layout_cond_sbv1.4_cfg_bs_4096/None/samples/DDPMScheduler/cfg-1.0/*samples.msgpack"
    # # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_layout_cond_sbv1.4_cfg_bs_4096/None/samples/DDPMScheduler/cfg-3.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_layout_cond_sbv1.4_cfg_per_label/**/**/**/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_layout_cond_klf4_cfg_bs_4096/None/samples/DDPMScheduler/cfg-1.0/*samples.msgpack"
    # # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_layout_cond_klf4_cfg_bs_4096/None/samples/DDPMScheduler/cfg-3.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_layout_cond_klf4_cfg_per_label/**/**/**/*samples.msgpack"

    # dp unaug cls
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_class_cond_sbv1.4_cfg_bs_4096/None/samples/DDPMScheduler/cfg-1.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_class_cond_sbv1.4_cfg_bs_4096/None/samples/DDPMScheduler/cfg-3.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_class_cond_sbv1.4_cfg_per_label/**/**/**/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_class_cond_klf4_cfg_bs_4096/None/samples/DDPMScheduler/cfg-1.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_class_cond_klf4_cfg_bs_4096/None/samples/DDPMScheduler/cfg-3.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_class_cond_klf4_cfg_per_label/**/**/**/*samples.msgpack"
    # dp promise unaug cls
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_class_cond_sbv1.4_cfg_bs_4096/None/samples/DDPMScheduler/cfg-1.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_class_cond_sbv1.4_cfg_bs_4096/None/samples/DDPMScheduler/cfg-3.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_class_cond_sbv1.4_cfg_per_label/**/**/**/*samples.msgpack"
    "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_class_cond_klf4_cfg_bs_4096/None/samples/DDPMScheduler/cfg-1.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_class_cond_klf4_cfg_bs_4096/None/samples/DDPMScheduler/cfg-3.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_class_cond_klf4_cfg_per_label/**/**/**/*samples.msgpack"

    # dp unaug
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_sbv1.4_cfg/None/samples/DDPMScheduler/cfg-1.0/*samples.msgpack"
    # # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_sbv1.4_cfg/None/samples/DDPMScheduler/cfg-3.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_sbv1.4_cfg_per_label/**/**/**/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_klf4_cfg/None/samples/DDPMScheduler/cfg-1.0/*samples.msgpack"
    # # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_klf4_cfg/None/samples/DDPMScheduler/cfg-3.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_unaug_klf4_cfg_per_label/**/**/**/*samples.msgpack"
    # # dp promise unaug
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_sbv1.4_cfg/None/samples/DDPMScheduler/cfg-1.0/*samples.msgpack"
    # # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_sbv1.4_cfg/None/samples/DDPMScheduler/cfg-3.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_sbv1.4_cfg_per_label/**/**/**/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_klf4_cfg/None/samples/DDPMScheduler/cfg-1.0/*samples.msgpack"
    # # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_klf4_cfg/None/samples/DDPMScheduler/cfg-3.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_unaug_klf4_cfg_per_label/**/**/**/*samples.msgpack"

    # dp aug
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_aug_sbv1.4_cfg/None/samples/DDPMScheduler/cfg-1.0/*samples.msgpack"
    # # "output/atria_trainer/RvlCdip/rvlcdip_dp_aug_sbv1.4_cfg/None/samples/DDPMScheduler/cfg-3.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_aug_sbv1.4_cfg_per_label/**/**/**/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_aug_klf4_cfg/None/samples/DDPMScheduler/cfg-1.0/*samples.msgpack"
    # # "output/atria_trainer/RvlCdip/rvlcdip_dp_aug_klf4_cfg/None/samples/DDPMScheduler/cfg-3.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_aug_klf4_cfg_per_label/**/**/**/*samples.msgpack"
    # # dp promise aug
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_aug_sbv1.4_cfg/None/samples/DDPMScheduler/cfg-1.0/*samples.msgpack"
    # # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_aug_sbv1.4_cfg/None/samples/DDPMScheduler/cfg-3.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_aug_sbv1.4_cfg_per_label/**/**/**/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_aug_klf4_cfg/None/samples/DDPMScheduler/cfg-1.0/*samples.msgpack"
    # # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_aug_klf4_cfg/None/samples/DDPMScheduler/cfg-3.0/*samples.msgpack"
    # "output/atria_trainer/RvlCdip/rvlcdip_dp_promise_aug_klf4_cfg_per_label/**/**/**/*samples.msgpack"
)

# Generate and store unique experiment names
experiment_names=($(generate_experiment_names train_dataset_override_path))

# Initialize CONFIGS array
CONFIGS=()

# Use the train_dataset_override_path in the script
for idx in ${!train_dataset_override_path[@]}; do
    path=${train_dataset_override_path[$idx]}
    experiment_name=${experiment_names[$idx]}
    if compgen -G "$path" > /dev/null; then
        echo "processing experiment: $experiment_name"
        MODEL=convnext_base
        CONFIGS+=("${SCRIPT_DIR}/train_cls.sh '${path}' $experiment_name")
    else
        echo "$experiment_name does not exist"
    fi
done

# Print the CONFIGS array
echo "Generated CONFIGS:"
for config in "${CONFIGS[@]}"; do
    $config
done