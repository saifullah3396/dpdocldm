#!/bin/bash -l
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

IMAGE_VER=2
IMAGE=/netscratch/$USER/envs/torchfusion_v$IMAGE_VER.sqsh
WORK_DIR=$SCRIPT_DIR/../../
MODELS_ROOT_DIR=/netscratch/$USER/dp_diffusion/models
DATASET_ROOT_DIR=/ds-sds/documents/
OUTPUT_DIR=/netscratch/$USER/dp_diffusion/outputs/
MOUNTS=/netscratch/$USER:/netscratch/$USER,/ds-sds:/ds-sds,/home/$USER/:/home/$USER/
CACHE_DIR=/netscratch/$USER/cache
PYTHON_PATH=$WORK_DIR/src
EXPORTS="TERM=linux,NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5,ROOT_DIR=/,USER_DIR=$USER,DATASET_ROOT_DIR=$DATASET_ROOT_DIR,MODELS_ROOT_DIR=$MODELS_ROOT_DIR,ATRIA_CACHE_DIR=$CACHE_DIR,PYTHONPATH=$PYTHON_PATH,TORCH_HOME=$CACHE_DIR/pretrained,OUTPUT_DIR=$OUTPUT_DIR"
NODES=1
TASKS=1
GPUS_PER_TASK=1
CPUS_PER_TASK=8
PARTITION=batch
MEMORY=80
UPDATE_IMAGE=0
SAVE_IMAGE_VER=$((IMAGE_VER + 1))
SAVE_IMAGE=/netscratch/$USER/envs/xai_torch_v$SAVE_IMAGE_VER.sqsh
POSITIONAL_ARGS=

usage() {
    echo "Usage:"
    echo "./gpu_run.sh <cmd>"
    echo ""
    echo " --image: Container image to use. "
    echo " --update_image: Whether to create a new image on close"
    echo " --work-dir: Path to work directory. "
    echo " --mounts: Directories to mount. "
    echo " --nodes : Number of nodes."
    echo " --tasks : Number of tasks per node."
    echo " --gpus_per_task : Number of GPUs per task."
    echo " --cpus_per_task : Number of GPUs per task."
    echo " -p | --partition : GPU partition"
    echo " -m | --memory : Total memory"
    echo " -h | --help : Displays the help"
    echo ""
}

for i in "$@"; do
    case $i in
    --help)
        shift # past argument
        usage
        exit
        ;;
    --update-image=*)
        UPDATE_IMAGE="${i#*=}"
        shift
        ;;
    --nodes=*)
        NODES="${i#*=}"
        shift # past argument
        shift # past value
        ;;
    --ntasks=*)
        TASKS="${i#*=}"
        shift # past argument
        shift # past value
        ;;
    --gpus-per-task=*)
        GPUS_PER_TASK="${i#*=}"
        shift # past argument
        shift # past value
        ;;
    --cpus-per-task=*)
        CPUS_PER_TASK="${i#*=}"
        shift # past argument
        shift # past value
        ;;
    --partition=*)
        PARTITION="${i#*=}"
        shift # past argument
        shift # past value
        ;;
    --memory=*)
        MEMORY="${i#*=}"
        shift # past argument
        shift # past value
        ;;
    *)
        POSITIONAL_ARGS+=("$i") # save positional arg
        shift                   # past argument
        ;;
    esac
done

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
# MEMORY=$(($MEMORY * $TASKS / $NODES))
set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

echo $(pwd)/scripts/slurm/install.sh

if [ $UPDATE_IMAGE == 0 ]; then
    if [ $GPUS_PER_TASK == 0 ]; then
        set -x
        srun \
            --container-image=$IMAGE \
            --container-workdir=$WORK_DIR \
            --container-mounts=$MOUNTS \
            --export=$EXPORTS \
            -K \
            --nodes=$NODES \
            --ntasks-per-node=$(($TASKS / $NODES)) \
            --ntasks=$TASKS \
            --cpus-per-task=$CPUS_PER_TASK \
            --mem="${MEMORY}G" \
            --partition=$PARTITION \
            --task-prolog="$(pwd)/scripts/slurm/install.sh" \
            $@
    else
        set -x
        srun \
            --container-image=$IMAGE \
            --container-workdir=$WORK_DIR \
            --container-mounts=$MOUNTS \
            --export=$EXPORTS \
            -K \
            --nodes=$NODES \
            --ntasks-per-node=$(($TASKS / $NODES)) \
            --ntasks=$TASKS \
            --gpus-per-task=$GPUS_PER_TASK \
            --cpus-per-task=$CPUS_PER_TASK \
            --mem="${MEMORY}G" \
            --partition=$PARTITION \
            --gpu-bind=none \
            --task-prolog="$(pwd)/scripts/slurm/install.sh" \
            $@
    fi
else

    if [ $GPUS_PER_TASK == 0 ]; then
        set -x
        srun \
            --container-image=$IMAGE \
            --container-workdir=$WORK_DIR \
            --container-mounts=$MOUNTS \
            --container-save=$SAVE_IMAGE \
            --export=$EXPORTS \
            -K \
            --nodes=$NODES \
            --ntasks-per-node=$(($TASKS / $NODES)) \
            --ntasks=$TASKS \
            --cpus-per-task=$CPUS_PER_TASK \
            --mem="${MEMORY}G" \
            -p $PARTITION \
            --task-prolog="$(pwd)/scripts/slurm/install.sh" \
            $@
    else
        echo "Better not to run update-image with gpu allocated."
    fi
fi
