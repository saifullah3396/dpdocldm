#!/bin/bash
DATASET_ROOT_DIR=${DATASET_ROOT_DIR:?You must set the environment variable DATASET_ROOT_DIR}
DATASET_ROOT_DIR=$DATASET_ROOT_DIR PYTHONPATH=$PYTHONPATH:../src:../atria/src:../docsets/src LOG_LEVEL=DEBUG python ../atria/src/atria/core/task_runners/atria_data_processor.py \
    hydra.searchpath=[./src/docsets/conf] \
    data_module=document_classification/ccpdf \
    $@