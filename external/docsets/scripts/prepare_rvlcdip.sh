#!/bin/bash

DATASET_ROOT_DIR=${DATASET_ROOT_DIR:?You must set the environment variable DATASET_ROOT_DIR}
LOG_LEVEL=DEBUG atria.prepare_data \
    hydra.searchpath=[./src/docsets/conf] \
    data_module=document_classification/rvlcdip \
    data_module.dataset_dir=$DATASET_ROOT_DIR/rvlcdip \
    data_module.dataset_cacher.cache_dir=$DATASET_ROOT_DIR/rvlcdip/atria \
    output_dir=./ \
    $@