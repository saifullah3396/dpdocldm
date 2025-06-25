#!/bin/bash

DATASET_ROOT_DIR=${DATASET_ROOT_DIR:?You must set the environment variable DATASET_ROOT_DIR}
LOG_LEVEL=DEBUG atria.prepare_data \
    hydra.searchpath=[./src/docsets/conf] \
    data_module=document_kie/funsd \
    data_module.dataset_dir=$DATASET_ROOT_DIR/funsd \
    data_module.dataset_cacher.cache_dir=$DATASET_ROOT_DIR/funsd/atria \
    output_dir=./ \
    $@