#!/bin/bash -l
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PYTHONPATH=$SCRIPT_DIR/src:$SCRIPT_DIR/external/atria/src:$SCRIPT_DIR/external/opacus_dpdm:$SCRIPT_DIR/external/fast-differential-privacy:$PYTHONPATH python ./src/dp_diffusion/conf/build_configurations.py
