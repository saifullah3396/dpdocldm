#!/bin/bash -l
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PYTHONPATH=$SCRIPT_DIR/src:$PYTHONPATH python ./src/atria/conf/build_configurations.py
