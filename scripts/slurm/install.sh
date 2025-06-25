#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
echo $SCRIPT_DIR/../../external/atria/

if [[ $SLURM_LOCALID == 0 ]]; then
  # pip install fire

  # # install python dependencies
  # pip install -r $SCRIPT_DIR/../../cluster_requirements.txt

  # # install libraries
  apt update
  apt install libgl1-mesa-glx -y
  # source /netscratch/saifullah/envs/TORCH_FUSION/bin/activate/
  # pip install fsspec==2023.9.2
  # pip install h5py captum anls textdistance editdistance sacremoses imagesize
  # pip install -r $SCRIPT_DIR/../../requirements.txt
  # pip install -r $SCRIPT_DIR/../../external/atria/requirements.txt
  # pip install -r $SCRIPT_DIR/../../external/insightx/requirements.txt
  # pip install captum lime
  apt update
  apt install hdf5-tools
  pip install numpy==1.26.4 opencv-python==4.5.4.60 pymupdf
  pip install -U diffusers accelerate

  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi
