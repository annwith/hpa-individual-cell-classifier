#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gpus=2
#SBATCH -p sequana_gpu
#SBATCH -J tr-vanilla
#SBATCH -o /scratch/lerdl/zanoni.dias/experiments/logs/setup-%j.out
#SBATCH -e /scratch/lerdl/zanoni.dias/experiments/logs/setup-%j.err
#SBATCH --time=01:00:00

# Copyright 2023 Lucas Oliveira David
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Setup.
#

echo "[puzzle/train.sequana] started running at $(date +'%Y-%m-%d %H:%M:%S')."

nodeset -e $SLURM_JOB_NODELIST

echo "PATH ${SCRATCH}/wsss-hpa"
cd $SCRATCH/wsss-hpa

module load sequana/current
module load gcc/7.4_sequana python/3.9.12_sequana cudnn/8.2_cuda-11.1_sequana
# module load gcc/7.4_sequana python/3.8.2_sequana cudnn/8.2_cuda-11.1_sequana

PY=python3.9
PIP=pip3.9

export PYTHONPATH=$(pwd)
# export OMP_NUM_THREADS=16

virtualenv $SCRATCH/envs/hpa --python $PY
source $SCRATCH/envs/hpa/bin/activate

pip install --user torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu111
pip install --user -r requirements.txt

