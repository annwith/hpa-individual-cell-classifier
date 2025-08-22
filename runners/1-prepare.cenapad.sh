#PBS -N hpa-priors
#PBS -q par16
#PBS -e experiments/logs/hpa-prepare.1.err
#PBS -o experiments/logs/hpa-prepare.1.log

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
# Train a model to perform multilabel classification over a WSSS dataset.
#

ENV=cenapad
SCRATCH=/home/lovelace/proj/proj1018/ldavid
WORK_DIR=$SCRATCH/wsss-hpa

echo "## Sourcing virtualenv ${SCRATCH}/dev"
source $SCRATCH/dev/bin/activate

## Dataset
# DATASET=voc12       # Pascal VOC 2012
# DATASET=coco14      # MS COCO 2014
# DATASET=deepglobe   # DeepGlobe Land Cover Classification
# DATASET=cityscapes  # Cityscapes Urban Semantic Segmentation
DATASET=hpa         # HPA Single Cell Classification

. $WORK_DIR/runners/config/env.sh
. $WORK_DIR/runners/config/dataset.sh

# region Setup
# echo "## Installing dependencies"
# cd $WORK_DIR
# $PIP install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --extra-index-url https://download.pytorch.org/whl/cu118
# $PIP install --force-reinstall cython==0.29.36
# $PIP install --no-build-isolation git+https://github.com/lucasb-eyer/pydensecrf.git
# $PIP install -r PNOC/requirements.txt
# $PIP install -r requirements.txt
# endregion

cd $WORK_DIR
export PYTHONPATH=$(pwd)

$PY scripts/hpa/prepare_ds.py

deactivate
