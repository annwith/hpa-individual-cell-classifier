#PBS -N hpa-2-place
#PBS -q testegpu
#PBS -e /home/lovelace/proj/proj1018/jmidlej/wsss-hpa-ju/logs/test.err
#PBS -o /home/lovelace/proj/proj1018/jmidlej/wsss-hpa-ju/logs/test.log
#PBS -m abe

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

DEBUG=0
WORK_DIR=/home/lovelace/proj/proj1018/jmidlej/wsss-hpa-ju

## environment region
unset CUDA_VISIBLE_DEVICES

PY=python3
PIP=pip
DATASETS_DIR=/home/lovelace/proj/proj1018/jmidlej
WORKERS_TRAIN=10
DEVICE=cuda

echo "Loading modules cudnn/8.2.0.53-11.3-gcc-9.3.0 python/3.10.10-gcc-9.4.0"
module load cudnn/8.2.0.53-11.3-gcc-9.3.0; module load python/3.10.10-gcc-9.4.0

# Activate virtual environment if it exists
echo "Activating virtual environment... ($WORK_DIR/dev/bin/activate)"
source $WORK_DIR/dev/bin/activate

# Navigate to the working directory
cd $WORK_DIR
echo "Working directory: $(pwd)"

## end region

## dataset region
DATASET=hparanzer     # HPA Single Cell Classification
TRAIN_CSV=$WORK_DIR/datasets/split/256_train+ext+rare.csv
DATA_DIR=$DATASETS_DIR/input/train_cell_256
IMAGE_SIZE=256
EVAL_BATCH=1

# end region

cd $WORK_DIR
export PYTHONPATH=$(pwd)

# wandb offline

## Architecture
ARCHITECTURE=resnest50
ARCH=rs50
# ARCHITECTURE=resnest101
# ARCH=rs101
# ARCHITECTURE=resnest269
# ARCH=rs269
# ARCHITECTURE=swin_b
# ARCH=swin_b_22k
# PRETRAINED_WEIGHTS=./experiments/models/pretrained/swin_base_patch4_window7_224_22k.pth
# ARCHITECTURE=swin_l
# ARCH=swin_l_22k
# PRETRAINED_WEIGHTS=./experiments/models/pretrained/swin_large_patch4_window7_224_22k.pth
# ARCHITECTURE=mit_b0
# ARCH=mit_b0
# PRETRAINED_WEIGHTS=./experiments/models/pretrained/mit_b0.pth
# ARCHITECTURE=mit_b5
# ARCH=mit_b5
# PRETRAINED_WEIGHTS=./experiments/models/pretrained/mit_b5.pth

TRAINABLE_STEM=true
TRAINABLE_STAGE4=true
TRAINABLE_BONE=true
DILATED=false
MODE=normal
CELL_LOGITS_TO_IMAGE_LOGITS=false

# PRETRAINED_WEIGHTS=./experiments/models/resnest269-0cc87c48.pth
PRETRAINED_WEIGHTS=imagenet

# Restore
RESTORE_DIR=$WORK_DIR/experiments/models/ranzer
MODEL_RESTORE=$RESTORE_DIR/hparanzer-256-rs50-lr0.0001-b6-ls0-aug2nd-adamw-cp-r3.pth

# Save
SAVE_FILE=$RESTORE_DIR/pred-hparanzer-256-rs50-lr0.0001-b6-ls0-aug2nd-adamw-cp-r3.csv

predict_ranzer() {
  echo "=================================================================="
  echo "[predict $TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

    $PY scripts/hpa/predict_ranzer.py \
    --device $DEVICE \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --backbone_weights $PRETRAINED_WEIGHTS \
    --trainable-stem $TRAINABLE_STEM \
    --trainable-backbone $TRAINABLE_BONE \
    --cell_logits_to_image_logits $CELL_LOGITS_TO_IMAGE_LOGITS \
    --image_size $IMAGE_SIZE \
    --dataset $DATASET \
    --train_csv $TRAIN_CSV \
    --data_dir $DATA_DIR \
    --validate_batch_size $EVAL_BATCH \
    --num_workers $WORKERS_TRAIN \
    --debug $DEBUG \
    --model_restore "$MODEL_RESTORE" \
    --save_file "$SAVE_FILE"

  echo "=================================================================="
  echo "[predict $TAG] finished at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="
}

# region Predict RANZER

predict_ranzer

# endregion
