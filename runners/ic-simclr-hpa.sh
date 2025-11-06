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

DEBUG=1
PRINT_RATIO=0.1
MONITOR_MEMORY_USAGE=true
WORK_DIR=/home/jumidlej/git-projects/hpa-individual-cell-classifier
# WORK_DIR=/home/unicamp/200208/models/hpa-individual-cell-classifier

## environment region

PY=python3
PIP=pip
WORKERS_TRAIN=8
DATASETS_DIR=/home/jumidlej/datasets
# DATASETS_DIR=/home/unicamp/200208/datasets

export CUDA_VISIBLE_DEVICES=0
DEVICE='cuda:0'

## end region

## dataset region
DATASET=hpa2nd     # HPA Single Cell Classification
TRAIN_CSV=$WORK_DIR/datasets/split/dino_with_labels.csv
DATA_DIR=$DATASETS_DIR/input/train_cell_256

IMAGE_SIZE=256
SAMPLER=default

WEAKLY_SUPERVISED=false
TEMPERATURE=0.07

# end region

cd $WORK_DIR
export PYTHONPATH=$(pwd)

# wandb offline

## Architecture
### Priors

ARCHITECTURE=resnest50
ARCH=rs50

TRAINABLE_STEM=true
TRAINABLE_STAGE4=true
TRAINABLE_BONE=true
DILATED=false
MODE=normal

PRETRAINED_WEIGHTS=none

# Training
OPTIMIZER=adamw  # sgd,lion,lamb
LR=0.0002
WD=0.0

WARMUP_EPOCHS=0
WARMUP_START_FACTOR=0.01

EPOCHS=10
EPOCH0=0
BATCH=2
EVAL_BATCH=1
ACCUMULATE_STEPS=8

MIXED_PRECISION=true

## Augmentation and normalization
NORM_MEAN=0.5,0.5,0.5,0.5
NORM_STD=0.5,0.5,0.5,0.5
AUGMENT_YAML=$WORK_DIR/configs/sin_256_final.yaml
AUG=aug2nd
# AUGMENT_YAML=""
# AUG=no

RESTORE_DIR=$WORK_DIR

# Restore
MODEL_RESTORE=""
OPTIMIZER_RESTORE=""
SCALER_RESTORE=""
TRAIN_META_RESTORE=""
SCHEDULER_RESTORE=""

# MODEL_RESTORE=$RESTORE_DIR/model-f0-e0.pth
# OPTIMIZER_RESTORE=$RESTORE_DIR/optimizer.pth
# SCALER_RESTORE=$RESTORE_DIR/scaler.pth
# TRAIN_META_RESTORE=$RESTORE_DIR/training_meta.pth
# SCHEDULER_RESTORE=$RESTORE_DIR/scheduler.pth

train() {
  echo "===================================================================================================="
  echo "[train $TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "===================================================================================================="

  WANDB_TAGS="$DATASET,$ARCH,lr:$LR,wd:$WD,ls:$LABELSMOOTHING,b:$BATCH,aug:$AUG,opt:$OPTIMIZER,sampler:$SAMPLER" \
  WANDB_RUN_GROUP="$DATASET-$ARCH-dual-head" \
    $PY scripts/hpa/train_simclr_ws.py \
    --device $DEVICE \
    --weakly_supervised $WEAKLY_SUPERVISED \
    --temperature $TEMPERATURE \
    --optimizer $OPTIMIZER \
    --lr $LR \
    --wd $WD \
    --warmup_epochs $WARMUP_EPOCHS \
    --warmup_start_factor $WARMUP_START_FACTOR \
    --accumulate_steps $ACCUMULATE_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --backbone_weights $PRETRAINED_WEIGHTS \
    --trainable-stem $TRAINABLE_STEM \
    --trainable-backbone $TRAINABLE_BONE \
    --image_size $IMAGE_SIZE \
    --normalization_mean $NORM_MEAN \
    --normalization_std $NORM_STD \
    --aug_yaml $AUGMENT_YAML \
    --first_epoch $EPOCH0 \
    --max_epoch $EPOCHS \
    --dataset $DATASET \
    --train_csv $TRAIN_CSV \
    --data_dir $DATA_DIR \
    --sampler $SAMPLER \
    --batch_size $BATCH \
    --num_workers $WORKERS_TRAIN \
    --debug $DEBUG \
    --print_ratio $PRINT_RATIO \
    --monitor_memory_usage $MONITOR_MEMORY_USAGE \
    --tag $TAG \
    --model_restore "$MODEL_RESTORE" \
    --optimizer_restore "$OPTIMIZER_RESTORE" \
    --scheduler_restore "$SCHEDULER_RESTORE" \
    --scaler_restore "$SCALER_RESTORE" \
    --train_meta_restore "$TRAIN_META_RESTORE"
  echo "===================================================================================================="
  echo "[train $TAG] finished at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "===================================================================================================="
}

# region Classification Experiments

EID=-test  # Experiment ID
TAG=$DATASET-${ARCH}-lr${LR}-b${BATCH}-$AUG-$OPTIMIZER-eid$EID

train

# endregion
