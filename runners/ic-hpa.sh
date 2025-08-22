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

DEBUG=0
PRINT_RATIO=0.1
MONITOR_MEMORY_USAGE=true
WORK_DIR=/home/jumidlej/hpa-individual-cell-classifier

## environment region

PY=python3
PIP=pip
WORKERS_TRAIN=10
DATASETS_DIR=/home/jumidlej/datasets

export CUDA_VISIBLE_DEVICES=0
DEVICE='cuda:0'

## end region

## dataset region
DATASET=hparanzer     # HPA Single Cell Classification
TRAIN_CSV=$WORK_DIR/datasets/split/256_train+ext+rare.csv
DATA_DIR=$DATASETS_DIR/input/train_cell_256
PERFORM_VALIDATION=true
VAL_FOLD=0

IMAGE_SIZE=256

CELL_COUNT=16
SAMPLER=default

# CELL_COUNT=-1
# SAMPLER=balanced_cell_count

# end region

cd $WORK_DIR
export PYTHONPATH=$(pwd)

# wandb offline

## Architecture
### Priors

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

# Confidences
IMAGE_CONF_AWARE_TRAINING=false
CELL_CONF_AWARE_TRAINING=false
CONF_PREDS=/home/jumidlej/predictions/pred-hparanzer-256-rs50-lr0.0002-b6-ls0-aug2nd-adamw-eid2-c16-e9-normalized.csv
CONF_ALPHA=1.0
CONF_GAMMA=0.5

REANNOTATE_NEG_LABELS=false
REANNOTATE_THRESHOLD=0.01

CELL_CONF_AS_CELL_LABELS=true

# Training
OPTIMIZER=adamw  # sgd,lion,lamb
POLY_LR_DECAY_OPTIMIZER=false
LR=0.0002
LR_ALPHA_SCRATCH=1.0
LR_ALPHA_BIAS=1.0
WD=0.0

WARMUP_EPOCHS=1
WARMUP_START_FACTOR=0.01

LABELSMOOTHING=0

# OPTIMIZER=momentum
# LR=0.01
# WD=0.001

# OPTIMIZER=lamb
# LR_ALPHA_SCRATCH=10.0
# LR_ALPHA_BIAS=2.0
# LR=0.0001
# WD=0.01

EPOCHS=10
EPOCH0=0
BATCH=4
EVAL_BATCH=1
ACCUMULATE_STEPS=6

CLASS_WEIGHT=none
CELL_POS_WEIGHT=1
CELL_LOSS_WEIGHT=0.1

EMA_ENABLED=false
EMA_WARMUP=1
EMA_STEPS=1
EMA_DECAY=0.99

MIXED_PRECISION=true

## Augmentation
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
    $PY scripts/hpa/train.py \
    --device $DEVICE \
    --optimizer $OPTIMIZER \
    --poly_lr_decay $POLY_LR_DECAY_OPTIMIZER \
    --lr $LR \
    --lr_alpha_scratch $LR_ALPHA_SCRATCH \
    --lr_alpha_bias $LR_ALPHA_BIAS \
    --wd $WD \
    --warmup_epochs $WARMUP_EPOCHS \
    --warmup_start_factor $WARMUP_START_FACTOR \
    --class_weight $CLASS_WEIGHT \
    --cell_pos_weight $CELL_POS_WEIGHT \
    --cell_loss_weight $CELL_LOSS_WEIGHT \
    --accumulate_steps $ACCUMULATE_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --label_smoothing $LABELSMOOTHING \
    --ema $EMA_ENABLED \
    --ema_warmup $EMA_WARMUP \
    --ema_steps $EMA_STEPS \
    --ema_decay $EMA_DECAY \
    --cell_conf_aware_training $CELL_CONF_AWARE_TRAINING \
    --image_conf_aware_training $IMAGE_CONF_AWARE_TRAINING \
    --conf_preds $CONF_PREDS \
    --conf_alpha $CONF_ALPHA \
    --conf_gamma $CONF_GAMMA \
    --reannotate_neg_labels $REANNOTATE_NEG_LABELS \
    --reannotate_threshold $REANNOTATE_THRESHOLD \
    --cell_conf_as_cell_labels $CELL_CONF_AS_CELL_LABELS \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --backbone_weights $PRETRAINED_WEIGHTS \
    --trainable-stem $TRAINABLE_STEM \
    --trainable-backbone $TRAINABLE_BONE \
    --cell_logits_to_image_logits $CELL_LOGITS_TO_IMAGE_LOGITS \
    --image_size $IMAGE_SIZE \
    --aug_yaml $AUGMENT_YAML \
    --first_epoch $EPOCH0 \
    --max_epoch $EPOCHS \
    --val_fold $VAL_FOLD \
    --validate $PERFORM_VALIDATION \
    --dataset $DATASET \
    --train_csv $TRAIN_CSV \
    --data_dir $DATA_DIR \
    --cell_count $CELL_COUNT \
    --sampler $SAMPLER \
    --batch_size $BATCH \
    --validate_batch_size $EVAL_BATCH \
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

EID=cell-conf-label-2  # Experiment ID
TAG=$DATASET-${ARCH}-lr${LR}-b${BATCH}-$AUG-$OPTIMIZER-eid$EID
# TAG=ranzer/test_r269

train

# endregion
