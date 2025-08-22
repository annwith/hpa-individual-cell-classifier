#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --gpus=4
#SBATCH -p sequana_gpu
#SBATCH -J rs50-priors-fold
#SBATCH -o /scratch/lerdl/lucas.david/experiments/logs/hpa/%j-priors-fold.out
#SBATCH --time=13:00:00

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

if [[ "`hostname`" == "sdumont"* ]]; then
  ENV=sdumont
  WORK_DIR=$SCRATCH/wsss-hpa
else
  ENV=local
  WORK_DIR=$HOME/workspace/repos/research/wsss/wsss-hpa
fi

## Dataset
# DATASET=voc12       # Pascal VOC 2012
# DATASET=coco14      # MS COCO 2014
# DATASET=deepglobe   # DeepGlobe Land Cover Classification
# DATASET=cityscapes  # Cityscapes Urban Semantic Segmentation
DATASET=hpa         # HPA Single Cell Classification

. $WORK_DIR/runners/config/env.sh
. $WORK_DIR/runners/config/dataset.sh

cd $WORK_DIR
export PYTHONPATH=$(pwd)

# $PIP install iterative-stratification

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
PRETRAINED_WEIGHTS=imagenet

# Training
OPTIMIZER=sgd  # sgd,lion,lamb
EPOCHS=15
EPOCH0=0
BATCH=32
ACCUMULATE_STEPS=1
RESTORE=""

CLASS_WEIGHT=none

LR_ALPHA_SCRATCH=10.0
LR_ALPHA_BIAS=2.0
LR_ALPHA_OC=1.0

# =========================
# $PIP install lion-pytorch
# OPTIMIZER=lion
# LR_ALPHA_SCRATCH=1.0
# LR_ALPHA_BIAS=1.0
# LR=0.00001
# WD=0.01
# =========================

EMA_ENABLED=true
EMA_WARMUP=1
EMA_STEPS=1

MIXED_PRECISION=true
PERFORM_VALIDATION=false
VALIDATE_PRIORS=false

## Augmentation
AUGMENT=none  # randaugment_clahe_collorjitter_mixup_cutmix_cutormixup
AUG=no
CUTMIX=0.5
MIXUP=0.5
LS=0.1

SAMPLER=default

## OC-CSE
OC_ARCHITECTURE=$ARCHITECTURE
OC_MASK_GN=true # originally done in OC-CSE
OC_STRATEGY=random
OC_F_MOMENTUM=0.9
OC_F_GAMMA=2.0
OC_PERSIST=false

## Schedule
P_INIT=0.0
P_ALPHA=4.0
P_SCHEDULE=0.5

OC_INIT=0.3
OC_ALPHA=1.0
OC_SCHEDULE=1.0

OW=1.0
OW_INIT=0.0
OW_SCHEDULE=1.0
OC_TRAIN_MASKS=cams
OC_TRAIN_MASK_T=0.2
OC_TRAIN_INT_STEPS=1

# Evaluation
MIN_TH=0.05
MAX_TH=0.81
CRF_T=0
CRF_GT=0.7

FOLD=0
FOLD_CV=stratified  # kfold, stratified
FOLD_SPLITS=5


train_fold() {
  echo "=================================================================="
  echo "[train $TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

#   WANDB_RUN_ID="$WANDB_ID" \
#   WANDB_RESUME="$WANDB_RESUME" \
  WANDB_TAGS="$DATASET,$ARCH,lr:$LR,wd:$WD,ls:$LS,b:$BATCH,aug:$AUG,opt:$OPTIMIZER,sampler:$SAMPLER,fold:$FOLD" \
  WANDB_RUN_GROUP="$DATASET-$ARCH-vanilla-fold" \
  CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/hpa/train_fold.py \
    --tag $TAG \
    --lr $LR \
    --wd $WD \
    --fold_id     $FOLD \
    --fold_cv     $FOLD_CV \
    --fold_splits $FOLD_SPLITS \
    --optimizer $OPTIMIZER \
    --lr_alpha_scratch $LR_ALPHA_SCRATCH \
    --lr_alpha_bias $LR_ALPHA_BIAS \
    --class_weight $CLASS_WEIGHT \
    --batch_size $BATCH \
    --accumulate_steps $ACCUMULATE_STEPS \
    --ema $EMA_ENABLED \
    --ema_warmup $EMA_WARMUP \
    --ema_steps $EMA_STEPS \
    --mixed_precision $MIXED_PRECISION \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --backbone_weights $PRETRAINED_WEIGHTS \
    --trainable-stem $TRAINABLE_STEM \
    --trainable-backbone $TRAINABLE_BONE \
    --image_size $IMAGE_SIZE \
    --min_image_size $MIN_IMAGE_SIZE \
    --max_image_size $MAX_IMAGE_SIZE \
    --augment $AUGMENT \
    --cutmix_prob $CUTMIX \
    --mixup_prob $MIXUP \
    --label_smoothing $LS \
    --first_epoch $EPOCH0 \
    --max_epoch $EPOCHS \
    --dataset $DATASET \
    --sampler $SAMPLER \
    --data_dir $DATA_DIR \
    --domain_train $DOMAIN_TRAIN \
    --domain_valid $DOMAIN_VALID \
    --validate $PERFORM_VALIDATION \
    --validate_priors $VALIDATE_PRIORS \
    --validate_max_steps $VALIDATE_MAX_STEPS \
    --validate_thresholds $VALIDATE_THRESHOLDS \
    --device $DEVICE \
    --num_workers $WORKERS_TRAIN \
    --restore "$RESTORE";
}

evaluate_classifier() {
  echo "=================================================================="
  echo "[Evaluate:$TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  WANDB_TAGS="$DATASET,$ARCH,lr:$LR,wd:$WD,ls:$LS,b:$BATCH,aug:$AUG,opt:$OPTIMIZER,sampler:$SAMPLER,fold:$FOLD" \
  WANDB_RUN_GROUP="$DATASET-$ARCH-vanilla-fold" \
  CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/cam/evaluate.py \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --tag $TAG \
    --dataset $DATASET \
    --data_dir $DATA_DIR \
    --domain $DOMAIN \
    --image_size $IMAGE_SIZE \
    --min_image_size $MIN_IMAGE_SIZE \
    --max_image_size $MAX_IMAGE_SIZE \
    --mixed_precision $MIXED_PRECISION \
    --save_preds experiments/predictions/$TAG-$DOMAIN-logit.npz \
    --device $DEVICE;
}

train_inference_fold() {
  DOMAIN=train_aug_val train_fold
  # DOMAIN=train_aug_val evaluate_classifier
}

WORKERS_TRAIN=8

OPTIMIZER=momentum
LR=0.001
WD=0.01

MIN_IMAGE_SIZE=448
MAX_IMAGE_SIZE=576
AUGMENT=mixup
AUG=resize-mixup
EMA_STEPS=1
EID=r3  # Experiment ID

FOLD=3
TAG=vanilla/$DATASET-$IMAGE_SIZE-${ARCH}-lr${LR}-b${BATCH}-ls$LS-$AUG-$OPTIMIZER-ema$EMA_STEPS-f$FOLD-$EID
DEVICES=0 train_inference_fold &

FOLD=0
TAG=vanilla/$DATASET-$IMAGE_SIZE-${ARCH}-lr${LR}-b${BATCH}-ls$LS-$AUG-$OPTIMIZER-ema$EMA_STEPS-f$FOLD-$EID
DEVICES=1 train_inference_fold &

# FOLD=2
# TAG=vanilla/$DATASET-$IMAGE_SIZE-${ARCH}-lr${LR}-b${BATCH}-ls$LS-$AUG-$OPTIMIZER-ema$EMA_STEPS-f$FOLD-$EID
# DEVICES=2 train_inference_fold &

# FOLD=4
# TAG=vanilla/$DATASET-$IMAGE_SIZE-${ARCH}-lr${LR}-b${BATCH}-ls$LS-$AUG-$OPTIMIZER-ema$EMA_STEPS-f$FOLD-$EID
# DEVICES=3 train_inference_fold &
wait

# FOLD=3
# TAG=vanilla/$DATASET-$IMAGE_SIZE-${ARCH}-lr${LR}-b${BATCH}-ls$LS-$AUG-$OPTIMIZER-ema$EMA_STEPS-f$FOLD-$EID
# DEVICES=0 train_inference_fold

