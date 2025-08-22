#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --gpus=4
#SBATCH -p sequana_gpu
#SBATCH -J priors
#SBATCH -o /scratch/lerdl/zanoni.dias/experiments/logs/hpa/%j-priors.out
#SBATCH -e /scratch/lerdl/zanoni.dias/experiments/logs/hpa/%j-priors.err
#SBATCH --time=28:00:00

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

source $SCRATCH/envs/hpa/bin/activate

$PIP install --user torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu111
$PIP install -r requirements.txt
$PIP install --force-reinstall cython==0.29.36
$PIP install --no-build-isolation git+https://github.com/lucasb-eyer/pydensecrf.git
# wandb offline

cd $WORK_DIR
export PYTHONPATH=$(pwd)
export PYTHONHTTPSVERIFY=0
export TORCH_HOME=$SCRATCH/.cache/torch

## Architecture
### Priors

# ARCHITECTURE=resnest50
# ARCH=rs50
# ARCHITECTURE=resnest101
# ARCH=rs101
ARCHITECTURE=resnest269
ARCH=rs269
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

# OPTIMIZER=momentum
# LR=0.01
# WD=0.001

# OPTIMIZER=lamb
# LR_ALPHA_SCRATCH=10.0
# LR_ALPHA_BIAS=2.0
# LR=0.0001
# WD=0.01

EMA_ENABLED=true
EMA_WARMUP=1
EMA_STEPS=1

MIXED_PRECISION=true
PERFORM_VALIDATION=true
VALIDATE_PRIORS=false

## Augmentation
AUGMENT=none  # randaugment_clahe_collorjitter_mixup_cutmix_cutormixup
AUG=no
CUTMIX=0.5
MIXUP=0.5
LABELSMOOTHING=0.1

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


train_cp() {
  echo "=================================================================="
  echo "[train $TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  WANDB_TAGS="$DATASET,$ARCH,lr:$LR,wd:$WD,ls:$LABELSMOOTHING,b:$BATCH,aug:$AUG,opt:$OPTIMIZER,sampler:$SAMPLER" \
  WANDB_RUN_GROUP="$DATASET-$ARCH-vanilla-cp" \
  CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/hpa/train_cp.py \
    --tag $TAG \
    --lr $LR \
    --wd $WD \
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
    --label_smoothing $LABELSMOOTHING \
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
    --restore "$RESTORE" \
    --conf_preds $CONF_PREDS \
    --conf_alpha $CONF_ALPHA \
    --conf_gamma $CONF_GAMMA \
    --conf_reduction $CONF_REDUCTION;
}

evaluate_classifier() {
  echo "=================================================================="
  echo "[Evaluate:$TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  WANDB_TAGS="$DATASET,$ARCH,lr:$LR,wd:$WD,ls:$LABELSMOOTHING,b:$BATCH,ac:$ACCUMULATE_STEPS,domain:$DOMAIN" \
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

inference_priors() {
  echo "=================================================================="
  echo "[Inference:$TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/cam/inference.py \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --tag $TAG \
    --dataset $DATASET \
    --domain $DOMAIN \
    --resize $INF_IMAGE_SIZE \
    --data_dir $DATA_DIR \
    --device $DEVICE;
}

evaluate_priors() {
  WANDB_TAGS="$DATASET,$ARCH,lr:$LR,ls:$LABELSMOOTHING,b:$BATCH,ac:$ACCUMULATE_STEPS,domain:$DOMAIN,crf:$CRF_T-$CRF_GT" \
  CUDA_VISIBLE_DEVICES="" \
  $PY scripts/evaluate.py \
    --experiment_name $TAG \
    --dataset $DATASET \
    --domain $DOMAIN \
    --data_dir $DATA_DIR \
    --min_th $MIN_TH \
    --max_th $MAX_TH \
    --crf_t $CRF_T \
    --crf_gt_prob $CRF_GT \
    --mode npy \
    --num_workers $WORKERS_INFER;
}

inference_instance_masks() {
  echo "=================================================================="
  echo "[instance-masks/Inference:$TAG] started at $(date +'%Y-%m-%d %H:%M:%S')."
  echo "=================================================================="

  CUDA_VISIBLE_DEVICES=$DEVICES \
    $PY scripts/hpa/inference.py \
    --architecture $ARCHITECTURE \
    --dilated $DILATED \
    --mode $MODE \
    --tag $TAG \
    --dataset $DATASET \
    --domain $DOMAIN \
    --scales 1.0 \
    --resize $INF_IMAGE_SIZE \
    --data_dir $DATA_DIR \
    --device $DEVICE
}

train_cp_evaluate_all() {
  train_cp
  evaluate_all
}

evaluate_all() {
  DOMAIN=train evaluate_classifier
  DOMAIN=valid evaluate_classifier
}


# region Vanilla Classification Experiments

# BATCH=64
# OPTIMIZER=lamb
# LR=0.001
# WD=0.01

# CLASS_WEIGHT=0.1,0.9,0.2,0.7,0.4,0.5,0.5,0.2,1.8,0.7,0.9,5.5,0.5,0.5,0.2,7.4,0.2,0.1,0.2
# CLASS_WEIGHT=0.1,1.0,0.5,1.0,1.0,1.0,1.0,0.5,1.0,1.0,1.0,10.0,1.0,0.5,0.5,5.0,0.2,0.5,1.0
# CLASS_WEIGHT=0.3,2.1,1.1,1.8,1.2,1.2,1.7,1.0,3.1,1.9,2.4,10.2,1.3,0.7,0.7,6.3,0.5,0.4,0.8
# AUG=clb

EID=r1  # Experiment ID
DOMAIN_TRAIN=train_aug_val

MIN_IMAGE_SIZE=448
MAX_IMAGE_SIZE=640
AUGMENT=none
AUG=no

EMA_STEPS=1

TAG=vanilla/$DATASET-$IMAGE_SIZE-${ARCH}-lr${LR}-b${BATCH}-ls$LABELSMOOTHING-$AUG-$OPTIMIZER-ema$EMA_STEPS-cp-$EID
train_cp
# evaluate_all

# endregion
