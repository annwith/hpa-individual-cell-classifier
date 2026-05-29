# hpa-individual-cell-classifier

Training and inference code for individual cell classification on the Human Protein Atlas (HPA) single-cell setting. This repository is based on the second-place solution of the Kaggle HPA Single Cell Classification competition: [HPA-singlecell-2nd-dual-head-pipeline](https://github.com/iseekwonderful/HPA-singlecell-2nd-dual-head-pipeline).

## About this repository

This repository contains code used in the dissertation "Classificação Fracamente Supervisionada da Localização Subcelular de Proteínas". In the dissertation, this code was used for experiments involving individual-cell classification in the Human Protein Atlas (HPA) setting, including experiments with conformal prediction.

The model weights used in the dissertation with this repository are available in [Google Drive](https://drive.google.com/drive/folders/1RU8gWto2Par0Ll3BlPtvH3NKfVAUTAcm?usp=drive_link). In particular:

* ConformalPrediction/: weights used in the dissertation experiments involving conformal prediction;

The weights obtained from the reproduction of the second-place solution [HPA-singlecell-2nd-dual-head-pipeline](https://github.com/iseekwonderful/HPA-singlecell-2nd-dual-head-pipeline) and used in the dissertation are also available:

* HPA2ndPlace/: weights obtained from the reproduction of the second-place Kaggle solution;

* HPA2ndPlacePretainedWeights/: pretrained weights used to initialize the architectures from the reproduced second-place solution.

These files are not stored directly in the GitHub repository due to their size.

## Repository structure

- `scripts/hpa/train.py`: main training entry point
- `runners/ic-hpa.sh`: training runner with a complete experiment configuration
- `runners/predict-ic-hpa.sh`: basic prediction runner
- `scripts/hpa/predict.py`: inference script used by the prediction runner
- `configs/`: augmentation configs
- `requirements.txt`: Python dependencies

## Setup

Clone the repository and install dependencies:

```bash
git clone [https://github.com/annwith/hpa-individual-cell-classifier.git](https://github.com/annwith/hpa-individual-cell-classifier.git)
cd hpa-individual-cell-classifier

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Before running

Both runner scripts contain local paths that should be edited for your machine before execution.

Check these variables:
- `WORK_DIR`
- `DATASETS_DIR`
- `TRAIN_CSV`
- `DATA_DIR`
- `CUDA_VISIBLE_DEVICES`

For prediction, also update:
- `MODEL_RESTORE`
- `SAVE_FILE`

## Training

### Option 1: run the training runner

The easiest way to reproduce the default training setup is:

```bash
bash runners/ic-hpa.sh
```

This runner wraps `scripts/hpa/train.py` and already passes the main training arguments.

The current default configuration in `runners/ic-hpa.sh` uses:
- `resnest50`
- `imagenet` backbone weights
- image size `256`
- training CSV `datasets/split/256_train+ext+rare.csv`
- data directory `train_cell_256`
- optimizer `adamw`
- learning rate `0.0002`
- `10` epochs
- batch size `6`
- gradient accumulation `4`
- mixed precision enabled
- augmentation config `configs/sin_256_final.yaml`

Model outputs are saved under:
`./experiments/models/<TAG>/`

where `<TAG>` is built automatically by the runner.

### Option 2: call train.py directly

A minimal example is:

```bash
python scripts/hpa/train.py \
  --train_csv /path/to/train.csv \
  --data_dir /path/to/train_cell_256 \
  --dataset hpa2nd \
  --architecture resnest50 \
  --backbone_weights imagenet \
  --image_size 256 \
  --batch_size 6 \
  --validate_batch_size 1 \
  --optimizer adamw \
  --lr 0.0002 \
  --max_epoch 10 \
  --accumulate_steps 4 \
  --mixed_precision true \
  --aug_yaml configs/sin_256_final.yaml \
  --tag my-run
```

The training script supports many other options for optimization, warmup, EMA, confidence-aware training, checkpoint restore.

## Prediction

Run the basic prediction pipeline with:

```bash
bash runners/predict-ic-hpa.sh
```

This runner calls `scripts/hpa/predict.py` and expects a trained checkpoint in `MODEL_RESTORE`.

A minimal direct example is:

```bash
python scripts/hpa/predict.py \
  --train_csv /path/to/train.csv \
  --data_dir /path/to/train_cell_256 \
  --architecture resnest50 \
  --backbone_weights imagenet \
  --image_size 256 \
  --validate_batch_size 1 \
  --model_restore /path/to/model.pth \
  --save_file /path/to/predictions.csv
```

The generated CSV contains both cell-level and image-level outputs, including filenames, labels, logits, probabilities, and a type column indicating whether the row corresponds to a cell or to a full image.

## Kaggle submission example

A public Kaggle submission notebook is available here:
[HPA CP Public Submission](https://www.kaggle.com/code/julianamidlej/1-hpa-cp-public-submission)
