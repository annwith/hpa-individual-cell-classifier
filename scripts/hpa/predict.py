import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import cv2
cv2.setNumThreads(0)

import datasets
from datasets.hpa import GetPredictionsDataset

from core.networks import *
from tools.ai.augment_utils import *
from tools.ai.demo_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.log_utils import *
from tools.ai.optim_utils import *
from tools.ai.randaugment import *
from tools.ai.torch_utils import *
from tools.general.io_utils import *
from tools.general.time_utils import *

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--debug', default=None, type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--train_csv', required=True, type=str)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--validate_batch_size', default=1, type=int)
parser.add_argument('--image_size', default=256, type=int)

# Network
parser.add_argument('--architecture', default='resnest50', type=str)
parser.add_argument('--mode', default='normal', type=str)
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--trainable-backbone', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--backbone_weights', default="imagenet", type=str)
parser.add_argument('--cell_logits_to_image_logits', default=False, type=str2bool)

# Restore training
parser.add_argument('--model_restore', default=None, type=str)

# Save path
parser.add_argument('--save_file', default='predictions.csv', type=str)


try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
  print(f"GPUS={GPUS}")
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)
THRESHOLDS = list(np.arange(0.10, 0.50, 0.05))


def predict_model(
    model, 
    dl,
    save_file):

    # Check if file exists
    if os.path.exists(save_file):
        raise FileExistsError(f"File {save_file} already exists.")

    tq = tqdm(dl)

    model.eval()
    results = []
    with torch.no_grad():
        for i, (ipt, mask, lbl, image_lbl, n_cell, filename) in enumerate(tq):

            n_cell = n_cell[0]
            filename = filename[0]

            ipt = ipt.view(-1, ipt.shape[-3], ipt.shape[-2], ipt.shape[-1])
            lbl = lbl.view(-1, lbl.shape[-1])
            
            if torch.cuda.is_available():
                ipt, lbl = ipt.cuda(), lbl.cuda()

            with torch.cuda.amp.autocast():
              cell_logits, image_logits = model(ipt, n_cell)
              cell_probs = torch.sigmoid(cell_logits).cpu().numpy()
              image_probs = torch.sigmoid(image_logits).cpu().numpy()
              cell_logits = cell_logits.cpu().numpy()
              image_logits = image_logits.cpu().numpy()
            
            # Add cell-level output
            for j in range(int(n_cell)):
                results.append({
                    'filename': f'{filename}_{j+1}',
                    **{f'logit_{k}': cell_logits[j, k] for k in range(cell_logits.shape[1])},
                    **{f'prob_{k}': cell_probs[j, k] for k in range(cell_probs.shape[1])},
                    'type': 'cell'
                })

            # Add image-level output
            results.append({
                'filename': filename,
                **{f'logit_{k}': image_logits[0, k] for k in range(image_logits.shape[1])},
                **{f'prob_{k}': image_probs[0, k] for k in range(image_probs.shape[1])},
                'type': 'image'
            })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(save_file, index=False)
    print(f"Saved validation outputs to {save_file}")


if __name__ == '__main__':
  args = parser.parse_args()

  # Set global variables
  DEVICE = args.device
  print(
    f"Using device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE == 'cuda:0' else 'CPU'})")
  if DEVICE == "cpu":
    args.mixed_precision = False
  
  save_file = args.save_file

  # Dataset
  df = pd.read_csv(args.train_csv)

  # Valitation dataset
  vs = GetPredictionsDataset(
    df=df,
    tfms=None,
    cell_path=args.data_dir,
    cell_size=args.image_size
  )

  valid_loader = DataLoader(
    dataset=vs, 
    batch_size=args.validate_batch_size,
    shuffle=False,
    num_workers=args.num_workers, 
    pin_memory=False)

  # Network
  model = HPAClassifier(
    args.architecture,
    num_classes=19,
    channels=4,
    backbone_weights=args.backbone_weights,
    mode=args.mode,
    dilated=args.dilated,
    trainable_stem=args.trainable_stem,
    trainable_backbone=args.trainable_backbone,
  )
  if args.model_restore:
    print(f"[ i ] Restoring weights from {args.model_restore}")
    model.load_state_dict(torch.load(args.model_restore), strict=True)
  model = model.to(DEVICE)

  if GPUS_COUNT > 1:
    print(f"GPUs={GPUS_COUNT}")
    model = torch.nn.DataParallel(model)

  predict_model(
    model, valid_loader, save_file)
