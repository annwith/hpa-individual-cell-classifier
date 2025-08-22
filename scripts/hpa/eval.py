import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import albumentations as A
from tqdm import tqdm

import cv2
cv2.setNumThreads(0)

import datasets
from datasets.hpa import ConfAwareHPADataset

import wandb
from core.networks import *
from tools.ai import ema as ema_mod
from tools.ai.augment_utils import *
from tools.ai.demo_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.log_utils import *
from tools.ai.optim_utils import *
from tools.ai.randaugment import *
from tools.ai.torch_utils import *
from tools.general import wandb_utils
from tools.general.io_utils import *
from tools.general.time_utils import *

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--debug', default=None, type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--sampler_seed', default=153, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--sampler', default="default", type=str)
parser.add_argument('--train_csv', required=True, type=str)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--conf_aware_training', default=False, type=str2bool)
parser.add_argument('--conf_preds', default=None, type=str)
parser.add_argument('--conf_alpha', default=1.0, type=float)
parser.add_argument('--conf_gamma', default=1.0, type=float)
parser.add_argument('--val_fold', default=0, type=int)
parser.add_argument('--validate_batch_size', default=32, type=int)
parser.add_argument('--cell_count', default=16, type=int)
parser.add_argument('--image_size', default=512, type=int)


# Network
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--trainable-backbone', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--backbone_weights', default="imagenet", type=str)
parser.add_argument('--cell_logits_to_image_logits', default=False, type=str2bool)

# Hyperparameter
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument("--first_epoch", default=0, type=int)
parser.add_argument('--max_epoch', default=15, type=int)
parser.add_argument('--accumulate_steps', default=1, type=int)
parser.add_argument('--mixed_precision', default=False, type=str2bool)
parser.add_argument('--amp_min_scale', default=None, type=float)
parser.add_argument('--validate', default=True, type=str2bool)

parser.add_argument('--optimizer', default="sgd", choices=OPTIMIZERS_NAMES)
parser.add_argument('--poly_lr_decay', default=False, type=str2bool)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--lr_alpha_scratch', default=10., type=float)
parser.add_argument('--lr_alpha_bias', default=2., type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--warmup_epochs', default=0, type=int)
parser.add_argument('--warmup_start_factor', default=0.01, type=float)
parser.add_argument('--label_smoothing', default=0, type=float)
parser.add_argument('--class_weight', default=None, type=str)
parser.add_argument('--cell_pos_weight', default=1.0, type=float)
parser.add_argument('--cell_loss_weight', default=1.0, type=float)
parser.add_argument('--ema', default=False, type=str2bool)
parser.add_argument('--ema_steps', default=32, type=int)
parser.add_argument('--ema_warmup', default=1, type=int)
parser.add_argument('--ema_decay', default=0.99, type=float)

parser.add_argument('--print_ratio', default=0.1, type=float)
parser.add_argument('--monitor_memory_usage', default=False, type=str2bool)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--aug_yaml', default='', type=str)

# Restore training
parser.add_argument('--model_restore', default=None, type=str)
parser.add_argument('--optimizer_restore', default=None, type=str)
parser.add_argument('--scheduler_restore', default=None, type=str)
parser.add_argument('--scaler_restore', default=None, type=str)
parser.add_argument('--train_meta_restore', default=None, type=str)


try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
  print(f"GPUS={GPUS}")
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)


def validate_model(
    model, 
    valid_dl, 
    args):
    
    # Move model to device
    model.to(DEVICE)

    # Set model to evaluation mode
    model.eval()

    # Set tqdm progress bar
    tq = tqdm(valid_dl)

    with torch.no_grad():
        results = []
        losses, predicted, truth = [], [], []
        for i, (ipt, lbl, img_lbl, n_cell) in enumerate(tq):

            ipt = ipt.view(-1, ipt.shape[-3], ipt.shape[-2], ipt.shape[-1])
            img_lbl = img_lbl.view(-1, 19)
            ipt, img_lbl = ipt.to(DEVICE), img_lbl.to(DEVICE)

            # Get logits and loss
            with torch.amp.autocast(DEVICE):
                _, output = model(ipt, n_cell)
                loss = F.binary_cross_entropy_with_logits(
                    output, img_lbl,
                    reduction='none')
                if not len(loss.shape) == 0:
                    loss = loss.mean()
                output = output.float()
            
            # Append loss to list
            losses.append(loss.item())

            # Predictions
            pred = torch.sigmoid(output.cpu()).numpy()

            # Append to lists
            predicted.append(pred)
            truth.append(img_lbl.cpu().numpy())
            
            results.append({
                'step': i,
                'loss': loss.item(),
            })
        
        # Concatenate results and calculate validation loss
        predicted = np.concatenate(predicted)
        truth = np.concatenate(truth)
        val_loss = np.array(losses).mean()
        
        # Classification report
        predicted_binary = (predicted > 0.5).astype(int)
        report = classification_report(
            truth, 
            predicted_binary, 
            output_dict=True,
            zero_division=0)

        # Convert to DataFrame for nicer formatting
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.round(4)

        return val_loss, report_df


if __name__ == '__main__':
  args = parser.parse_args()

  # Set global variables
  TAG = args.tag
  SEED = args.seed
  set_seed(SEED)

  DEVICE = args.device
  print(
    f"Using device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE == 'cuda:0' else 'CPU'})")
  if DEVICE == "cpu":
    args.mixed_precision = False

  # Dataset
  df = pd.read_csv(args.train_csv)
  valid_df = df[df.fold == args.val_fold]

  # Valitation dataset
  if args.validate:
    vs = ConfAwareHPADataset(
      df=valid_df,
      tfms=None,
      cell_path=args.data_dir,
      cell_count=args.cell_count,
      cell_size=args.image_size,
      label_smoothing=args.label_smoothing,
      mode='valid'
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

  if GPUS_COUNT > 1:
    print(f"GPUs={GPUS_COUNT}")
    model = torch.nn.DataParallel(model)

  val_loss, report_df = validate_model(
    model, valid_loader, args)
  
  print(f"[ i ] Validation loss: {val_loss:.4f}")
  print(f"[ i ] Validation report:\n{report_df}")
      