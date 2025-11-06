import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Sampler
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
from lightly.loss import NTXentLoss

import albumentations as A

import cv2
cv2.setNumThreads(0)

import datasets
from datasets.hpa import SimCLRDataset

import wandb
from core.networks import *
import core.vision_transformer as vits
from tools.ai import ema as ema_mod
from tools.general.io_utils import *
from tools.general import wandb_utils
from tools.ai.optim_utils import *
from tools.ai.log_utils import *
from tools.general.time_utils import *


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)

        # similarity matrix
        similarity = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # remove self-comparisons
        logits_mask = torch.ones_like(similarity) - torch.eye(features.size(0), device=device)

        # build mask: samples are positives if they share >= 1 label
        mask = (labels @ labels.T) > 0   # (batch_size, batch_size)

        # exponentiated similarities (ignoring self-comparisons)
        exp_sim = torch.exp(similarity) * logits_mask

        # log-probabilities
        log_prob = similarity - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)

        # mean log-prob over positives
        # (avoid division by zero with clamp)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)

        # final loss
        loss = -mean_log_prob_pos.mean()
        return loss


def str2floatlist(arg):
    return [float(x.strip()) for x in arg.split(',')]


parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--debug', default=None, type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--sampler_seed', default=153, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--dataset', default='default', type=str)
parser.add_argument('--sampler', default='default', type=str)
parser.add_argument('--train_csv', required=True, type=str)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--image_size', default=512, type=int)


# Network
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--trainable-backbone', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--backbone_weights', default="imagenet", type=str)

# Hyperparameter
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument("--first_epoch", default=0, type=int)
parser.add_argument('--max_epoch', default=15, type=int)
parser.add_argument('--accumulate_steps', default=1, type=int)
parser.add_argument('--mixed_precision', default=False, type=str2bool)
parser.add_argument('--amp_min_scale', default=None, type=float)

parser.add_argument('--weakly_supervised', default=False, type=str2bool)
parser.add_argument('--temperature', default=0.07, type=float)
parser.add_argument('--optimizer', default="sgd", choices=OPTIMIZERS_NAMES)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--warmup_epochs', default=0, type=int)
parser.add_argument('--warmup_start_factor', default=0.01, type=float)
parser.add_argument('--print_ratio', default=0.1, type=float)
parser.add_argument('--monitor_memory_usage', default=False, type=str2bool)

# Normalization and data augmentation
parser.add_argument('--normalization_mean', default='0.485,0.456,0.406,0.406', type=str2floatlist)
parser.add_argument('--normalization_std', default='0.229,0.224,0.225,0.225', type=str2floatlist)
parser.add_argument('--aug_yaml', default='', type=str)

# Restore training
parser.add_argument('--model_restore', default=None, type=str)
parser.add_argument('--optimizer_restore', default=None, type=str)
parser.add_argument('--scheduler_restore', default=None, type=str)
parser.add_argument('--scaler_restore', default=None, type=str)
parser.add_argument('--train_meta_restore', default=None, type=str)

# Tag
parser.add_argument('--tag', default='', type=str)


try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
  print(f"GPUS={GPUS}")
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)


def get_transformations(aug_yaml):
    return A.load(aug_yaml, data_format='yaml')


def create_train_dataloader(
    args: argparse.Namespace, 
    train_dataset: SimCLRDataset
) -> DataLoader:
    """
    Creates a PyTorch DataLoader for the training set based on sampler arguments.

    Args:
        args (argparse.Namespace): A namespace object containing command-line arguments.
                                   Expected attributes include: sampler, batch_size,
                                   num_workers, sampler_seed.
        train_dataset (Dataset): The training dataset object.

    Returns:
        DataLoader: The configured DataLoader for the training set.
    """
    print(f'[ i ] Using default sampler: {args.sampler}')
    sampler, shuffle = datasets.get_train_sampler_and_shuffler(
        args.sampler,
        seed=args.sampler_seed
    )

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers, 
        pin_memory=False
    )
        
    return train_loader


def create_optimizer(
  args: argparse.Namespace,
  model: nn.Module):
   
  print(f"[ i ] Using regular optimizer.")
  optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=args.lr,
    weight_decay=args.wd
  )

  if args.optimizer_restore:
    print(f"[ i ] Restoring optimizer state from {args.optimizer_restore}")
    optimizer.load_state_dict(torch.load(args.optimizer_restore))

  return optimizer


def create_scheduler(
  args: argparse.Namespace,
  optimizer: torch.optim.Optimizer,
  warmup_steps: int,
  main_steps: int
):

  if args.scheduler_restore:
    print(f"[ i ] Restoring scheduler state from {args.scheduler_restore}")
    scheduler_state_dict = torch.load(args.scheduler_restore)

    print("[ i ] Scheduler state dict:")
    for key, value in scheduler_state_dict.items():
        print(f"  {key}: {value}")

  print(f"[ i ] Main steps: {main_steps}")
  main_scheduler = CosineAnnealingLR(
    optimizer, 
    T_max=main_steps)

  if warmup_steps > 0:
    print(f"[ i ] Warmup steps: {warmup_steps}")
    warmup_scheduler = LinearLR(
      optimizer, 
      start_factor=args.warmup_start_factor, 
      total_iters=warmup_steps)
    scheduler = SequentialLR(
      optimizer, 
      schedulers=[warmup_scheduler, main_scheduler], 
      milestones=[warmup_steps]
    )
  else:
    scheduler = main_scheduler
  
  if args.scheduler_restore:
    for e in range(scheduler_state_dict['last_epoch']):
      scheduler.step()
      
    print(f"Current scheduler epoch: {scheduler.last_epoch}")
    print(f"[ i ] Initial scheduler lr: {scheduler.get_last_lr()}")

  return scheduler


def build_simclr_model(
  args: argparse.Namespace, 
  projection_dim=128, 
  hidden_dim=2048
) -> nn.Module:
  """Builds the SimCLR model."""
  print("[ i ] Building SimCLR model.")

  if (args.backbone_weights).lower() == "none":
    args.backbone_weights = None

  backbone = Backbone(
    model_name=args.architecture,
    channels=4,
    weights=args.backbone_weights,
    mode=args.mode,
    dilated=args.dilated,
    trainable_stem=args.trainable_stem,
    trainable_backbone=args.trainable_backbone,
  )
  if args.backbone_weights == None:
    print("[ i ] Initializing backbone weights.")
    backbone.initialize(backbone.modules())

  projection_head = ProjectionHead(
    in_dim=backbone.out_dim, 
    out_dim=projection_dim, 
    hidden_dim=hidden_dim)

  model = SimCLRModel(backbone, projection_head)

  return model


if __name__ == '__main__':
  args = parser.parse_args()

  ###########################
  # Set global variables    #
  ###########################

  TAG = args.tag
  SEED = args.seed
  SAVE_EVERY_EPOCH = True
  DEVICE = args.device

  ###########################
  # Initial configuration   #
  ###########################

  # Seed
  set_seed(SEED)

  # Device
  print(
    f"Using device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE == 'cuda:0' else 'CPU'})")
  
  # Mixed precision not available on CPU
  if DEVICE == "cpu":
    args.mixed_precision = False

  # Set up WandB
  wb_run = wandb_utils.setup(TAG, args)
  log_config(vars(args), TAG)

  # Create directory model
  if os.path.isdir('./experiments/models/' + TAG):
    print(f"Model directory already exists: ./experiments/models/{TAG}")
    raise FileExistsError(
      f"Model directory already exists: ./experiments/models/{TAG}. "
      "Please change the tag or remove the existing directory.")
  
  model_dir = create_directory('./experiments/models/' + TAG + '/')
  model_path = model_dir + f'model.pth'

  ###########
  # Dataset #
  ###########

  train_df = pd.read_csv(args.train_csv)

  # Debugging
  if int(args.debug):
    train_df = train_df.sample(n=100, random_state=SEED)

  # Data transformations
  base_tfms = Compose([
    ToTensor(),  # Converts image to PyTorch tensor (C x H x W)
    Normalize(mean=args.normalization_mean, 
              std=args.normalization_std),  # Normalizes each channel
  ])
  if args.aug_yaml:
    print(f"Using augmentations from {args.aug_yaml}")
    aug_tfms = get_transformations(args.aug_yaml)

  # Train dataset
  ts = SimCLRDataset(
    df=train_df,
    base_tfms=base_tfms,
    aug_tfms=aug_tfms,
    cell_path=args.data_dir,
    cell_size=args.image_size
  )
  
  # Data loaders
  train_loader = create_train_dataloader(args, ts)
  train_iterator = datasets.Iterator(train_loader)
  log_loader(train_loader, ts, check_sampler=False)

  #########
  # Steps #
  #########

  step_val = len(train_loader)
  step_log = int(step_val * args.print_ratio)
  step_init = args.first_epoch * step_val
  step_max = args.max_epoch * step_val
  print(f"[ i ] Iterations: first={step_init} logging={step_log} validation={step_val} max={step_max}")

  ###############
  # Build model #
  ###############

  model = build_simclr_model(args)
  model = model.to(DEVICE)
  model.train()

  # Restore model weights
  if args.model_restore:
    print(f"[ i ] Restoring weights from {args.model_restore}")
    model.load_state_dict(torch.load(args.model_restore), strict=True)

  if GPUS_COUNT > 1:
    print(f"GPUs={GPUS_COUNT}")
    model = torch.nn.DataParallel(model)

  ####################
  # Contrastive loss #
  ####################

  if args.weakly_supervised:
    print("[ i ] Using weakly supervised contrastive loss.")
    criterion = SupConLoss(temperature=args.temperature)
  else:
    print("[ i ] Using standard NT-Xent loss.")
    criterion = NTXentLoss(temperature=args.temperature)
  criterion = criterion.to(DEVICE)

  #############
  # Optimizer #
  #############

  optimizer = create_optimizer(args, model)

  ###################
  # Mixed precision #
  ###################

  scaler = torch.amp.GradScaler(DEVICE, enabled=args.mixed_precision)
  if args.scaler_restore:
    print(f"[ i ] Restoring scaler state from {args.scaler_restore}")
    scaler.load_state_dict(torch.load(args.scaler_restore))
  
  ##############
  # Schedulers #
  ##############

  warmup_steps = args.warmup_epochs * int(step_val // args.accumulate_steps)
  main_steps = (step_max - step_init) // args.accumulate_steps - warmup_steps
  scheduler = create_scheduler(args, optimizer, warmup_steps, main_steps)

  ##############################
  # Optimizer and scheduler lr #
  ##############################

  print(f"[ i ] Initial optimizer lr: {scheduler.get_last_lr()}")

  #########
  # Train #
  #########

  if args.train_meta_restore:
    print(f"[ i ] Restoring training meta from {args.train_meta_restore}")

    training_meta = torch.load(args.train_meta_restore)
    step_init = training_meta['step'] + 1
    args.first_epoch = training_meta['epoch'] + 1
    
    print(f"[ i ] Restored step={step_init}, epoch={args.first_epoch}")

  train_meter = MetricsContainer(['contrastive_loss']) # Use a new metric name
  train_timer = Timer()

  tqdm_bar = tqdm(range(step_init, step_max), 'SimCLR Training', mininterval=2.0)
  for step in tqdm_bar:
    ((view1, view2), labels) = train_iterator.get()

    images = torch.cat([view1, view2], dim=0).to(DEVICE)
    labels = labels.repeat(2, 1) if labels.ndim == 2 else labels.repeat(2)
    labels = labels.to(DEVICE)
    # print(f"Labels shape: {labels.shape}")

    with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
        projections = model(images)
        
        if args.weakly_supervised:
            loss = criterion(projections, labels)
        else:
            z_i = projections[:args.batch_size]
            z_j = projections[args.batch_size:]
            loss = criterion(z_i, z_j)

    scaler.scale(loss).backward()

    if (step + 1) % args.accumulate_steps == 0:
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad()

    loss_value = loss.detach().cpu().item()
    train_meter.update({'contrastive_loss': loss_value})
    
    # Update your progress bar and logging
    epoch = step // step_val
    epoch_loss = train_meter.get()
    learning_rate = float(scheduler.get_last_lr()[0])
    
    tqdm_bar.set_description(
        f"[epoch={epoch} loss={epoch_loss:.5f} lr={learning_rate:.5f}]"
    )
    
    # Log to WandB
    if (step + 1) % step_log == 0:
        lrs = scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            wandb.log({f"train/lr_group_{i}": lr, "train/epoch": epoch})
        wandb.log({
            "train/contrastive_loss": epoch_loss,
            "train/epoch": epoch
        })

    # Save checkpoints periodically (e.g., at the end of each epoch)
    is_val_step = (step + 1) % step_val == 0
    if is_val_step:
        # NOTE: NO VALIDATION! Just save the model.
        model_path = model_dir + f'model-e{epoch}.pth'
        print(f"[ i ] Saving model checkpoint to {model_path}")
        save_model(model, model_path, parallel=GPUS_COUNT > 1)
        train_meter.clear()

  print(TAG)
  wb_run.finish()
