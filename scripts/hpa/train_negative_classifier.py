import argparse
import os

import psutil
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torch.nn.functional as F
import albumentations as A
from tqdm import tqdm

import cv2
cv2.setNumThreads(0)

import datasets
from datasets.hpa2ranzer import NegativeClassifierDataset

import wandb
from core.networks import *
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
parser.add_argument('--dataset', type=str)
parser.add_argument('--sampler', default="default", type=str)
parser.add_argument('--train_csv', required=True, type=str)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--conf_aware_training', default=False, type=str2bool)
parser.add_argument('--conf_preds', default=None, type=str)
parser.add_argument('--conf_alpha', default=1.0, type=float)
parser.add_argument('--conf_gamma', default=1.0, type=float)
parser.add_argument('--validate_batch_size', default=32, type=int)
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
parser.add_argument('--first_epoch', default=0, type=int)
parser.add_argument('--max_epoch', default=15, type=int)
parser.add_argument('--accumulate_steps', default=1, type=int)
parser.add_argument('--mixed_precision', default=False, type=str2bool)
parser.add_argument('--amp_min_scale', default=None, type=float)
parser.add_argument('--validate', default=True, type=str2bool)

parser.add_argument('--optimizer', default="sgd", choices=OPTIMIZERS_NAMES)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--lr_alpha_scratch', default=10., type=float)
parser.add_argument('--lr_alpha_bias', default=2., type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--warmup_epochs', default=0, type=int)
parser.add_argument('--warmup_start_factor', default=0.01, type=float)
parser.add_argument('--label_smoothing', default=0, type=float)

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
THRESHOLDS = list(np.arange(0.10, 0.50, 0.05))


def get_transformations(aug_yaml):
    return A.load(aug_yaml, data_format='yaml')


def validate_model(
    model, 
    valid_dl, 
    args):
    
    model.to(DEVICE)
    model.eval()

    tq = tqdm(valid_dl)
    with torch.no_grad():
        results = []
        losses, predicted, truth = [], [], []
        for i, (images, labels) in enumerate(tq):
            labels = labels.float()
            
            # Send to device
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            labels = labels.view(-1, 1)  # or labels = labels.unsqueeze(1)

            # Get logits and loss 
            logits = model(images)

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("NaN/Inf in logits at step", i)
                print(torch.max(logits), torch.min(logits))
                continue
            if torch.isnan(labels).any() or torch.isinf(labels).any():
                print("NaN/Inf in labels at step", i)
                continue

            loss = F.binary_cross_entropy_with_logits(
                logits, labels,
                reduction='mean')
            if torch.isnan(loss):
                print(f"[!] NaN loss at step {i}")
                continue
            losses.append(loss.item())

            pred = torch.sigmoid(logits.cpu()).numpy()
            predicted.append(pred)
            truth.append(labels.cpu().numpy())

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
  SAVE_EVERY_EPOCH = True

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

  # Dataset
  df = pd.read_csv(args.train_csv)

  # Dataset splitting
  if args.validate:
    print(f"[ i ] Splitting dataset into train and validation sets.")

    unique_images = df['image_filename'].unique()
    train_imgs, val_imgs = train_test_split(
        unique_images, test_size=0.2, random_state=SEED
    )

    train_df = df[df['image_filename'].isin(train_imgs)]
    valid_df = df[df['image_filename'].isin(val_imgs)]

    print(f"Train: {len(train_df)} rows, {len(train_imgs)} images")
    print(f"Val: {len(valid_df)} rows, {len(val_imgs)} images")

  if int(args.debug):
    train_df = train_df.sample(n=4000, random_state=SEED)
    if args.validate:
      valid_df = valid_df.sample(n=1000, random_state=SEED)

  if args.aug_yaml:
    print(f"Using augmentations from {args.aug_yaml}")
    train_tfms = get_transformations(args.aug_yaml)

  # Train dataset
  ts = NegativeClassifierDataset(
    df=train_df,
    tfms=train_tfms,
    cell_path=args.data_dir,
    cell_size=args.image_size,
    conf_aware=args.conf_aware_training,
    conf_path=args.conf_preds,
    label_smoothing=args.label_smoothing,
    mode='train'
  )

  # Valitation dataset
  if args.validate:
    vs = NegativeClassifierDataset(
      df=valid_df,
      tfms=None,
      cell_path=args.data_dir,
      cell_size=args.image_size,
      label_smoothing=args.label_smoothing,
      mode='valid'
    )

  train_loader = DataLoader(
    dataset=ts, 
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=False)
  
  if args.validate:
    valid_loader = DataLoader(
      dataset=vs, 
      batch_size=args.validate_batch_size,
      shuffle=False,
      num_workers=args.num_workers, 
      pin_memory=False)

  train_iterator = datasets.Iterator(train_loader)
  log_loader(train_loader, ts, check_sampler=False)

  # Steps
  step_val = len(train_loader)
  step_log = int(step_val * args.print_ratio)
  step_init = args.first_epoch * step_val
  step_max = args.max_epoch * step_val
  print(f"[ i ] Iterations: first={step_init} logging={step_log} validation={step_val} max={step_max}")

  # Network
  model = NegativeClassifier(
    args.architecture,
    num_classes=1,
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

  param_groups, param_names = model.get_parameter_groups(with_names=True)
  model.train()

  model = model.to(DEVICE)

  if GPUS_COUNT > 1:
    print(f"GPUs={GPUS_COUNT}")
    model = torch.nn.DataParallel(model)

  print(f"[ i ] Using regular optimizer.")
  optimizer = get_regular_optimizer(
    args.lr, args.wd, param_groups,
    algorithm=args.optimizer,
    alpha_scratch=args.lr_alpha_scratch,
    alpha_bias=args.lr_alpha_bias,
  )
  if args.optimizer_restore:
    print(f"[ i ] Restoring optimizer state from {args.optimizer_restore}")
    optimizer.load_state_dict(torch.load(args.optimizer_restore))

    print("Optimizer LR:", optimizer.param_groups[0]["lr"])
  log_opt_params("Vanilla", param_names)

  # Mixed precision
  scaler = torch.amp.GradScaler(DEVICE, enabled=args.mixed_precision)
  if args.scaler_restore:
    print(f"[ i ] Restoring scaler state from {args.scaler_restore}")
    scaler.load_state_dict(torch.load(args.scaler_restore))
  
  # Schedulers
  if args.scheduler_restore:
    print(f"[ i ] Restoring scheduler state from {args.scheduler_restore}")
    scheduler_state_dict = torch.load(args.scheduler_restore)

    print("[ i ] Scheduler state dict:")
    for key, value in scheduler_state_dict.items():
        print(f"  {key}: {value}")
    
  print(f"[ i ] Using warmup and cosine annealing scheduler.")

  warmup_steps = args.warmup_epochs * int(step_val // args.accumulate_steps)
  print(f"[ i ] Warmup steps: {warmup_steps}")
  warmup_scheduler = LinearLR(
    optimizer, 
    start_factor=args.warmup_start_factor, 
    total_iters=warmup_steps)
  
  main_steps = args.max_epoch * int(step_val // args.accumulate_steps) - warmup_steps
  print(f"[ i ] Main steps: {main_steps}")
  main_scheduler = CosineAnnealingLR(
    optimizer, 
    T_max=main_steps)

  scheduler = SequentialLR(
      optimizer, 
      schedulers=[warmup_scheduler, main_scheduler], 
      milestones=[warmup_steps])
  
  if args.scheduler_restore:
    for e in range(scheduler_state_dict['last_epoch']):
      scheduler.step()
    print(f"Current scheduler epoch: {scheduler.last_epoch}")
    print(f"[ i ] Initial scheduler lr: {scheduler.get_last_lr()}")

  # Optimizer and scheduler lr
  print(f"[ i ] Initial optimizer lr: {get_learning_rate_from_optimizer(optimizer)}")

  if args.train_meta_restore:
    print(f"[ i ] Restoring training meta from {args.train_meta_restore}")
    training_meta = torch.load(args.train_meta_restore)
    step_init = training_meta['step'] + 1
    args.first_epoch = training_meta['epoch'] + 1
    print(f"[ i ] Restored step={step_init}, epoch={args.first_epoch}")

  # Train
  train_meter = MetricsContainer(['loss'])
  train_timer = Timer()

  tqdm_bar = tqdm(range(step_init, step_max), 'Training', mininterval=2.0)
  for step in tqdm_bar:
    images, labels = train_iterator.get()
    labels = labels.float()

    # Send to device
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    labels = labels.view(-1, 1)  # or labels = labels.unsqueeze(1)

    # DEBUG: print values and shapes
    # print(f"images shape: {images.shape}, dtype: {images.dtype}")
    # print(f"labels shape: {labels.shape}, dtype: {labels.dtype}")

    with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
      logits = model(images)
      loss = F.binary_cross_entropy_with_logits(
                                logits, labels,
                                reduction='none')
      
      if args.conf_aware_training:
          raise NotImplementedError(
            "Confidence-aware training is not implemented yet. "
            "Please set --conf_aware_training=False or implement the logic.")
          conformity = 1 - torch.abs(labels - conf)
          w = args.conf_alpha * conformity ** args.conf_gamma
          weighted_loss = loss * w
      else:
          weighted_loss = loss

      # Calculate mean if needed
      if not len(weighted_loss.shape) == 0:
          weighted_loss = weighted_loss.mean()

    scaler.scale(weighted_loss).backward()

    if (step + 1) % args.accumulate_steps == 0:
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad()
      scheduler.step()

    weighted_loss = weighted_loss.detach().cpu().item()
    train_meter.update({'loss': weighted_loss})

    epoch = step // step_val
    is_log_step = (step + 1) % step_log == 0
    is_val_step = (step + 1) % step_val == 0

    epoch_loss = train_meter.get()
    learning_rate = float(get_learning_rate_from_optimizer(optimizer))
  
    if args.monitor_memory_usage:
      # Get CPU and RAM usage
      cpu_mem = psutil.virtual_memory()
      cpu_mem_used = cpu_mem.used / (1024 ** 3)  # Convert to GB
      cpu_mem_free = cpu_mem.available / (1024 ** 3)  # Convert to GB

      # Get GPU usage
      gpu_mem_used = torch.cuda.memory_allocated(0) / (1024 ** 3)
      gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
      gpu_mem_free = (
          torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
      ) / (1024 ** 3)
      tqdm_bar.set_description(
        f"[epoch={epoch} loss={epoch_loss:.5f} "
        f"lr={learning_rate:.5f} cpu={cpu_mem_used:.2f}/{cpu_mem_free:.2f} GB "
        f"gpu={gpu_mem_used:.2f}/{gpu_mem_reserved:.2f}/{gpu_mem_free:.2f} GB]")
    else:
      tqdm_bar.set_description(
        f"[epoch={epoch} loss={epoch_loss:.5f} "
        f"lr={learning_rate:.5f}]")

    if is_log_step:
      data = {
        'iteration': step + 1,
        'learning_rate': learning_rate,
        'loss': weighted_loss,
        'epoch_loss': epoch_loss,
        'time': train_timer.tok(clear=True),
      }
      wb_logs = {f"train/{k}": v for k, v in data.items()}
      wb_logs["train/epoch"] = epoch
      wandb.log(wb_logs, commit=not (args.validate and is_val_step))

    if args.validate and is_val_step:
      val_loss, report_df = validate_model(
        model, valid_loader, args)
      
      val_data = {
        'iteration': step + 1,
        'val_loss': val_loss,
        'val_classification_report': wandb.Table(dataframe=report_df)
      }
      wb_logs = {f"val/{k}": v for k, v in val_data.items()}
      wb_logs["val/epoch"] = epoch
      wandb.log(wb_logs, commit=True)
      
      print(
        f'step={step+1:,} '
        f'val_loss={val_loss:.4f} '
      )
      print(report_df)
      
    if is_val_step:
      if SAVE_EVERY_EPOCH:
        model_path = model_dir + f'model-e{epoch}.pth'
        print(f"[ i ] Saving model to {model_path}")
        save_model(model, model_path, parallel=GPUS_COUNT > 1)

      torch.save(optimizer.state_dict(), model_dir + 'optimizer.pth')
      torch.save(scheduler.state_dict(), model_dir + 'scheduler.pth')
      torch.save(scaler.state_dict(), model_dir + 'scaler.pth')
      torch.save({'epoch': epoch, 'step': step}, model_dir + 'training_meta.pth')

      train_meter.clear()

  print(TAG)
  wb_run.finish()
