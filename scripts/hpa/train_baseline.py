import os
import argparse

import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize

import cv2
cv2.setNumThreads(0)

import datasets
from datasets.hpa import HPABaseline

from train_simclr_ws import SupConLoss
from core.networks import *
from tools.ai import ema as ema_mod
from tools.general.io_utils import create_directory, str2floatlist, str2bool
from tools.general.time_utils import Timer
from tools.general import wandb_utils
from tools.ai.optim_utils import get_optimizer, get_regular_optimizer, \
  get_learning_rate_from_optimizer, OPTIMIZERS_NAMES
from tools.ai.log_utils import log_config, log_loader, log_opt_params, \
  get_memory_usage_GB, MetricsContainer
from tools.ai.torch_utils import set_seed, save_model


parser = argparse.ArgumentParser()

# -----------------------------------------------
# Dataset hyperparameters
# -----------------------------------------------
parser.add_argument('--debug', default=None, type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--sampler_seed', default=153, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--dataset', default='default', type=str)
parser.add_argument('--sampler', default='default', type=str)
parser.add_argument('--train_csv', required=True, type=str)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--image_conf_aware_training', default=False, type=str2bool)
parser.add_argument('--conf_preds', default=None, type=str)
parser.add_argument('--conf_alpha', default=1.0, type=float)
parser.add_argument('--conf_gamma', default=1.0, type=float)
parser.add_argument('--val_fold', default=0, type=int)
parser.add_argument('--validate_batch_size', default=32, type=int)
parser.add_argument('--image_size', default=512, type=int)

# -----------------------------------------------
# Network hyperparameters
# -----------------------------------------------
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--trainable-backbone', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--backbone_weights', default="imagenet", type=str)
parser.add_argument('--is_simclr_model', default=False, type=str2bool)
parser.add_argument('--is_dino_model', default=False, type=str2bool)
parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')

# -----------------------------------------------
# Training hyperparameters
# -----------------------------------------------
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
parser.add_argument('--pos_weight', default=1.0, type=float)
parser.add_argument('--ema', default=False, type=str2bool)
parser.add_argument('--ema_steps', default=32, type=int)
parser.add_argument('--ema_warmup', default=1, type=int)
parser.add_argument('--ema_decay', default=0.99, type=float)

parser.add_argument('--print_ratio', default=0.1, type=float)
parser.add_argument('--monitor_memory_usage', default=False, type=str2bool)

# -----------------------------------------------
# SupCon hyperparameters
# -----------------------------------------------
parser.add_argument('--is_supconclassifier_model', default=False, type=str2bool)
parser.add_argument('--supcon_temperature', default=0.07, type=float)
parser.add_argument('--supcon_alpha', default=0.8, type=float)
parser.add_argument('--supcon_use_hard_mask', default=True, type=str2bool)
parser.add_argument('--bce_loss_weight', default=1.0, type=float)
parser.add_argument('--supcon_loss_weight', default=1.0, type=float)
parser.add_argument('--projection_dim', default=128, type=int)
parser.add_argument('--hidden_dim', default=2048, type=int)

# -----------------------------------------------
# Normalization and data augmentation settings
# -----------------------------------------------
parser.add_argument('--normalization_mean', default='0.485,0.456,0.406,0.406', type=str2floatlist)
parser.add_argument('--normalization_std', default='0.229,0.224,0.225,0.225', type=str2floatlist)
parser.add_argument('--aug_yaml', default='', type=str)

# -----------------------------------------------
# Restore training from previous checkpoint
# -----------------------------------------------
parser.add_argument('--model_restore', default=None, type=str)
parser.add_argument('--optimizer_restore', default=None, type=str)
parser.add_argument('--scheduler_restore', default=None, type=str)
parser.add_argument('--scaler_restore', default=None, type=str)
parser.add_argument('--train_meta_restore', default=None, type=str)

# -----------------------------------------------
# Tag
# -----------------------------------------------
parser.add_argument('--tag', default='', type=str)

# -----------------------------------------------
# GPU
# -----------------------------------------------
try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
  print(f"GPUS={GPUS}")
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)


# -----------------------------------------------
# Helper functions
# -----------------------------------------------


def find_best_thresholds(y_true, y_pred):
    n_classes = y_true.shape[1]
    thresholds = []

    for c in range(n_classes):
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.05, 0.95, 50):
            f1 = f1_score(
              y_true[:, c], 
              (y_pred[:, c] > t).astype(int),
              zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds.append(best_t)

    return np.array(thresholds)


def validate_model(
    model, 
    valid_dl, 
    args):
    
    # Move model to device
    model.to(DEVICE)

    # Set model to evaluation mode
    model.eval()

    # Set tqdm progress bar
    tq = tqdm(valid_dl, "Validation", mininterval=1.0, ncols=80)
    with torch.no_grad():
        results = []
        losses, predicted, truth = [], [], []
        for i, (ipt, img_lbl) in enumerate(tq):
            ipt, img_lbl = ipt.to(DEVICE), img_lbl.to(DEVICE)

            # Get logits and loss
            with torch.amp.autocast(device_type=DEVICE.type):
                if args.is_supconclassifier_model:
                    _, output, _ = model(ipt)
                else:
                    output = model(ipt)
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
        thresholds = find_best_thresholds(truth, predicted)
        predicted_binary = (predicted > thresholds).astype(int)
        report = classification_report(
            truth, 
            predicted_binary, 
            output_dict=True,
            zero_division=0)

        # Convert to DataFrame for nicer formatting
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.round(4)

        return val_loss, report_df


def create_datasets(args: argparse.Namespace):
  # Read CSV and split
  df = pd.read_csv(args.train_csv)
  train_df, valid_df = (df[df.fold != args.val_fold],
                        df[df.fold == args.val_fold])

  # Debugging
  if int(args.debug):
    train_df = train_df.sample(n=400, random_state=SEED)
    if args.validate:
      valid_df = valid_df.sample(n=100, random_state=SEED)

  # Data transformations
  base_tfms = Compose([
    ToTensor(),  # Converts image to PyTorch tensor (C x H x W)
    Normalize(mean=args.normalization_mean, 
              std=args.normalization_std),  # Normalizes each channel
  ])
  if args.aug_yaml:
    print(f"Using augmentations from {args.aug_yaml}")
    aug_tfms = datasets.get_transformations(args.aug_yaml)

  # Check if training is confidence-aware
  conf_aware_training = args.image_conf_aware_training

  # Train dataset
  ts = HPABaseline(
    df=train_df,
    base_tfms=base_tfms,
    aug_tfms=aug_tfms,
    image_path=args.data_dir,
    image_size=args.image_size,
    conf_aware=conf_aware_training,
    conf_path=args.conf_preds,
    mode='train'
  )

  # Validation dataset
  vs = None
  if args.validate:
    vs = HPABaseline(
      df=valid_df,
      base_tfms=base_tfms,
      aug_tfms=None,
      image_path=args.data_dir,
      image_size=args.image_size,
      mode='valid'
    )

  return ts, vs


def create_train_valid_dataloaders(args: argparse.Namespace):
  # Sampler and DataLoader
  sampler, shuffle = datasets.get_train_sampler_and_shuffler(
    sampler=args.sampler,
    source=ts,
    seed=args.sampler_seed,
    clip_value=10
    )

  train_loader = DataLoader(
    dataset=ts, 
    batch_size=args.batch_size,
    shuffle=shuffle,
    sampler=sampler,
    num_workers=args.num_workers, 
    pin_memory=False)
  
  valid_loader = None
  if args.validate:
    valid_loader = DataLoader(
      dataset=vs, 
      batch_size=args.validate_batch_size,
      shuffle=False,
      num_workers=args.num_workers, 
      pin_memory=False)
  
  return train_loader, valid_loader


def create_model(args: argparse.Namespace):
  # SupConClassifier model
  if args.is_supconclassifier_model:
      print("Creating SupConClassifier model.")
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
      print(f"[ i ] Backbone output dimension: {backbone.out_dim}")
      if args.backbone_weights == None:
        print("[ i ] Initializing backbone weights.")
        backbone.initialize(backbone.modules())

      projection_head = ProjectionHead(
        in_dim=backbone.out_dim, 
        out_dim=args.projection_dim, 
        hidden_dim=args.hidden_dim)
      
      model = JointSupConClassifier(
        num_classes=19,
        backbone=backbone,
        projection_head=projection_head)
      
      if args.model_restore:
        print(f"[ i ] Restoring weights from previous training {args.model_restore}")
        model.load_state_dict(torch.load(args.model_restore), strict=True)
  else:
    model = Classifier(
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
    print(f"[ i ] Restoring weights from previous training {args.model_restore}")
    model.load_state_dict(torch.load(args.model_restore), strict=True)
  
  return model


def create_optimizer(
  args: argparse.Namespace, param_groups, param_names, step_init, step_max):
  if args.poly_lr_decay:
    print(f"[ i ] Using polynomial learning rate decay.")
    optimizer = get_optimizer(
      args.lr, args.wd, int(step_max // args.accumulate_steps), param_groups,
      algorithm=args.optimizer,
      alpha_scratch=args.lr_alpha_scratch,
      alpha_bias=args.lr_alpha_bias,
      start_step=int(step_init // args.accumulate_steps),
    )
  else:
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
    print("[ i ] Optimizer LR:", optimizer.param_groups[0]["lr"])
  
  log_opt_params("Vanilla", param_names, verbose=2)
  
  return optimizer


def create_scaler(args: argparse.Namespace):
  try:
    scaler = torch.amp.GradScaler(DEVICE, enabled=args.mixed_precision)
    print("[ i ] Using torch.amp.GradScaler for mixed precision.", DEVICE)
  except AttributeError:
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
    print("[ i ] Using torch.cuda.amp.GradScaler for mixed precision.", DEVICE)
    
  if args.scaler_restore:
    print(f"[ i ] Restoring scaler state from {args.scaler_restore}")
    scaler.load_state_dict(torch.load(args.scaler_restore))
  return scaler


def create_scheduler(
  args: argparse.Namespace, optimizer, step_init, step_max, step_val, warmup_steps):
  
  if args.scheduler_restore:
    print(f"[ i ] Restoring scheduler state from {args.scheduler_restore}")
    scheduler_state_dict = torch.load(args.scheduler_restore)

    print("[ i ] Scheduler state dict:")
    for key, value in scheduler_state_dict.items():
        print(f"  {key}: {value}")
    
  if not args.poly_lr_decay:
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

  return scheduler


def smooth_labels(y, smoothing=0.1):
    n = y.size(1)
    return (1 - smoothing) * y + smoothing / n


if __name__ == '__main__':
  args = parser.parse_args()

  # -----------------------------------------------
  # Set global variables
  # -----------------------------------------------

  TAG = args.tag
  SEED = args.seed
  SAVE_EVERY_EPOCH = False
  SAVE_BEST_VAL_LOSS = True
  set_seed(SEED)

  # Set device
  DEVICE = args.device
  if DEVICE == "cpu":
    args.mixed_precision = False
    print("[ i ] Using CPU for training. Disabling mixed precision.")
  else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ i ] Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")

  # Set class weights
  if args.class_weight and args.class_weight != "none":
    CLASS_WEIGHT = torch.Tensor(list(map(float, args.class_weight.split(",")))).to(DEVICE)
  else:
    CLASS_WEIGHT = None

  # Positive weight for cell classification
  pos_weight = torch.tensor([args.pos_weight] * 19).to(DEVICE)
  print(f"[ i ] Loss positive weight: {pos_weight}")

  # -----------------------------------------------
  # Initial configuration
  # -----------------------------------------------

  # Set up WandB
  wb_run = wandb_utils.setup(TAG, args)
  log_config(vars(args), TAG)

  # Create directory model
  if os.path.isdir('./experiments/models/' + TAG):
    print(f"Model directory already exists: ./experiments/models/{TAG}")
    raise FileExistsError(
      f"Model directory already exists: ./experiments/models/{TAG}. "
      "Please change the tag or remove the existing directory.")
  
  # Set model directory
  model_dir = create_directory('./experiments/models/' + TAG + '/')
  model_path = model_dir + f'model-f{args.val_fold}.pth'

  # Define SupCon loss if needed
  if args.is_supconclassifier_model:
    print(f"[ i ] Using SupConClassifier architecture.")
    supcon_loss = SupConLoss(
      temperature=args.supcon_temperature,
      alpha=args.supcon_alpha,
      use_hard_mask=args.supcon_use_hard_mask)

  # -----------------------------------------------
  # Dataset
  # -----------------------------------------------

  ts, vs = create_datasets(args)
  train_loader, valid_loader = create_train_valid_dataloaders(args)
  train_iterator = datasets.Iterator(train_loader)
  log_loader(train_loader, ts, check_sampler=False)

  # -----------------------------------------------
  # Steps
  # -----------------------------------------------

  step_val = len(train_loader)
  step_log = int(step_val * args.print_ratio)
  step_init = args.first_epoch * step_val
  step_max = args.max_epoch * step_val
  warmup_steps = args.warmup_epochs * int(step_val // args.accumulate_steps)
  ema_warmup_steps = args.ema_warmup * int(step_val // args.accumulate_steps)
  print(f"[ i ] Iterations: first={step_init} logging={step_log} validation={step_val} max={step_max}")

  # -----------------------------------------------
  # Build model
  # -----------------------------------------------

  model = create_model(args)
  param_groups, param_names = model.get_parameter_groups(with_names=True)
  model = model.to(DEVICE)
  model.train()

  if args.ema:
    ema_model = ema_mod.init(model, DEVICE, args.ema)

  if GPUS_COUNT > 1:
    print(f"GPUs={GPUS_COUNT}")
    model = torch.nn.DataParallel(model)
    if args.ema:
      ema_model = torch.nn.DataParallel(ema_model)

  # -----------------------------------------------
  # Optimizer
  # -----------------------------------------------

  optimizer = create_optimizer(args, param_groups, param_names, step_init, step_max)

  # -----------------------------------------------
  # Mixed precision
  # -----------------------------------------------

  scaler = create_scaler(args)
  
  # -----------------------------------------------
  # Schedulers
  # -----------------------------------------------

  scheduler = create_scheduler(args, optimizer, step_init, step_max, step_val, warmup_steps)

  # -----------------------------------------------
  # Optimizer and scheduler lr
  # -----------------------------------------------

  print(f"[ i ] Initial optimizer lr: {get_learning_rate_from_optimizer(optimizer)}")

  if args.train_meta_restore:
    print(f"[ i ] Restoring training meta from {args.train_meta_restore}")
    training_meta = torch.load(args.train_meta_restore)
    step_init = training_meta['step'] + 1
    args.first_epoch = training_meta['epoch'] + 1
    print(f"[ i ] Restored step={step_init}, epoch={args.first_epoch}")

  # -----------------------------------------------
  # Train
  # -----------------------------------------------

  best_val_loss = float('inf')
  train_meter = MetricsContainer(['loss'])
  train_timer = Timer()

  tqdm_bar = tqdm(
    range(step_init, step_max), 
    'Training', 
    mininterval=5.0, 
    dynamic_ncols=True)
  for step in tqdm_bar:
    if args.image_conf_aware_training:
      images, image_labels, image_confs = train_iterator.get()
      image_confs = image_confs.to(DEVICE)
    else:
      images, image_labels = train_iterator.get()
    
    # -----------------------------------------------
    # Send to device
    # -----------------------------------------------
    images = images.to(DEVICE)
    image_labels = image_labels.to(DEVICE)

    with torch.autocast(device_type='cuda', enabled=args.mixed_precision):
      # -----------------------------------------------
      # Forward pass
      # -----------------------------------------------
      if args.is_supconclassifier_model:
        image_logits, embeddings = model(images)
      else:
        image_logits = model(images)
      
      # -----------------------------------------------
      # Label smoothing 
      # -----------------------------------------------
      if args.label_smoothing > 0:
        image_labels = smooth_labels(image_labels, args.label_smoothing)

      # -----------------------------------------------
      # Calculate losses
      # -----------------------------------------------
      img_loss = F.binary_cross_entropy_with_logits(
                                image_logits, image_labels,
                                reduction='none') # Per sample, per class loss
      
      if args.image_conf_aware_training:
          img_conformity = 1 - torch.abs(image_labels - image_confs)
          img_w = args.conf_alpha * img_conformity ** args.conf_gamma
          img_loss = img_loss * img_w

      if not len(img_loss.shape) == 0:
          img_loss = img_loss.mean()
      
      # Calculate total BCE loss
      loss = img_loss

      if args.is_supconclassifier_model:
        supcon_loss_value = supcon_loss(embeddings, image_labels)
        loss = loss * args.bce_loss_weight + supcon_loss_value * args.supcon_loss_weight

    # -----------------------------------------------
    # Backward pass and optimization step
    # -----------------------------------------------
    scaler.scale(loss).backward()
    if (step + 1) % args.accumulate_steps == 0:
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad()

      # Step the scheduler
      if not args.poly_lr_decay:
        scheduler.step()

      # Update EMA model
      if args.ema:
        optimizer_global_step = (step + 1) // args.accumulate_steps
        ema_mod.copy(model, ema_model, optimizer_global_step,
                    args.ema, args.ema_decay, args.ema_steps, ema_warmup_steps)
    
    # Detach loss for logging
    loss = loss.detach().cpu().item()

    # -----------------------------------------------
    # Logging
    # -----------------------------------------------
    train_meter.update({'loss': loss})

    epoch = step // step_val
    is_log_step = (step + 1) % step_log == 0
    is_val_step = (step + 1) % step_val == 0

    epoch_loss = train_meter.get()
    learning_rate = float(get_learning_rate_from_optimizer(optimizer))    

    if is_log_step:   
      if args.monitor_memory_usage:
        cpu_mem_used, cpu_mem_free, gpu_mem_used, gpu_mem_reserved, gpu_mem_free = get_memory_usage_GB()
        tqdm_bar.set_postfix({
          "epoch": epoch,
          "loss": f"{epoch_loss:.5f}",
          "lr": f"{learning_rate:.5f}",
          "cpu": f"{cpu_mem_used:.1f}/{cpu_mem_free:.1f}GB",
          "gpu": f"{gpu_mem_used:.1f}/{gpu_mem_reserved:.1f}/{gpu_mem_free:.1f}GB"
        })
      else:
        tqdm_bar.set_postfix({
            "epoch": epoch,
            "loss": f"{epoch_loss:.5f}",
            "lr": f"{learning_rate:.5f}"
        })   

      data = {
        'iteration': step + 1,
        'learning_rate': learning_rate,
        'loss': loss,
        'epoch_loss': epoch_loss,
        'time': train_timer.tok(clear=True),
      }
      wb_logs = {f"train/{k}": v for k, v in data.items()}
      wb_logs["train/epoch"] = epoch
      wandb.log(wb_logs, commit=not (args.validate and is_val_step))

    # -----------------------------------------------
    # Validation
    # -----------------------------------------------
    if args.validate and is_val_step:
      val_loss, report_df = validate_model(
        model, valid_loader, args)
      
      val_data = {
        'iteration': step + 1,
        'val_loss': val_loss,
        'val_classification_report': wandb.Table(dataframe=report_df)
      }

      if args.ema:
        ema_loss, ema_report_df = validate_model(
          ema_mod.inference_model(
            model, ema_model, optimizer_global_step, args.ema, ema_warmup_steps),
          valid_loader, args)
        val_data.update({
          'ema_val_loss': ema_loss,
          'ema_val_classification_report': wandb.Table(dataframe=ema_report_df)
        })

      wb_logs = {f"val/{k}": v for k, v in val_data.items()}
      wb_logs["val/epoch"] = epoch
      wandb.log(wb_logs, commit=True)
      
      print(
        f'step={step + 1} '
        f'val_loss={val_loss:.4f} '
      )
      print(report_df)
      
    # -----------------------------------------------
    # Save model
    # -----------------------------------------------
    if is_val_step:
      if SAVE_EVERY_EPOCH:
        model_path = model_dir + f'model-f{args.val_fold}-e{epoch}.pth'
      if args.ema:
        print(f"[ i ] Saving EMA model to {model_path}")
        save_model(
          ema_mod.inference_model(
            model, ema_model, optimizer_global_step, args.ema, ema_warmup_steps),
          model_path, parallel=GPUS_COUNT > 1)
      else:
        print(f"[ i ] Saving model to {model_path}")
        save_model(model, model_path, parallel=GPUS_COUNT > 1)

      if SAVE_BEST_VAL_LOSS and val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_path = model_dir + f'model-f{args.val_fold}-best.pth'
        if args.ema:
          print(f"[ i ] Saving BEST EMA model to {best_model_path}")
          save_model(
            ema_mod.inference_model(
              model, ema_model, optimizer_global_step, args.ema, ema_warmup_steps),
            best_model_path, parallel=GPUS_COUNT > 1)
        else:
          print(f"[ i ] Saving BEST model to {best_model_path}")
          save_model(model, best_model_path, parallel=GPUS_COUNT > 1)

      torch.save(optimizer.state_dict(), model_dir + 'optimizer.pth')
      if not args.poly_lr_decay:
        torch.save(scheduler.state_dict(), model_dir + 'scheduler.pth')
      torch.save(scaler.state_dict(), model_dir + 'scaler.pth')
      torch.save({'epoch': epoch, 'step': step}, model_dir + 'training_meta.pth')

      train_meter.clear()
  
  # -----------------------------------------------
  # Final save
  # -----------------------------------------------
  model_path = model_dir + f'model-f{args.val_fold}-final.pth'
  if args.ema:
    print(f"[ i ] Saving final EMA model to {model_path}")
    save_model(
      ema_mod.inference_model(
        model, ema_model, optimizer_global_step, args.ema, ema_warmup_steps),
      model_path, parallel=GPUS_COUNT > 1)
  else:
    print(f"[ i ] Saving final model to {model_path}")
    save_model(model, model_path, parallel=GPUS_COUNT > 1)

  print(TAG)
  wb_run.finish()
