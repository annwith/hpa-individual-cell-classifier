import argparse
import os

import psutil
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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
from tools.general.io_utils import *
from tools.general import wandb_utils
from tools.ai.optim_utils import *
from tools.ai.log_utils import *
from tools.general.time_utils import *

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
parser.add_argument('--cell_conf_aware_training', default=False, type=str2bool)
parser.add_argument('--image_conf_aware_training', default=False, type=str2bool)
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

parser.add_argument('--reannotate_neg_labels', default=False, type=str2bool)
parser.add_argument('--reannotate_threshold', default=0.5, type=float)

parser.add_argument('--cell_conf_as_cell_labels', default=False, type=str2bool)

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


class BalancedCellCountSampler(Sampler):
  def __init__(self, args, num_cells, batch_size=3, threshold=70, seed=42):
    self.args = args
    self.num_cells = num_cells
    self.batch_size = batch_size
    self.threshold = threshold
    self.seed = seed
    self.indices = list(range(len(num_cells)))
    random.seed(seed)

  def __iter__(self):
    # Separate indices
    large_idxs = [i for i in self.indices if self.num_cells[i] >= self.threshold]
    large_idxs.sort(key=lambda i: self.num_cells[i], reverse=True) # decreasing cell count
    small_idxs = [i for i in self.indices if self.num_cells[i] < self.threshold]
    small_idxs.sort(key=lambda i: self.num_cells[i])  # increasing cell count

    if args.debug:
      print(f'[ i ] Number of large indices: {len(large_idxs)}')
      print(f'[ i ] Large indices: {large_idxs[:10]}')
      print(f'[ i ] Large values: {[self.num_cells[i] for i in large_idxs[:10]]}')
      print(f'[ i ] Number of small indices: {len(small_idxs)}')
      print(f'[ i ] Small indices: {small_idxs[:10]}')
      print(f'[ i ] Small values: {[self.num_cells[i] for i in small_idxs[:10]]}')

    special_batches = []
    used_small = set()
    used_large = set()

    # Form special batches
    small_pointer = 0
    for large_idx in large_idxs:
      if small_pointer + 1 >= len(small_idxs):
        break  # not enough smalls left
      batch = [large_idx, small_idxs[small_pointer], small_idxs[small_pointer + 1]]
      special_batches.append(batch)
      used_large.add(large_idx)
      used_small.update([small_idxs[small_pointer], small_idxs[small_pointer + 1]])
      small_pointer += 2

    # One batch print
    if args.debug and len(special_batches) > 0:
      print(f'[ i ] Special batch: {special_batches[-1]}')
      print(f'[ i ] Special batch values: {[self.num_cells[i] for i in special_batches[-1]]}')

    # Remaining indices (not already used)
    remaining = list(set(self.indices) - used_large - used_small)
    random.shuffle(remaining)

    # Group remaining into batches
    random_batches = []
    for i in range(0, len(remaining), self.batch_size):
      batch = remaining[i:i + self.batch_size]
      if len(batch) == self.batch_size:
        random_batches.append(batch)

    # Print random batch
    if args.debug:
      print(f'[ i ] Number of random batches: {len(random_batches)}')
      if len(random_batches) > 0:
        print(f'[ i ] Random batch: {random_batches[-1]}')
        print(f'[ i ] Random batch values: {[self.num_cells[i] for i in random_batches[-1]]}')

    # Combine special and random batches
    final_batches = special_batches + random_batches
    random.shuffle(final_batches)

    # Flatten to a list of indices
    final_indices = [idx for batch in final_batches for idx in batch]

    return iter(final_indices)

  def __len__(self):
    return len(self.indices)


def collect_changeable_number_of_cells(batch):
    # Desempacota o batch
    ipts, lbls, img_lbls, conf_lbls, conf_img_lbls, cnts = zip(*batch)

    # Concatena células (ex: ipt = [tensor(C_i) for i in batch] -> tensor(C_total, ...))
    ipts = torch.cat(ipts, dim=0)
    lbls = torch.cat(lbls, dim=0)
    conf_lbls = torch.cat(conf_lbls, dim=0)

    # lbls geralmente são rótulos da imagem inteira (1 por imagem), então pode ser empilhado
    img_lbls = torch.stack(img_lbls, dim=0)
    conf_img_lbls = torch.stack(conf_img_lbls, dim=0)

    # cnts indica quantas células por imagem — ex: [12, 8, 10] — mantido como tensor
    cnts = torch.tensor(cnts)

    return ipts, lbls, img_lbls, conf_lbls, conf_img_lbls, cnts


def get_transformations(aug_yaml):
    return A.load(aug_yaml, data_format='yaml')


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

            # DEBUG: Print each value and its shape and type
            # if args.debug:
            #     print(f"ipt: {ipt.shape}, {ipt.dtype}")
            #     print(f"img_lbl: {img_lbl.shape}, {img_lbl.dtype}")

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

            # DEBUG: Print each value and its shape and type
            # if args.debug:
            #     print(f"predicted: {pred.shape}, {pred.dtype}")
            #     print(f"truth: {img_lbl.shape}, {img_lbl.dtype}")

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

        # DEBUG: Print each value and its shape and type
        # if args.debug:
        #     print(f"val_loss: {val_loss}")
        #     print(f"predicted: {predicted.shape}, {predicted.dtype}")
        #     print(f"truth: {truth.shape}, {truth.dtype}")
        
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

  if args.class_weight and args.class_weight != "none":
    CLASS_WEIGHT = torch.Tensor(list(map(float, args.class_weight.split(",")))).to(DEVICE)
  else:
    CLASS_WEIGHT = None

  SAVE_EVERY_EPOCH = True

  # Positive weight for cell classification
  pos_weight = torch.ones(19) / args.cell_pos_weight
  pos_weight = pos_weight.to(DEVICE)
  print(f"[ i ] Cell positive weight: {pos_weight}")

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

  train_df, valid_df = (df[df.fold != args.val_fold],
                        df[df.fold == args.val_fold])

  # Debugging
  if int(args.debug):
    train_df = train_df.sample(n=400, random_state=SEED)
    if args.validate:
      valid_df = valid_df.sample(n=100, random_state=SEED)

  # Data transformations
  if args.aug_yaml:
    print(f"Using augmentations from {args.aug_yaml}")
    train_tfms = get_transformations(args.aug_yaml)

  # Check if training is confidence-aware
  if args.cell_conf_aware_training or args.image_conf_aware_training or args.cell_conf_as_cell_labels:
    conf_aware_training = True
  else:
    conf_aware_training = False

  # Train dataset
  ts = ConfAwareHPADataset(
    df=train_df,
    tfms=train_tfms,
    cell_path=args.data_dir,
    cell_count=args.cell_count,
    cell_size=args.image_size,
    conf_aware=conf_aware_training,
    conf_path=args.conf_preds,
    label_smoothing=args.label_smoothing,
    mode='train'
  )

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
  
  # Sampler and DataLoader
  if args.sampler == 'balanced_cell_count':
    print('[ i ] Using balanced_cell_count sampler')

    num_cells = ts.get_num_cells()
    sampler_threshold = 70 # Change for cfg._.sampler_threshold
    
    print(f'[ i ] Sampler threshold: {sampler_threshold}')
    
    sampler = BalancedCellCountSampler(
      args, num_cells, batch_size=args.batch_size,
      threshold=sampler_threshold, seed=args.sampler_seed)
    
    train_loader = DataLoader(
      dataset=ts, 
      batch_size=args.batch_size,
      num_workers=args.num_workers,
      collate_fn=collect_changeable_number_of_cells, 
      sampler=sampler, 
      drop_last=True, 
      pin_memory=False)
  else:
    sampler, shuffle = datasets.get_train_sampler_and_shuffler(
      args.sampler, # Apenas o default funciona (None, True)
      seed=args.sampler_seed)

    train_loader = DataLoader(
      dataset=ts, 
      batch_size=args.batch_size,
      shuffle=shuffle,
      sampler=sampler,
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
  ema_warmup_steps = args.ema_warmup * int(step_val // args.accumulate_steps)
  print(f"[ i ] Iterations: first={step_init} logging={step_log} validation={step_val} max={step_max}")

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
  log_model("Vanilla", model, args)

  param_groups, param_names = model.get_parameter_groups(with_names=True)
  model.train()

  if args.ema:
    ema_model = ema_mod.init(model, DEVICE, args.ema)
  
  model = model.to(DEVICE)

  if GPUS_COUNT > 1:
    print(f"GPUs={GPUS_COUNT}")
    model = torch.nn.DataParallel(model)

    if args.ema:
      ema_model = torch.nn.DataParallel(ema_model)

  # Optimizer
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
    images, cell_labels, image_labels, cell_conf, image_conf, n_cells = train_iterator.get()
    
    if args.cell_count > 0:
      images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
      cell_labels = cell_labels.view(-1, cell_labels.shape[-1])
      cell_conf = cell_conf.view(-1, cell_conf.shape[-1])

    # Send to device
    images = images.to(DEVICE)
    cell_labels = cell_labels.to(DEVICE)
    image_labels = image_labels.to(DEVICE)
    cell_conf = cell_conf.to(DEVICE)
    image_conf = image_conf.to(DEVICE)

    # DEBUG: print values and shapes
    # print(f"images shape: {images.shape}, dtype: {images.dtype}")
    # print(f"cell_labels shape: {cell_labels.shape}, dtype: {cell_labels.dtype}")
    # print(f"image_labels shape: {image_labels.shape}, dtype: {image_labels.dtype}")
    # print(f"cell_conf shape: {cell_conf.shape}, dtype: {cell_conf.dtype}")
    # print(f"image_conf shape: {image_conf.shape}, dtype: {image_conf.dtype}")

    # If a cell conf[18] > 0.5, change the cell label to positive
    # If so, change the image label [18] to positive on that image

    if args.reannotate_neg_labels:
      # Step 1: Override cell labels for class 18 based on confidence
      class_18_idx = 18
      high_conf_mask = cell_conf[:, class_18_idx] > args.reannotate_threshold
      cell_labels[high_conf_mask, class_18_idx] = 1

      # Step 2: Propagate to image-level label
      # We need to know which cells belong to which image.
      # If not available, use `n_cells` (number of cells per image in the batch)
      if args.cell_count > 0:
          # We assume cells are ordered per image in the batch.
          # Reconstruct batch size
          batch_size = len(n_cells)
          start = 0
          for i in range(batch_size):
              num = n_cells[i]
              # Get all cells for this image
              cell_slice = cell_labels[start:start+num]
              # If any cell has class 18 == 1, set image label class 18 = 1
              if cell_slice[:, class_18_idx].any():
                  image_labels[i, class_18_idx] = 1
              start += num

    with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
      cell_logits, image_logits = model(
        images, 
        cnt=n_cells, 
        cell_logits_to_image_logits=args.cell_logits_to_image_logits)

      if args.cell_conf_as_cell_labels:
        # print(f"[ i ] Using cell confidence as cell labels.")
        neg_idx = 18
        masked_conf = torch.where(cell_labels.bool(), cell_conf, torch.zeros_like(cell_conf))

        # Always keep conf for the negative class
        masked_conf[:, neg_idx] = cell_conf[:, neg_idx]
        
        cell_loss = F.binary_cross_entropy_with_logits(
                                  cell_logits, masked_conf,
                                  pos_weight=pos_weight,
                                  reduction='none') # Per sample, per class loss
      else:
        cell_loss = F.binary_cross_entropy_with_logits(
                                  cell_logits, cell_labels,
                                  pos_weight=pos_weight,
                                  reduction='none') # Per sample, per class loss
      
      img_loss = F.binary_cross_entropy_with_logits(
                                image_logits, image_labels,
                                reduction='none') # Per sample, per class loss
      
      weighted_cell_loss = cell_loss
      weighted_img_loss = img_loss
      
      if args.cell_conf_aware_training:
          conformity = 1 - torch.abs(cell_labels - cell_conf)
          w = args.conf_alpha * conformity ** args.conf_gamma
          weighted_cell_loss = cell_loss * w

      if args.image_conf_aware_training:
          img_conformity = 1 - torch.abs(image_labels - image_conf)
          img_w = args.conf_alpha * img_conformity ** args.conf_gamma
          weighted_img_loss = img_loss * img_w

      # Calculate mean if needed
      if not len(weighted_cell_loss.shape) == 0:
          weighted_cell_loss = weighted_cell_loss.mean()
      if not len(weighted_img_loss.shape) == 0:
          weighted_img_loss = weighted_img_loss.mean()
      
      # Calculate total loss
      # if args.supervised_negative_training:
      #   loss = weighted_cell_loss + weighted_img_loss
      # else:
      loss = weighted_cell_loss * args.cell_loss_weight + weighted_img_loss

    scaler.scale(loss).backward()

    if (step + 1) % args.accumulate_steps == 0:
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad()

      if not args.poly_lr_decay:
        scheduler.step()

      if args.ema:
        ema_mod.copy(model, ema_model, optimizer.global_step,
                    args.ema, args.ema_decay, args.ema_steps, ema_warmup_steps)

    loss = loss.detach().cpu().item()

    train_meter.update({'loss': loss})

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
        'loss': loss,
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
        'step={iteration:,} '
        'val_loss={val_loss:.4f} '
      )
      print(report_df)
      
    if is_val_step:
      if SAVE_EVERY_EPOCH:
        model_path = model_dir + f'model-f{args.val_fold}-e{epoch}.pth'

      if args.ema:
        print(f"[ i ] Saving EMA model to {model_path}")
        save_model(
          ema_mod.inference_model(model, ema_model, optimizer.global_step, args.ema, ema_warmup_steps),
          model_path, parallel=GPUS_COUNT > 1)
      else:
        print(f"[ i ] Saving model to {model_path}")
        save_model(model, model_path, parallel=GPUS_COUNT > 1)

      torch.save(optimizer.state_dict(), model_dir + 'optimizer.pth')

      if not args.poly_lr_decay:
        torch.save(scheduler.state_dict(), model_dir + 'scheduler.pth')
      torch.save(scaler.state_dict(), model_dir + 'scaler.pth')
      torch.save({'epoch': epoch, 'step': step}, model_dir + 'training_meta.pth')

      train_meter.clear()
  
  if args.ema:
    print(f"[ i ] Saving EMA model to {model_path}")
    save_model(
      ema_mod.inference_model(model, ema_model, optimizer.global_step, args.ema, ema_warmup_steps),
      model_path, parallel=GPUS_COUNT > 1)
  else:
    print(f"[ i ] Saving model to {model_path}")
    save_model(model, model_path, parallel=GPUS_COUNT > 1)

  print(TAG)
  wb_run.finish()
