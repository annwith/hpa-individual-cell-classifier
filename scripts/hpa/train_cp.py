import os
import argparse

import timm
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize

import cv2
cv2.setNumThreads(0)

import datasets
from datasets.hpa import ConfAwareHPADataset

from train_simclr_ws import SupConLoss
from core.networks import *
import core.vision_transformer as vits
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
parser.add_argument('--conf_preds', default=None, type=str)
parser.add_argument('--conf_alpha', default=1.0, type=float)
parser.add_argument('--conf_gamma', default=1.0, type=float)
parser.add_argument('--val_fold', default=0, type=int)
parser.add_argument('--validate_batch_size', default=32, type=int)
parser.add_argument('--cell_count', default=16, type=int)
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
parser.add_argument('--cell_logits_to_image_logits', default=False, type=str2bool)

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
parser.add_argument('--cell_pos_weight', default=1.0, type=float)
parser.add_argument('--cell_loss_weight', default=1.0, type=float)
parser.add_argument('--ema', default=False, type=str2bool)
parser.add_argument('--ema_steps', default=32, type=int)
parser.add_argument('--ema_warmup', default=1, type=int)
parser.add_argument('--ema_decay', default=0.99, type=float)

parser.add_argument('--print_ratio', default=0.1, type=float)
parser.add_argument('--monitor_memory_usage', default=False, type=str2bool)

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
            with torch.amp.autocast(device_type=DEVICE.type):
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

        # --- CÁLCULO DO CONFORMAL SCORE (S) ---
        # S = 1 - probabilidade da classe correta
        # Se for multi-label, pegamos a prob. onde o rótulo é 1
        scores = []
        for p, t in zip(predicted, truth):
            # Para cada amostra, pegamos o score de não-conformidade 
            # referente à classe verdadeira (ground truth)
            # Em cenários multi-class/label, Graham (2024) sugere:
            sample_score = 1.0 - p[t == 1] 
            scores.extend(sample_score.tolist())
        
        scores = np.array(scores)
        n = len(scores)
        
        # --- CÁLCULO DO QUANTIL (q_hat) ---
        # Aplicando a correção para amostras finitas: (n+1)(1-alpha)/n
        alpha = 0.1  # Nível de confiança desejado (exemplo: 0.1 para 90% de confiança)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = np.clip(q_level, 0, 1) # Garantir que está entre 0 e 1
        
        q_hat = np.quantile(scores, q_level, method='higher')

        # --- LOGICA DE CARDINALIDADE |P| (Opcional para log) ---
        # prediction_set = (predicted >= (1 - q_hat))
        # cardinalities = prediction_set.sum(axis=1)
        
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

        return val_loss, report_df, q_hat


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

  conf_aware_training = False

  # Train dataset
  ts = ConfAwareHPADataset(
    df=train_df,
    base_tfms=base_tfms,
    aug_tfms=aug_tfms,
    cell_path=args.data_dir,
    cell_count=args.cell_count,
    cell_size=args.image_size,
    conf_aware=conf_aware_training,
    conf_path=args.conf_preds,
    mode='train'
  )

  # Validation dataset
  vs = None
  if args.validate:
    vs = ConfAwareHPADataset(
      df=valid_df,
      base_tfms=base_tfms,
      aug_tfms=None,
      cell_path=args.data_dir,
      cell_count=args.cell_count,
      cell_size=args.image_size,
      mode='valid'
    )

  return ts, vs


def create_train_valid_dataloaders(args: argparse.Namespace):
  # Sampler and DataLoader
  if args.sampler == 'balanced_cell_count':
    print('[ i ] Using balanced_cell_count sampler')

    num_cells = ts.get_num_cells()
    sampler_threshold = 70 # Change for cfg._.sampler_threshold
    
    print(f'[ i ] Sampler threshold: {sampler_threshold}')
    
    sampler = datasets.BalancedCellCountSampler(
      args, num_cells, batch_size=args.batch_size,
      threshold=sampler_threshold, seed=args.sampler_seed)
    
    train_loader = DataLoader(
      dataset=ts, 
      batch_size=args.batch_size,
      num_workers=args.num_workers,
      collate_fn=datasets.collect_changeable_number_of_cells, 
      sampler=sampler, 
      drop_last=True, 
      pin_memory=False)
  else:
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
  # ViT-based model
  if 'vit' in args.architecture:
    backbone = vits.vit_small(
      img_size=[224], 
      patch_size=16, 
      in_chans=4, 
      num_classes=19,
      drop_path_rate=0.1
    )

    if args.backbone_weights == 'imagenet':
      timm_model = timm.create_model(
        "vit_small_patch16_224", pretrained=True, in_chans=3, num_classes=19)

      # Get the original conv weights
      proj = timm_model.patch_embed.proj
      w = proj.weight.data  # shape [embed_dim, 3, 16, 16]

      # Make new conv layer with 4 input channels
      new_proj = nn.Conv2d(
          in_channels=4,
          out_channels=proj.out_channels,
          kernel_size=proj.kernel_size,
          stride=proj.stride,
          padding=proj.padding,
          bias=(proj.bias is not None)
      )

      # Copy pretrained weights
      with torch.no_grad():
          new_proj.weight[:, :3] = w 
          new_proj.weight[:, 3] = w[:, 0] 
          if proj.bias is not None:
              new_proj.bias.copy_(proj.bias)

      # Replace layer in backbone
      timm_model.patch_embed.proj = new_proj

      state_dict = timm_model.state_dict()
      msg = backbone.load_state_dict(state_dict, strict=False)
      print('[ i ] ImageNet pretrained weights loaded with msg: {}'.format(msg))

    elif os.path.isfile(args.backbone_weights):
      state_dict = torch.load(args.backbone_weights, map_location="cpu", weights_only=False)
      
      if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
        print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[args.checkpoint_key]

      state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
      state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
      
      msg = backbone.load_state_dict(state_dict, strict=False)
      print('[ i ] Pretrained weights found at {} and loaded with msg: {}'.format(args.backbone_weights, msg))
    
    model = vits.ViT_MIL_Classifier(
      backbone=backbone,
      num_classes=19
    )

  # CNN-based model
  else:
    # SimCLR pretrained model or DINO pretrained model
    if args.is_simclr_model or args.is_dino_model:
      print("Loading backbone from SimCLR pretrained model.")
      model = HPAClassifier(
        args.architecture,
        num_classes=19,
        channels=4,
        backbone_weights=None,
        mode=args.mode,
        dilated=args.dilated,
        trainable_stem=args.trainable_stem,
        trainable_backbone=args.trainable_backbone,
      )

      # Load SimCLR weights
      if args.is_simclr_model:
        state_dict = torch.load(args.backbone_weights, map_location="cpu")
        state_dict = {
            k.replace("backbone.", ""): v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
        print(f"Backbone state dict keys: {list(state_dict.keys())} ...")
        model.backbone.load_state_dict(state_dict, strict=True)

      # Load DINO weights
      if args.is_dino_model:
        state_dict = torch.load(args.backbone_weights, map_location="cpu", weights_only=False)
    
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
          print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
          state_dict = state_dict[args.checkpoint_key]

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        
        # Remove head weights
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}

        print(f"Backbone state dict keys: {list(state_dict.keys())} ...")
        model.backbone.load_state_dict(state_dict, strict=True)
    
    # Regular model
    else:
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


def reannotate_negative_labels(
    args: argparse.Namespace,
    cell_labels: torch.Tensor,
    image_labels: torch.Tensor,
    cell_conf: torch.Tensor,
    n_cells: list,
):
    # Formato das labels: (n_cells_total, 19) e (batch_size, 19)

    # --- Step 1: Cell level ---
    class_18_idx = 18
    high_conf_mask = cell_conf[:, class_18_idx] > args.reannotate_threshold
    cell_labels[high_conf_mask, class_18_idx] = 1
    cell_labels[high_conf_mask, :class_18_idx] = 0

    # Correção: Garantir que n_cells seja um tensor no dispositivo correto
    device = image_labels.device
    n_cells_tensor = n_cells.to(device)

    # --- Step 2: Image level ---
    # Create an index vector that maps every cell to its image ID
    # e.g., if n_cells = [2, 1], batch_indices = [0, 0, 1]
    batch_indices = torch.repeat_interleave(
        torch.arange(len(n_cells_tensor), device=n_cells_tensor.device), 
        n_cells_tensor
    )

    # Extract the values (0 or 1)
    values = cell_labels[:, class_18_idx]

    # Use scatter_reduce_ to find the max value per image index
    # "amax" ensures that if any cell is 1, the image becomes 1
    image_labels[:, class_18_idx].scatter_reduce_(
        0, 
        batch_indices, 
        values, 
        reduce="amax", 
        include_self=False
    )

    return cell_labels, image_labels


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
  SAVE_EVERY_EPOCH = True
  q_hat = 1
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
  pos_weight = torch.tensor([args.cell_pos_weight] * 19).to(DEVICE)
  print(f"[ i ] Cell positive weight: {pos_weight}")

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
  model_path = model_dir + f'model.pth'

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

  train_meter = MetricsContainer(['loss'])
  train_timer = Timer()

  tqdm_bar = tqdm(range(step_init, step_max), 'Training', mininterval=2.0)
  for step in tqdm_bar:
    images, cell_labels, image_labels, cell_conf, image_conf, n_cells = train_iterator.get()
    
    # -----------------------------------------------
    # Reshape if needed
    # -----------------------------------------------
    if args.cell_count > 0:
      images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
      cell_labels = cell_labels.view(-1, cell_labels.shape[-1])
      cell_conf = cell_conf.view(-1, cell_conf.shape[-1])

    # -----------------------------------------------
    # Send to device
    # -----------------------------------------------
    images = images.to(DEVICE)
    cell_labels = cell_labels.to(DEVICE)
    image_labels = image_labels.to(DEVICE)
    cell_conf = cell_conf.to(DEVICE)
    image_conf = image_conf.to(DEVICE)

    with torch.autocast(device_type='cuda', enabled=args.mixed_precision):
      # -----------------------------------------------
      # Forward pass
      # -----------------------------------------------
      cell_logits, image_logits = model(
        images, 
        n_cells, 
        cell_logits_to_image_logits=args.cell_logits_to_image_logits)
        
      cell_probs = torch.sigmoid(cell_logits) # Probs
      image_probs = torch.sigmoid(image_logits) # Probs

      # Regra da Predição Conformal:
      # A classe 'j' pertence ao conjunto P se: S_j <= q_hat
      # S_j é (1 - prob_j). Logo: (1 - prob_j) <= q_hat  =>  prob_j >= (1 - q_hat)
      threshold = 1 - q_hat
      cell_prediction_sets = (cell_probs >= threshold).float()
      image_prediction_sets = (image_probs >= threshold).float()

      # Cardinalidade |P| para cada amostra do batch
      # (Soma de quantas classes superaram o threshold)
      cell_cardinality = cell_prediction_sets.sum(dim=1)
      image_cardinality = image_prediction_sets.sum(dim=1)

      # -----------------------------------------------
      # Calculate losses
      # -----------------------------------------------
      cell_loss = F.binary_cross_entropy_with_logits(
                                  cell_logits, cell_labels,
                                  pos_weight=pos_weight,
                                  reduction='none') # Per sample, per class loss
        
      img_loss = F.binary_cross_entropy_with_logits(
                                image_logits, image_labels,
                                reduction='none') # Per sample, per class loss

      # Se for multi-label, a loss_base costuma ter shape [batch, num_classes]
      # Tiramos a média por classe primeiro para ter uma loss por amostra [batch]
      cell_loss_per_sample = cell_loss.mean(dim=1)
      img_loss_per_sample = img_loss.mean(dim=1)

      # A MÁGICA DO CitL:
      # Multiplica a loss da amostra pela sua incerteza (|P|)
      cell_loss_reweighted = cell_loss_per_sample * cell_cardinality
      img_loss_reweighted = img_loss_per_sample * image_cardinality

      # Média final para o backward
      cell_loss_final = cell_loss_reweighted.mean()
      img_loss_final = img_loss_reweighted.mean()
      
      # Calculate total BCE loss
      loss = cell_loss_final * args.cell_loss_weight + img_loss_final

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
  
    if args.monitor_memory_usage:
      cpu_mem_used, cpu_mem_free, gpu_mem_used, gpu_mem_reserved, gpu_mem_free = get_memory_usage_GB()
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

    # -----------------------------------------------
    # Validation
    # -----------------------------------------------
    if args.validate and is_val_step:
      val_loss, report_df, q_hat = validate_model(
        model, valid_loader, args)
      
      val_data = {
        'iteration': step + 1,
        'val_loss': val_loss,
        'val_classification_report': wandb.Table(dataframe=report_df),
        'q_hat': q_hat,
      }

      if args.ema:
        ema_loss, ema_report_df, ema_q_hat = validate_model(
          ema_mod.inference_model(
            model, ema_model, optimizer_global_step, args.ema, ema_warmup_steps),
          valid_loader, args)
        val_data.update({
          'ema_val_loss': ema_loss,
          'ema_val_classification_report': wandb.Table(dataframe=ema_report_df),
          'ema_q_hat': ema_q_hat,
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

      torch.save(optimizer.state_dict(), model_dir + 'optimizer.pth')
      if not args.poly_lr_decay:
        torch.save(scheduler.state_dict(), model_dir + 'scheduler.pth')
      torch.save(scaler.state_dict(), model_dir + 'scaler.pth')
      torch.save({'epoch': epoch, 'step': step}, model_dir + 'training_meta.pth')

      train_meter.clear()
  
  # -----------------------------------------------
  # Final save
  # -----------------------------------------------
  if args.ema:
    print(f"[ i ] Saving EMA model to {model_path}")
    save_model(
      ema_mod.inference_model(
        model, ema_model, optimizer_global_step, args.ema, ema_warmup_steps),
      model_path, parallel=GPUS_COUNT > 1)
  else:
    print(f"[ i ] Saving model to {model_path}")
    save_model(model, model_path, parallel=GPUS_COUNT > 1)

  print(TAG)
  wb_run.finish()
