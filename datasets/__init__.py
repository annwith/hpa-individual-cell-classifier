from typing import Tuple
import random

import numpy as np
from torch.utils.data import WeightedRandomSampler, Sampler
from sklearn.utils import compute_sample_weight
import albumentations as A

from tools.ai.augment_utils import *


SAMPLERS = ("default", "balanced-sample", "balanced-class")


def get_train_sampler_and_shuffler(
    sampler: str,
    source = None,
    seed: Optional[int] = None,
    clip_value: int = 10,
) -> Tuple[WeightedRandomSampler, bool]:
  if sampler not in SAMPLERS:
    raise ValueError(f"Unknown sampler '{sampler}'. Known samplers are: {SAMPLERS}.")

  if sampler == "default":
    return None, True

  if sampler.startswith("balanced"):
    labels = source.get_labels() # NumPy array of shape (num_samples, num_classes)

    if sampler == "balanced-sample":
      weights = compute_sample_weight(class_weight="balanced", y=labels)
      n_samples = labels.shape[0]  # Total de imagens
      n_classes = labels.shape[1]  # Total de classes
      class_counts = labels.sum(axis=0)
      # n_samples / (n_classes * freq)
      class_weights = n_samples / (n_classes * class_counts)
      print("[ i ] Weights shape:", weights.shape)
      print("[ i ] Unique weights:", np.unique(weights))
      print("[ i ] Computed class weights (balanced-sample):")
      [print(f"Class {i} weight: {w:.3f}") for i, w in enumerate(class_weights)]

    if sampler == "balanced-class":
      freq = labels.sum(axis=0, keepdims=True)
      weights = (labels * (freq.max()/freq)).max(axis=1).clip(max=clip_value)
      print("[ i ] Weights shape:", weights.shape)
      print("[ i ] Unique weights:", np.unique(weights))
      print("[ i ] Computed class weights (balanced-class):")
      [print(f"Class {i} weight: {freq.max()/f:.3f}") for i, f in enumerate(freq[0])]

    generator = torch.Generator()
    if seed is not None: generator.manual_seed(seed)

    return (
      WeightedRandomSampler(
        weights, 
        len(source), 
        replacement=True, 
        generator=generator),
      None)


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

    if self.args.debug:
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
    if self.args.debug and len(special_batches) > 0:
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
    if self.args.debug:
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


class Iterator:

  def __init__(self, loader):
    self.loader = loader
    self.init()

  def init(self):
    self.iterator = iter(self.loader)

  def get(self):
    try:
      data = next(self.iterator)
    except StopIteration:
      self.init()
      data = next(self.iterator)

    return data