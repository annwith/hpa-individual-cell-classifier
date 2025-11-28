from typing import Tuple

import numpy as np
from torch.utils.data import WeightedRandomSampler
from sklearn.utils import compute_sample_weight

from tools.ai.augment_utils import *


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
