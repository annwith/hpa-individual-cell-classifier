from typing import Tuple

import numpy as np

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


def imagenet_stats():
  return (
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
  )


SAMPLERS = ("default", "balanced-sample", "balanced-class")

def get_train_sampler_and_shuffler(
    sampler: str,
    source = None,
    seed: Optional[int] = None,
    clip_value: int = 10,
) -> Tuple["Sampler", bool]:
  if sampler not in SAMPLERS:
    raise ValueError(f"Unknown sampler '{sampler}'. Known samplers are: {SAMPLERS}.")

  if sampler == "default":
    return None, True

  if sampler.startswith("balanced"):
    from torch.utils.data import WeightedRandomSampler
    labels = np.asarray([source.get_label(_id) for _id in source.sample_ids])

    if sampler == "balanced-sample":
      from sklearn.utils import compute_sample_weight
      weights = compute_sample_weight("balanced", labels)

    if sampler == "balanced-class":
      freq = labels.sum(0, keepdims=True)
      weights = (labels * (freq.max()/freq)).max(1).clip(max=clip_value)

    generator = torch.Generator()
    if seed is not None: generator.manual_seed(seed)

    return (
      WeightedRandomSampler(weights, len(source), replacement=True, generator=generator),
      None)
