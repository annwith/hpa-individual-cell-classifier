import wandb
import os

os.environ["WANDB_MODE"] = "offline"


def setup(name, config, job_type="train", tags=None):
  wb_run = wandb.init(
    name=name,
    job_type=job_type,
    config=config,
    tags=tags,
  )

  return wb_run

