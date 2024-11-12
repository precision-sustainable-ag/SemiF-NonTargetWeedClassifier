"""
Task list:
1. label_images - initial manual labelling of images (approx 1k) to train the labeling model
2. predict - auto-labeling data to generate more data for training
3. curate_images - check labeled images (can be auto-labeled by predict.py) and verify the quality
4. train
"""
import logging
import os
import sys

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

sys.path.append("src")


log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_LABEL(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    log.info(f"Starting task {','.join(cfg.tasks)}")
    for tsk in cfg.tasks:
        try:
            task = get_method(f"{tsk}.main")
            task(cfg)
        except Exception as e:
            log.exception("Failed")
            return None


if __name__ == "__main__":
    run_LABEL()