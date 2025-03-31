import comet_ml
import torch
import os

import torch.autograd.gradcheck
from SPEN.cfg import SPEEDConfig, SPARKConfig
from SPEN.data import get_speed_dataloader, get_spark_dataloader
from SPEN.module import ImageModule
from SPEN.utils import parse2config

from SPEN.TorchModel import Trainer
from SPEN.TorchModel.logger import CometLogger
from SPEN.TorchModel.callbacks import Checkpoint, LRMonitor, ModelSummary

from lightning.pytorch import seed_everything
from pathlib import Path


if __name__ == "__main__":
    config = SPEEDConfig()
    config = parse2config(config)
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

    torch.set_float32_matmul_precision("high")
    
    dirpath = Path(f"./result/{config.name}")
    if not dirpath.exists():
        dirpath.mkdir(parents=True)
    
    seed_everything(config.seed)

    # ==========Trainer==========
    # ----------debug----------
    if config.debug:
        config.num_workers = 6
        config.cache = False
        config.batch_size = 10
        config.epochs = 10
        config.offline = True
    # ----------Callbacks----------
    checkpoint = Checkpoint(
        dirpath=str(dirpath),
        filename="best",
        monitor="val/score",
        monitor_mode="min"
    )
    lr_monitor = LRMonitor()
    model_summary = ModelSummary(input_size=(1, 1, *config.image_size), depth=4)
    callbacks = [checkpoint, lr_monitor, model_summary]
    # ----------Logger----------
    comet_logger = CometLogger(
        api_key=config.comet_api,
        project_name="SPEEDBest",
        experiment_name=config.name,
        online=not config.offline,
    )
    
    # ==========Model==========
    model = ImageModule(config=config)

    train_dataloader, val_dataloader = get_speed_dataloader(config)
    # train_dataloader, val_dataloader = get_spark_dataloader(config)

    trainer = Trainer(
        model=model,
        config=config,
        callbacks=callbacks,
        device=config.device,
        logger=comet_logger,
        gradient_clip_val=config.gradient_clip_val,
    )

    trainer.fit(train_dataloader, val_dataloader)