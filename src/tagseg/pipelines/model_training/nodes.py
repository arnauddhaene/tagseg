import logging
from typing import Any, Dict

from aim.pytorch_lightning import AimLogger
import torch
from kedro.config import ConfigLoader
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from tagseg.models.segresnetvae import Net


def load_model(training_params: Dict[str, Any]) -> Dict[str, Any]:

    return Net(dict(
        learning_rate=training_params['learning_rate'],
        weight_decay=training_params['weight_decay'],
        momentum=training_params['momentum']
    ))


def train_model(
    model: nn.Module,
    loader_train: DataLoader,
    loader_val: DataLoader,
) -> Dict[str, nn.Module]:

    params = ConfigLoader("conf/base").get("parameters*", "parameters*/**")

    aim_logger = AimLogger(
        experiment=params["experiment"]["name"],
        train_metric_prefix='train_',
        val_metric_prefix='val_',
    )

    epochs: int = params["training"]["epochs"]

    trainer = pl.Trainer(
        logger=aim_logger,
        default_root_dir="data/07_model_output/",
        accelerator="gpu", devices=1, num_nodes=1,
        auto_select_gpus=True,
        auto_lr_find=True,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        max_epochs=epochs
    )

    trainer.fit(model, loader_train, loader_val)

    return model


def save_model(model: nn.Module, training_params) -> None:
    log = logging.getLogger(__name__)

    model_sd = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    path = training_params['checkpoint']
    torch.save(model_sd, path)

    log.info(f"Trained model saved to {path}")
