import logging
from typing import Any, Dict

import aim
import torch
from kedro.config import ConfigLoader
from torch import nn
from torch.utils.data import DataLoader

from tagseg.models.segmenter import Net
from tagseg.models.trainer import Trainer


def load_model(training_params: Dict[str, Any]) -> Dict[str, Any]:

    return Net(
        load_model=training_params['pretrain_model'],
        learning_rate=training_params['learning_rate'],
        weight_decay=training_params['weight_decay'],
    )


def train_model(
    model: nn.Module,
    loader_train: DataLoader,
    loader_val: DataLoader,
) -> Dict[str, nn.Module]:

    params = ConfigLoader("conf/base").get("parameters*", "parameters*/**")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name: str = params["experiment"]["name"]
    logger = aim.Run(experiment=experiment_name)
    logger["hparams"] = {
        **params,
        "device": device.type,
        "augmentation": str(model.train_aug),
        "model": model._model.__class__.__name__,
    }

    epochs: int = params["training"]["epochs"]

    trainer = Trainer(
        epochs=epochs,
        device=device,
        logger=logger,
        amp=True
    )

    trainer.fit(model, loader_train, loader_val)

    return model._model


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
