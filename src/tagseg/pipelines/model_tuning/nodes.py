import logging

from aim.pytorch_lightning import AimLogger

from torch import nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl


def find_best_lr(
    model: nn.Module,
    loader_train: DataLoader,
    loader_val: DataLoader,
) -> None:
    log = logging.getLogger(__name__)

    aim_logger = AimLogger(
        experiment='Tuning: finding best LR',
        train_metric_prefix='train_',
        val_metric_prefix='val_',
    )

    trainer = pl.Trainer(
        logger=aim_logger,
        accelerator="gpu", devices=1,
        auto_select_gpus=True,
        enable_progress_bar=True,
        amp_backend='native',
        precision=32
    )

    lr_finder = trainer.tuner.lr_find(
        model, train_dataloaders=loader_train, val_dataloaders=loader_val)

    fig = lr_finder.plot(suggest=True)
    fig.savefig('data/08_reporting/best_lr.png')

    log.info(lr_finder.results)
    suggestion = lr_finder.suggestion()
    log.info(f'LR suggestion: {suggestion}')

    return suggestion
