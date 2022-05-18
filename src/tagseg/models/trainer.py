from tqdm import tqdm
from typing import Union, Dict

import aim

import torch
from torch import nn
from torch.utils.data import DataLoader

from .segmenter import Net


class Trainer():

    def __init__(
        self,
        epochs: bool,
        logger: aim.Run,
        device: torch.device = 'cpu',
        amp: bool = True,
    ) -> None:
        self.__dict__.update(dict(
            epochs=epochs, logger=logger, amp=amp, device=device
        ))

    @staticmethod
    def tensor_tuple_to(device: torch.device, batch):
        return tuple(map(lambda el: el.double().to(device), batch))

    @staticmethod
    def update_list_dict(
        new: Dict[str, Union[torch.Tensor, int]],
        storage: Dict[str, Union[float, int]]
    ):
        # iterate over all new entries
        for metric, value in new.items():
            # extract value
            if isinstance(value, torch.Tensor):
                value = value.item()
            # add to storage
            if metric in storage.keys():
                storage[metric] += value
            else:
                storage[metric] = value

        return storage

    def training_epoch(self, model, train_dataloader, optimizer):
        model.train()
        torch.set_grad_enabled(True)

        outs = dict()

        pbar = tqdm(
            enumerate(train_dataloader), total=len(train_dataloader), unit="batch", position=0, leave=True)
        pbar.set_description("Rolling stats - ...")

        for batch_idx, batch in pbar:

            batch = self.tensor_tuple_to(self.device, batch)

            with torch.cuda.amp.autocast(enabled=self.amp):
                out = model.training_step(batch, batch_idx)

            # clear gradients
            optimizer.zero_grad()

            # backward
            out['loss'].backward()

            # update parameters
            optimizer.step()

            # update training metrics
            outs = self.update_list_dict(out, outs)
            
            # update progress bar description
            status = f"Rolling stats - mLoss: {outs.get('loss') / outs.get('examples'):.4f}"
            status += f" | mDice(MYO): {outs.get('dice') / outs.get('batches'):.4f}"
            pbar.set_description(status)

        return model.training_epoch_end(outs)

    def validate(self, model, val_dataloader):
        torch.set_grad_enabled(False)
        model.eval()

        val_outs = dict()
        
        pbar = tqdm(
            enumerate(val_dataloader), total=len(val_dataloader), unit="batch", leave=False)

        for val_batch_idx, val_batch in pbar:
            pbar.set_description('Iterating through validation batches')
            
            val_batch = self.tensor_tuple_to(self.device, val_batch)
            val_out = model.validation_step(val_batch, val_batch_idx)
            
            # update validation metrics
            val_outs = self.update_list_dict(val_out, val_outs)

        return model.validation_epoch_end(val_outs)

    def fit(
        self,
        model: Net,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ):
        if self.device.type == 'cuda':
            model._model.to(self.device)
            model._model = nn.DataParallel(model._model)

        optimizer, scheduler = model.configure_optimizers()

        pbar = tqdm(range(self.epochs), unit="epoch", leave=False)
        pbar.set_description(f"Epoch {0:03}")

        for epoch in pbar:

            train_stats = self.training_epoch(model, train_dataloader, optimizer)
            epoch_loss = train_stats['loss']
            epoch_dice = train_stats['dice']
            self.logger.track(epoch_loss, name='loss', context=dict(subset="train"))
            self.logger.track(epoch_dice, name='dice', context=dict(subset="train"))

            val_stats = self.validate(model, val_dataloader)
            val_epoch_loss = val_stats['loss']
            val_epoch_dice = val_stats['dice']
            self.logger.track(val_epoch_loss, name='loss', context=dict(subset="val"))
            self.logger.track(val_epoch_dice, name='dice', context=dict(subset="val"))

            scheduler.step(val_epoch_dice)

            status = f"Epoch {epoch:03} \t Loss {epoch_loss:.4f} \t Dice {epoch_dice:.4f}"
            status += f"\t Val. Loss {val_epoch_loss:.4f} \t Val. Dice {val_epoch_dice:.4f}"

            pbar.set_description(status)
