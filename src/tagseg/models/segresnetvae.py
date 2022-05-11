from typing import Any, Dict

import pytorch_lightning as pl
import torch
from torch import nn

from monai.transforms import AsDiscrete, EnsureType, Compose
from monai.networks import nets
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

from ..metrics.shape import ShapeDistLoss


class Net(pl.LightningModule):

    def __init__(self, hparams: Dict[str, Any]) -> None:
        super(Net, self).__init__()
        self.save_hyperparameters()

        # Add hparams as attributes
        self.__dict__.update(hparams)

        self._model = nets.SegResNetVAE(
            in_channels=1,
            out_channels=2,
            input_image_size=(128, 128),
            spatial_dims=2,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.dice_criterion = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
        self.shape_criterion = ShapeDistLoss(include_background=False, to_onehot_y=True)

        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.hd_metric = HausdorffDistanceMetric(include_background=False)

        self.best_val_dice = 0
        self.best_val_epoch = 0

        self.post_pred = Compose([
            EnsureType("tensor", device="cpu"),
            AsDiscrete(argmax=True, to_onehot=2)
        ])
        self.post_label = Compose([
            EnsureType("tensor", device="cpu"),
            AsDiscrete(to_onehot=2)
        ])

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        return (
            self.criterion(outputs, targets[:, 0])
            + self.dice_criterion(outputs, targets)
            + 1e-3 * self.shape_criterion(outputs, targets)
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        return self._model(x)[0]

    def training_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label'].long()
        loss = self.loss_fn(self.forward(images), labels)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['image'], batch['label'].long()

        roi_size = (64, 64)
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward)
        loss = self.loss_fn(outputs, labels)
        
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        self.dice_metric(y_pred=outputs, y=labels)
        self.hd_metric(y_pred=outputs, y=labels)

        return {"val_loss": loss, "val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]

        mean_val_loss = torch.tensor(val_loss / num_items)

        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_hd = self.hd_metric.aggregate().item()
        self.hd_metric.reset()
         
        self.log('val_dice', mean_val_dice)
        self.log('val_hd', mean_hd)
        self.log('val_loss', mean_val_loss)

        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        print(
            f"ep {self.current_epoch:03d} | "
            f"loss {mean_val_loss:.4f} | "
            f"dice {mean_val_dice:.4f} | "
            f"hd {mean_hd:.4f}"
        )

    
