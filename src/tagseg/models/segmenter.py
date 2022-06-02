import logging

import torch
from torch import nn

import numpy as np

from skimage import morphology, measure
from medpy.metric.binary import dc

from monai.networks import nets, one_hot
from monai.networks.layers import Norm
from monai.losses import DiceCELoss
from monai.metrics import compute_hausdorff_distance

import kornia.augmentation as K

from ..metrics import DiceMetric, ShapeDistLoss


class Net():

    def __init__(
        self,
        load_model: str = None,
        model_type: str = 'SegResNetVAE',
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-3,
        gamma: float = 1e-2
    ) -> None:
        super(Net, self).__init__()
        # Add hparams as attribute
        self.__dict__.update(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gamma=gamma,
            model_type=model_type
        )

        self.num_classes = 2
        self.in_channels = 1
        self.spatial_dims = 2

        if self.model_type == 'UNet':
            self._model = nets.UNet(
                in_channels=self.in_channels,
                out_channels=self.num_classes,
                spatial_dims=self.spatial_dims,
                channels=(16, 32, 64, 128),
                strides=(2, 2, 2),
                num_res_units=1,
                dropout=0.1,
                norm=Norm.BATCH,
            )

            # Weight initialization
            self._model.apply(self.weights_init)

        elif self.model_type == 'SegResNetVAE':
            self._model = nets.SegResNetVAE(
                in_channels=self.in_channels,
                out_channels=self.num_classes,
                input_image_size=(256, 256),
                spatial_dims=self.spatial_dims
            )

        else:
            raise ValueError(f'Expected UNet or SegResNetVAE model. Got {self.model_type} instead.')

        # Start with double floats
        # We're using AMP when training on GPUs
        self._model = self._model.double()

        # Load checkpointed version of the model
        if load_model is not None:
            self._model.load_state_dict(torch.load(load_model))

        self.criterion = DiceCELoss(include_background=True, to_onehot_y=True, sigmoid=True)
        self.shape_criterion = ShapeDistLoss(include_background=False, to_onehot_y=True, sigmoid=True)
        self.dice_metric = DiceMetric(include_background=False)

        proba: float = 0.2

        self.train_aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=proba),
            K.RandomVerticalFlip(p=proba),
            K.RandomElasticTransform(p=proba),
            K.RandomGaussianNoise(p=proba),
            K.RandomSharpness(p=proba),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.1), p=proba),
            data_keys=["input", "mask"],
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=10)
        grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        return optimizer, scheduler, grad_scaler

    def loss_fn(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(outputs, targets) + self.gamma * self.shape_criterion(outputs, targets)

    def forward(self, x) -> torch.Tensor:
        if self.model_type == 'UNet':
            return self._model(x)

        elif self.model_type == 'SegResNetVAE':
            return self._model(x)[0]

        else:
            raise ValueError(f'Expected UNet or SegResNetVAE model. Got {self.model_type} instead.')

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images, labels = self.train_aug(images, labels)

        outputs = self.forward(images)
        loss = self.loss_fn(outputs, labels)

        with torch.no_grad():
            y_pred = outputs.sigmoid()
            y = one_hot(labels, num_classes=2)

            dice = self.dice_metric(y_pred=y_pred, y=y)

        return {"loss": loss, "dice": dice, "examples": len(outputs), "batches": 1}

    def training_epoch_end(self, outputs):
        mean_loss = outputs.get('loss') / outputs.get('examples')
        mean_dice = outputs.get('dice') / outputs.get('batches')
        return dict(loss=mean_loss, dice=mean_dice)
        
    def validation_step(self, batch, batch_idx):
        images, labels = batch

        # roi_size = (384, 384)
        # sw_batch_size = 4
        # outputs = sliding_window_inference(
        #    images, roi_size, sw_batch_size, self.forward)

        outputs = self.forward(images)
        loss = self.loss_fn(outputs, labels)
        
        y_pred = outputs.sigmoid()
        y = one_hot(labels, num_classes=2)

        dice = self.dice_metric(y_pred=y_pred, y=y)

        return {"loss": loss, "dice": dice, "examples": len(outputs), "batches": 1}

    def validation_epoch_end(self, outputs):
        mean_val_loss = outputs.get('loss') / outputs.get('examples')
        mean_val_dice = outputs.get('dice') / outputs.get('batches')
        return dict(loss=mean_val_loss, dice=mean_val_dice)

    def test_step(self, batch):
        images, labels = batch

        outputs = self.forward(images)
        
        y_pred = outputs.sigmoid().argmax(dim=1).detach().cpu()[0] == 1
        y = one_hot(labels, num_classes=2)
        
        y_pred = morphology.binary_closing(y_pred)
        blobs, num = measure.label(y_pred, background=0, return_num=True)
        sizes = [(blobs == i).sum() for i in range(1, num + 1)]
        
        if len(sizes) > 0:
            blob_index = np.argmax(sizes) + 1
            y_pred = (blobs == blob_index)

            hd = compute_hausdorff_distance(y_pred=y_pred, y=y)
            dice = dc(y_pred, y)

            return {"dice": dice, "hd": hd}
        else:
            return {"dice": np.nan, "hd": np.nan}

    def checkpoint(self, path: str):
        log = logging.getLogger(__name__)

        model_sd = (
            self._model.module.state_dict()
            if isinstance(self._model, nn.DataParallel)
            else self._model.state_dict()
        )

        torch.save(model_sd, path)
        log.info(f"Model checkpoint saved to {path}")

    def train(self):
        self._model.train()
    
    def eval(self):
        self._model.eval()

    @staticmethod
    def weights_init(layer: torch.tensor) -> None:
        """
        Initialize model weights
        Args:
            layer (torch.tensor): Model layer
        """
        module_types = [nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d]

        if any([isinstance(layer, _type) for _type in module_types]):
            nn.init.xavier_normal_(layer.weight.data)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias.data)
