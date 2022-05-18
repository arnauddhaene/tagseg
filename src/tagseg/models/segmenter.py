import torch
from torch import nn
# from torch.utils.data.dataloader import default_collate

# import matplotlib.pyplot as plt

# from monai.transforms import AsDiscrete, Compose, Activations
from monai.networks import nets, one_hot
# from monai.networks.layers import Norm
# from monai.data import decollate_batch
# from monai.metrics import DiceMetric, compute_meandice , HausdorffDistanceMetric
from monai.losses import DiceCELoss
# from monai.inferers import sliding_window_inference

import kornia.augmentation as K

from ..metrics import DiceMetric


class Net():

    def __init__(
        self,
        load_model: str = None,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-3
    ) -> None:
        super(Net, self).__init__()
        # Add hparams as attribute
        self.__dict__.update(
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )

        self.num_classes = 2

        # self._model = nets.UNet(
        #     in_channels=1,
        #     out_channels=self.num_classes,
        #     spatial_dims=2,
        #     channels=(16, 32, 64, 128, 256),
        #     strides=(2, 2, 2, 2),
        #     num_res_units=2,
        #     norm=Norm.BATCH
        # ).double()

        self._model = nets.SegResNetVAE(
            in_channels=1, out_channels=self.num_classes, input_image_size=(256, 256), spatial_dims=2
        ).double()

        # Weight initialization
        # self._model.apply(self.weights_init)

        # Load checkpointed version of the model
        if load_model is not None:
            self._model.load_state_dict(torch.load(load_model))

        self.criterion = DiceCELoss(include_background=False, to_onehot_y=True, sigmoid=True)

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=3)
        return optimizer, scheduler

    def loss_fn(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(outputs, targets)

    def forward(self, x) -> torch.Tensor:
        return self._model(x)[0]

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
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight.data)
            nn.init.zeros_(layer.bias.data)
