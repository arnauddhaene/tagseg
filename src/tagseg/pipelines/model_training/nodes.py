import logging
from typing import Any, Dict

import aim
import kornia.augmentation as K
import torch
from kedro.config import ConfigLoader
from kornia.utils import one_hot
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from tagseg.metrics.metrics import DiceLoss, ShapeLoss, dice_score, evaluate
from tagseg.models.unet import UNet
from tagseg.data.utils import directional_field


def load_model(data_params: Dict[str, nn.Module]):

    log = logging.getLogger(__name__)
    conf_params = ConfigLoader("conf/base").get("parameters*", "parameters*/**")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    only_myo = data_params["only_myo"]
    model = UNet(n_channels=1, n_classes=(2 if only_myo else 4), bilinear=True).double()

    pretrained_path = conf_params["pretrained_model"]
    if pretrained_path is not None:
        # Load old saved version of the model as a state dictionary
        saved_model_sd = torch.load(pretrained_path)
        # Extract UNet if saved model is parallelized
        model.load_state_dict(saved_model_sd)
        log.info(f"Weights loaded from saved model at {pretrained_path}")

    if device.type == "cuda":
        model = nn.DataParallel(model)
        model.n_classes = model.module.n_classes
        log.info("Model parallelized on CUDA")

    return dict(model=model, device=device)


def train_model(
    model: nn.Module,
    loader_train: DataLoader,
    loader_val: DataLoader,
    device: torch.device,
    train_params: Dict[str, Any],
    experiment_name: str,
) -> Dict[str, nn.Module]:

    log = logging.getLogger(__name__)
    conf_params = ConfigLoader("conf/base").get("parameters*", "parameters*/**")
    if conf_params["data_params"]["only_myo"]:
        index_to_class = dict(zip(range(2), ["BG", "MYO"]))
    else:
        index_to_class = dict(zip(range(4), ["BG", "LV", "MYO", "RV"]))

    run = aim.Run(experiment=experiment_name)
    run["hparams"] = {**conf_params}

    log.info(f"Created aim.Run instance of name {experiment_name}")

    amp = True
    model = model.to(device)

    proba: float = 0.2

    train_aug = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=proba),
        K.RandomVerticalFlip(p=proba),
        K.RandomElasticTransform(p=proba),
        K.RandomGaussianNoise(p=proba),
        K.RandomSharpness(p=proba),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.1), p=proba),
        data_keys=["input", "mask"],
    )

    # Define loss
    criterion = nn.CrossEntropyLoss()
    dice_criterion = DiceLoss(exclude_bg=True)
    shape_criterion = ShapeLoss(exclude_bg=True)

    def loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            criterion(outputs, targets)
            + dice_criterion(outputs, targets)
            + 1e-3 * shape_criterion(outputs, targets)
        )

    learning_rate: float = train_params["learning_rate"]
    weight_decay: float = train_params["weight_decay"]
    # momentum: float = train_params["momentum"]
    epochs: int = train_params["epochs"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        # momentum=momentum,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    run["hparams"] = {
        **run["hparams"],
        "optimizer": optimizer.__class__.__name__,
        "main_criterion": criterion.__class__.__name__,
        "secondary_criterion": [
            dice_criterion.__class__.__name__,
            shape_criterion.__class__.__name__,
        ],
        "device": device.type,
        "augmentation": str(train_aug),
        "model": model.__class__.__name__,
    }

    log.info(f"Launching training of {model.__class__.__name__} for {epochs} epochs")

    pbar = tqdm(range(epochs), unit="epoch", leave=False)
    for epoch in pbar:

        dice = torch.zeros(model.n_classes).to(device)
        acc_loss = 0.0

        model.train()

        batch_pbar = tqdm(
            loader_train, total=len(loader_train), unit="batch", leave=False
        )
        for inputs, targets in batch_pbar:
            optimizer.zero_grad(set_to_none=True)

            batch_pbar.set_description(f"Acummulated loss: {acc_loss:.4f}")
            
            df_inp = one_hot(targets.long(), model.n_classes).numpy()

            # move to device
            # target is index of classes
            inputs, targets = inputs.double().to(device), targets.to(device)

            # Run augmentation pipeline every batch
            inputs, targets = train_aug(inputs, targets.unsqueeze(1))
            targets = targets.squeeze(1).long()
            target_dfs = torch.Tensor(directional_field(df_inp)).double().to(device)

            with torch.cuda.amp.autocast(enabled=amp):
                outputs, dfs, auxsegs = model(inputs)
                loss = loss_fn(outputs, targets) \
                    + F.mse_loss(dfs, target_dfs) \
                    + F.cross_entropy(auxsegs, targets)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            dice += dice_score(outputs, targets)
            acc_loss += loss.item()

        acc_loss /= len(loader_train)

        # Tracking training performance
        run.track(acc_loss, name="loss", epoch=epoch, context=dict(subset="train"))

        train_perf = dice / len(loader_train)
        avg_dice = train_perf[1:].mean()

        for i, val in enumerate(train_perf):
            run.track(
                val,
                name=f"dice_{index_to_class[i]}",
                epoch=epoch,
                context=dict(subset="train"),
            )

        status = f"Epoch {epoch:03} \t Loss {acc_loss:.4f} \t Dice {avg_dice:.4f}"

        # Tracking validation performance
        val_perf, val_loss = evaluate(
            model,
            loader_val,
            loss_fn,
            track_images=((epoch + 1) % 5 == 0),
            run=run,
            device=device,
        )
        run.track(val_loss, name="loss", epoch=epoch, context=dict(subset="val"))
        avg_val_dice = val_perf[1:].mean()
        scheduler.step(avg_val_dice)

        for i, val in enumerate(val_perf):
            run.track(
                val,
                name=f"dice_{index_to_class[i]}",
                epoch=epoch,
                context=dict(subset="val"),
            )

        status += f"\t Val. Loss {val_loss:.4f} \t Val. Dice {avg_val_dice:.4f}"

        pbar.set_description(status)

    return model


def save_model(model: nn.Module, path: str) -> None:
    log = logging.getLogger(__name__)

    model_sd = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    torch.save(model_sd, path)

    log.info(f"Trained model saved to {path}")
