import logging
from typing import Any, Dict

import aim
import kornia.augmentation as K
import torch
from kedro.config import ConfigLoader
from monai.networks import nets
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tagseg.metrics.metrics import DiceLoss, ShapeLoss, dice_score, evaluate

# from tagseg.models.unet_ss import UNetSS


def load_model(training_params: Dict[str, Any]) -> Dict[str, Any]:

    log = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nets.SegResNetVAE(
        in_channels=1, out_channels=2, input_image_size=(256, 256), spatial_dims=2
    ).double()

    pretrained_path = training_params["pretrain_model"]
    if pretrained_path is not None:
        # Load old saved version of the model as a state dictionary
        saved_model_sd = torch.load(pretrained_path)
        # Extract UNet if saved model is parallelized
        model.load_state_dict(saved_model_sd)
        log.info(f"Weights loaded from saved model at {pretrained_path}")

    if device.type == "cuda":
        model = nn.DataParallel(model)
        model.n_classes = 2  # model.module.n_classes
        log.info("Model parallelized on CUDA")

    return dict(model=model, device=device)


def train_model(
    model: nn.Module,
    loader_train: DataLoader,
    loader_train_ss: DataLoader,
    loader_val: DataLoader,
    device: torch.device,
) -> Dict[str, nn.Module]:

    log = logging.getLogger(__name__)
    conf_params = ConfigLoader("conf/base").get("parameters*", "parameters*/**")

    index_to_class = dict(zip(range(2), ["BG", "MYO"]))

    experiment_name = conf_params["experiment"]["name"]

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

    ss_aug = K.AugmentationSequential(
        K.RandomGaussianNoise(p=proba),
        K.RandomSharpness(p=proba),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.1), p=proba),
    )

    # Define loss
    criterion = nn.CrossEntropyLoss()
    dice_criterion = DiceLoss(exclude_bg=True)
    shape_criterion = ShapeLoss(exclude_bg=True)

    # Self-supervised cosine embedding loss
    ss_criterion = nn.CosineEmbeddingLoss()

    def loss_fn(
        outputs: Dict[str, torch.Tensor], targets: torch.Tensor
    ) -> torch.Tensor:
        return (
            criterion(outputs["logits"], targets)
            + dice_criterion(outputs["logits"], targets)
            + 1e-3 * shape_criterion(outputs["logits"], targets)
        )

    def ss_loss_fn(
        outputs_a: Dict[str, torch.Tensor], outputs_b: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return ss_criterion(
            outputs_a["intermediate"].flatten(start_dim=1, end_dim=3),
            outputs_b["intermediate"].flatten(start_dim=1, end_dim=3),
            torch.ones(conf_params["batch_size"]).to(device),
        )

    learning_rate: float = conf_params["training"]["learning_rate"]
    weight_decay: float = conf_params["training"]["weight_decay"]
    # momentum: float = train_params["momentum"]
    epochs: int = conf_params["training"]["epochs"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
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

            # move to device
            # target is index of classes
            inputs, targets = inputs.double().to(device), targets.to(device)

            # Run augmentation pipeline every batch
            inputs, targets = train_aug(inputs, targets.unsqueeze(1))
            targets = targets.squeeze(1).long()

            with torch.cuda.amp.autocast(enabled=amp):
                outputs = dict(logits=model(inputs)[0])
                loss = loss_fn(outputs, targets)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            dice += dice_score(outputs["logits"], targets)
            acc_loss += loss.item()

        total_samples = len(loader_train)

        if conf_params["training"]["self_supervised"]:
            batch_pbar_ss = tqdm(
                loader_train_ss, total=len(loader_train_ss), unit="batch", leave=False
            )
            for inputs in batch_pbar_ss:
                optimizer.zero_grad(set_to_none=True)

                batch_pbar.set_description(
                    f"Acummulated loss (self-supervised): {acc_loss:.4f}"
                )

                # move to device
                # target is index of classes
                inputs = inputs.double().to(device)

                # Run augmentation pipeline every batch
                inputs, _ = train_aug(inputs, torch.empty_like(inputs))
                augs = ss_aug(inputs)

                with torch.cuda.amp.autocast(enabled=amp):
                    outputs_a = model(inputs)
                    outputs_b = model(augs)

                    loss = ss_loss_fn(outputs_a, outputs_b)

                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                acc_loss += loss.item()

            total_samples += len(loader_train_ss)

        acc_loss /= total_samples

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
