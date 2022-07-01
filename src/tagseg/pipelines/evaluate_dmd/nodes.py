from typing import Dict, Any, List
from tqdm import tqdm
from pathlib import Path

import pandas as pd

from monai.networks import one_hot
from monai.metrics import compute_hausdorff_distance

import torchio as tio

import torch
from torch import nn

from tagseg.models.segmenter import Net
from tagseg.models.trainer import Trainer


def evaluate_dmd(
    dataset: tio.SubjectsDataset,
    eval_params: Dict[Any, str]
) -> tio.SubjectsDataset:

    model = Net(
        load_model=eval_params.get('load_model'),
        model_type=eval_params.get('model_type')
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        model._model.to(device)
        model._model = nn.DataParallel(model._model)

    storage: List[Dict[str, Any]] = []
    subjects: List[tio.Subject] = []

    features = list(dataset[0].keys())
    features.remove('image')
    features.remove('mask')

    for subject in tqdm(dataset, total=len(dataset)):

        batch = subject['image'][tio.DATA], subject['mask'][tio.DATA]
        batch = Trainer.tensor_tuple_to(device, batch)
        image, label = batch
    
        preds = model.forward(image)

        y_pred = preds.sigmoid()
        y = one_hot(label, num_classes=2)

        dice = model.dice_metric(y_pred=y_pred, y=y)
        hd95 = compute_hausdorff_distance(
            one_hot(y_pred.argmax(dim=1, keepdim=True), num_classes=2), y,
            include_background=False, percentile=95)

        storage.append({
            'dice': dice.item(),
            'hd95': hd95.item(),
            **dict(zip(features, map(subject.get, features)))
        })

        subjects.append(tio.Subject({
            'image': tio.ScalarImage(tensor=image.detach().cpu()),
            'mask': tio.LabelMap(tensor=label.detach().cpu()),
            'pred': tio.LabelMap(tensor=y_pred.argmax(dim=1).unsqueeze(0).detach().cpu()),
            'dice': dice.item(),
            'hd95': hd95.item(),
            **dict(zip(features, map(subject.get, features)))
        }))

    save_dir = Path(eval_params.get('save_path'))
    Path.mkdir(save_dir, exist_ok=True)

    pd.DataFrame(storage).to_csv(save_dir / 'dmd_results_train.csv')

    return tio.SubjectsDataset(subjects)
