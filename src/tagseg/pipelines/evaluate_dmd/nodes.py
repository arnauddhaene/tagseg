from typing import Dict, Any, List
from tqdm import tqdm
from pathlib import Path

import pandas as pd

from monai.networks import one_hot
from monai.metrics import compute_hausdorff_distance

from skimage import measure

import torchio as tio

import torch
from torch import nn
from torchvision import transforms

from kedro.extras.datasets.pickle import PickleDataSet

from tagseg.models.segmenter import Net
from tagseg.models.trainer import Trainer


def evaluate_dmd(
    dataset: tio.SubjectsDataset,
    eval_params: Dict[Any, str]
) -> tio.SubjectsDataset:

    ext = '_train' if len(dataset) > 300 else ''

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

        raw_shape = subject['raw_shape']
        voxel_spacing = subject['voxel_spacing']

        batch = subject['image'][tio.DATA], subject['raw_mask'][tio.DATA]
        batch = Trainer.tensor_tuple_to(device, batch)
        image, label = batch

        postprocess = transforms.Resize(
            raw_shape,
            interpolation=transforms.InterpolationMode.NEAREST
        )
    
        # TODO: decide if this is actually implemented
        preds = model.forward(image)
        y_pred = postprocess(preds.sigmoid())
        pred = y_pred.argmax(dim=1)[0]
        mask = label[0]

        # Add 1 as 0 is considered background otherwise
        npred = measure.label(pred.cpu() + 1)
        nmask = measure.label(mask.cpu() + 1)

        y_myo = one_hot(torch.tensor(nmask == 2).reshape(1, 1, *raw_shape), num_classes=2)
        y_lv = one_hot(torch.tensor(nmask == 3).reshape(1, 1, *raw_shape), num_classes=2)

        y_pred_myo = one_hot(torch.tensor(npred == 2).reshape(1, 1, *raw_shape), num_classes=2)
        y_pred_lv = one_hot(torch.tensor(npred == 3).reshape(1, 1, *raw_shape), num_classes=2)

        dice_myo = model.dice_metric(y_pred=y_pred_myo, y=y_myo)
        dice_lv = model.dice_metric(y_pred=y_pred_lv, y=y_lv)
        hd95_myo = compute_hausdorff_distance(
            one_hot(y_pred_myo.argmax(dim=1, keepdim=True), num_classes=2), y_myo,
            include_background=False, percentile=95)
        hd95_lv = compute_hausdorff_distance(
            one_hot(y_pred_lv.argmax(dim=1, keepdim=True), num_classes=2), y_lv,
            include_background=False, percentile=95)

        storage.append({
            'dice_myo': dice_myo.item(),
            'dice_lv': dice_lv.item(),
            'hd95_myo': hd95_myo.item() * voxel_spacing,
            'hd95_lv': hd95_lv.item() * voxel_spacing,
            **dict(zip(features, map(subject.get, features)))
        })

        subjects.append(tio.Subject({
            'image': tio.ScalarImage(tensor=image.detach().cpu()),
            'raw_mask': tio.LabelMap(tensor=label.detach().cpu()),
            'pred': tio.LabelMap(tensor=y_pred.argmax(dim=1).unsqueeze(0).detach().cpu()),
            'dice_myo': dice_myo.item(),
            'dice_lv': dice_lv.item(),
            'hd95_myo': hd95_myo.item() * voxel_spacing,
            'hd95_lv': hd95_lv.item() * voxel_spacing,
            **dict(zip(features, map(subject.get, features)))
        }))

    model_name = Path(eval_params.get('load_model')).stem
    csv_save_dir = Path('data/08_reporting') / model_name
    pt_save_dir = Path('data/07_model_output') / model_name
    Path.mkdir(csv_save_dir, exist_ok=True)
    Path.mkdir(pt_save_dir, exist_ok=True)

    pd.DataFrame(storage).to_csv(csv_save_dir / ('dmd_results' + ext + '.csv'))
    PickleDataSet(filepath=str(pt_save_dir / ('dmd_results' + ext + '.pt'))).save(
        tio.SubjectsDataset(subjects))

    return "Suck it"
