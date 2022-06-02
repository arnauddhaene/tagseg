from typing import Dict, Any, List
from tqdm import tqdm
from pathlib import Path
import logging

import pandas as pd

from monai.networks import one_hot

import torchio as tio

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tagseg.models.cyclegan import Generator
from tagseg.models.segmenter import Net
from tagseg.models.trainer import Trainer


def tag_subjects(
    dataset: tio.SubjectsDataset, transformation_params: Dict[str, Any]
):
    log = logging.getLogger(__name__)

    subjects: List[tio.Subject] = []

    feats = list(dataset[0].keys())
    feats.remove('image')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nc: int = 1  # no. of channels
    generator: torch.nn.Module = Generator(nc, nc)

    saved_model: str = transformation_params["generator_model"]
    generator.load_state_dict(torch.load(saved_model))
    generator.to(device)

    log.info(f"Loaded {str(generator.__class__)} from {saved_model}.")

    generator.eval()

    batch_size = transformation_params['batch_size']

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        t = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        cines = batch.get('image').get('data').squeeze(1)
        tagged = Variable(t(*cines.shape).copy_(cines)).to(device)
        tagged = (0.5 * generator(tagged).data + 1.0).unsqueeze(1)

        span = range(i * batch_size, i * batch_size + cines.shape[0])
        for j, subject_idx in enumerate(span):
            subjects.append(
                tio.Subject({
                    'image': tio.ScalarImage(tensor=tagged[j].detach().cpu()),
                    **dict(zip(feats, map(dataset[subject_idx].get, feats)))
                })
            )

    return tio.SubjectsDataset(subjects)


def evaluate(
    tagged_mnm: tio.SubjectsDataset,
    tagged_scd: tio.SubjectsDataset,
    dmd: tio.SubjectsDataset,
    eval_params: Dict[Any, str]
) -> Dict[str, tio.SubjectsDataset]:

    model = Net(
        load_model=eval_params.get('load_model'),
        model_type=eval_params.get('model_type')
    )

    ds_map = {
        'mnm_results': tagged_mnm,
        'scd_results': tagged_scd,
        'dmd_results': dmd
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        model._model.to(device)
        model._model = nn.DataParallel(model._model)

    frames: Dict[str, pd.DataFrame] = {}
    dsets: Dict[str, tio.SubjectsDataset] = {}

    for key, dataset in ds_map.items():

        storage: List[Dict[str, Any]] = []
        subjects: List[tio.Subject] = []

        features = list(dataset[0].keys())
        features.remove('image')
        features.remove('mask')

        for subject in dataset:

            batch = subject['image'][tio.DATA], subject['mask'][tio.DATA]
            batch = Trainer.tensor_tuple_to(device, batch)
            image, label = batch

            image = 0.18098551 * (image - image.mean()) / image.std() + 0.7507453
        
            preds = model.forward(image)

            y_pred = preds.sigmoid()
            y = one_hot(label, num_classes=2)

            dice = model.dice_metric(y_pred=y_pred, y=y)

            storage.append({
                'dice': dice.item(),
                **dict(zip(features, map(subject.get, features)))
            })

            subjects.append(tio.Subject({
                'image': tio.ScalarImage(tensor=image.detach().cpu()),
                'mask': tio.LabelMap(tensor=label.detach().cpu()),
                'pred': tio.LabelMap(tensor=y_pred.argmax(dim=1).unsqueeze(0).detach().cpu()),
                'dice': dice.item(),
                **dict(zip(features, map(subject.get, features)))
            }))

        frames[key] = pd.DataFrame(storage)
        dsets[key] = tio.SubjectsDataset(subjects)

    save_dir = Path(eval_params.get('save_path'))
    Path.mkdir(save_dir)

    for key, value in frames.items():
        value.to_csv(save_dir / f'{key}.csv')

    return dsets
