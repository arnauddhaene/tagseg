from typing import List
from pathlib import Path

import streamlit as st
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

from kedro.extras.datasets.pickle import PickleDataSet
from kedro.config import ConfigLoader

from tagseg.models.unet import UNet


conf_loader = ConfigLoader(["conf/base", "conf/local"])
catalog = conf_loader.get("catalog*", "catalog*/**")


@st.cache(allow_output_mutation=True)
def load_data(tagged: bool = True, only_myo: bool = True) -> TensorDataset:
    ds_name: str = "acdc_data"
    ds_name += "_tagged" if tagged else "_cine"
    ds_name += "_only_myo" if only_myo else ""

    dataset = PickleDataSet(filepath=catalog[ds_name]["filepath"])

    return dataset.load()


@st.cache()
def load_model(model_path: str, only_myo: bool = True) -> nn.Module:
    model: nn.Module = UNet(n_channels=1, n_classes=2 if only_myo else 4, bilinear=True).double()
    # Load old saved version of the model as a state dictionary
    saved_model_sd = torch.load(model_path)
    # Extract UNet if saved model is parallelized
    model.load_state_dict(saved_model_sd)
    return model


st.title('Performance evaluator')

r1, r2 = st.columns(2)

tagged: bool = r1.radio('Image style', ('Tagged', 'Cine')) == 'Tagged'
only_myo: bool = r2.radio('Image style', ('Only Myo', 'All')) == 'Only Myo'
dataset = load_data(tagged=tagged, only_myo=only_myo)

model_store: List[Path] = Path('data/06_models').iterdir()
model_path: str = st.selectbox('Select model', model_store)

model = load_model(model_path, only_myo)

img: int = st.slider('Select datapoint', value=1, min_value=0, max_value=len(dataset), step=1)

inp: torch.Tensor = dataset[img][0].unsqueeze(0).double().clone()
out: torch.Tensor = model(inp)
out = F.softmax(out, dim=1).argmax(dim=1).detach().numpy()[0]

fig, axes = plt.subplots(1, 3)

axes[0].imshow(dataset[img][0][0], cmap='gray')
axes[0].set_title('Input image')
axes[1].imshow(dataset[img][1])
axes[1].set_title('Ground truth mask')
axes[2].imshow(out)
axes[2].set_title('Predicted mask')

for ax in axes:
    ax.axis('off')

st.pyplot(fig)
