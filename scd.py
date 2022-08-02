from pathlib import Path

import pydicom

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st


def get_points(path: Path) -> np.ndarray:
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    
    # remove last line if empty
    if lines[-1] == '':
        lines = lines[:-1]
    return np.array([x_y.split(' ') for x_y in lines]).astype(np.float64)


path_patientdata = Path('data/01_raw/scd_patientdata.xlsx')

df = pd.read_excel(path_patientdata)
patient_to_original = dict(zip(df.PatientID, df.OriginalID))
patients = df.PatientID

patient = st.selectbox('Select Patient ID', patients)
original = patient_to_original[patient]
original = original[:-1] + '0' + original[-1:]
st.write(f"""Corresponding original ID: `{original}`""")

contours = Path(f'data/01_raw/SCD_ManualContours/{original}/contours-manual/IRCCI-expert/').iterdir()

conts = {}

for cont in contours:

    _, _, no, trace, _ = cont.stem.split('-')

    item = dict(trace=trace, name=str(cont))

    if no not in conts.keys():
        conts[no] = [item]
    else:
        conts[no].append(item)

cont_no = st.selectbox('Select group of contours', conts.keys())
selected_cont = conts[cont_no]

images = []

image_folders = [p for p in (Path('data/01_raw/SCD_DeidentifiedImages') / patient).iterdir() if p.is_dir()]

for imfol in image_folders:
    _, _, _, _, dirx, diry, xdim, ydim, zdim = imfol.name.split('_')
    
    nos_paths = [path for path in imfol.iterdir() if path.suffix == '.dcm']
    nos = list(map(lambda p: p.stem.split('_')[-1], nos_paths))

    images.append(
        dict(name=imfol.name, dirx=dirx, diry=diry, xdim=xdim, ydim=ydim, zdim=zdim,
             nos=list(np.array(nos).astype(np.int32)), nos_paths=nos_paths)
    )
    
coll = pd.DataFrame(images)
important = list(coll.columns)
important.remove('name')
important.remove('nos_paths')
st.write(coll[important])

im, no = st.columns([4, 1])

# image_folder = im.selectbox('Select image folder', image_folders, format_func=lambda p: p.name)
image_folder_idx = im.selectbox('Select image folder', range(len(image_folders)))
image_folder = image_folders[image_folder_idx]
pot_numbers = coll[coll.name == image_folder.name].nos.iloc[0]
number = no.selectbox('Select image number', pot_numbers)

image_path = image_folder / \
    coll[coll.name == image_folder.name].nos_paths.iloc[0][pot_numbers.index(number)].name

if image_path.is_file():

    img = pydicom.dcmread(image_path).pixel_array

    st.write(f'Image shape: {img.shape}')

    fig, ax = plt.subplots()

    ax.imshow(img, cmap='gray')

    st.write(f'{len(selected_cont)} contour(s) available')
    for _contour in selected_cont:
        pts = get_points(Path(_contour['name']))
        ax.scatter(pts[:, 0], pts[:, 1], marker='.', s=2)

    ax.axis('off')

    st.pyplot(fig)
