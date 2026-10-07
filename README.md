# 3D Brain Tumour Segmentation (BraTS 2020)

Volumetric segmentation of gliomas in multimodal MRI with a **3D U-Net**, trained on the [BraTS 2020 dataset](https://www.kaggle.com/awsaf49/brats20-dataset-training-validation).

## Overview

- **Data preparation** (`brats2020_get_data_ready.py`, `Getting_data_ready_patches.py`): load the NIfTI scans, min-max normalise the modalities, stack them into multi-channel volumes, crop them to 128³ and save them as NumPy arrays.
- **Filtering** (`Filtering data_patches.py`): keep only volumes where at least 1% of voxels are labelled tumour, so training sees meaningful examples.
- **Data loading** (`custom_datagen.py`): a custom generator that streams 3D volumes in batches to fit GPU memory.
- **Model** (`simple_3d_unet.py`): a 3D U-Net with 4 output classes (background, necrotic/non-enhancing core, oedema, enhancing tumour).
- **Training** (`train_brats2020.py`): class-weighted **Dice loss + focal loss** against class imbalance, Adam optimiser, IoU monitoring, 100 epochs.

## Dataset

BraTS 2020 contains multi-institutional pre-operative MRI scans (T1, T1Gd, T2 and T2-FLAIR) with expert annotations of tumour sub-regions.

**Stack:** Python · TensorFlow/Keras · segmentation-models-3D · NiBabel · NumPy
