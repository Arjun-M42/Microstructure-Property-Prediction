# Microstructure Property Prediction with ViT

This project predicts microstructure properties from 2D slices extracted from 3D `.mat` volumes using a Vision Transformer (ViT) regression model.

The current pipeline:

1. Reads 3D binary microstructure volumes from `data/train/*.mat`.
2. Extracts orthogonal 2D slices along the `XY`, `XZ`, and `YZ` planes.
3. Resizes each slice to `224 x 224` and stores it as a PNG image.
4. Loads 9 regression targets per structure from `kappa_train.mat`.
5. Trains a ViT model on the slice images.
6. Aggregates slice-level predictions back to structure-level predictions during evaluation/inference.

## Project Structure

```text
Project2/
|-- data/
|   |-- train/
|   |-- val/
|   |-- test/
|   |-- kappa_train.mat
|   |-- kappa_val.mat
|   |-- kappa_test.mat
|   |-- m_train.mat
|   |-- m_val.mat
|   `-- m_test.mat
|-- models/
|-- processed_data/
|   |-- train_images/
|   |-- val_images/
|   `-- test_images/
|-- src/
|   |-- dataset.py
|   |-- model.py
|   |-- predict.py
|   `-- train.py
|-- convert_mat_to_images.py
|-- requirements.txt
`-- README.md
```

## End-to-End Pipeline

### 1. Raw volumetric data

Each file in `data/train`, `data/val`, or `data/test` is a MATLAB `.mat` file containing a 3D volume. The preprocessing script expects the microstructure volume to be stored under the key `BW`.

### 2. Slice generation

`convert_mat_to_images.py` converts each 3D structure into multiple 2D slices:

- `XY` slices by stepping through the `z` axis
- `XZ` slices by stepping through the `y` axis
- `YZ` slices by stepping through the `x` axis

Current preprocessing settings:

- `SLICE_STEP = 24`
- `IMG_SIZE = 224`

Each output image is saved with a name like:

```text
structure_125_XY_48.png
```

That filename encodes:

- structure id: `125`
- plane: `XY`
- slice index: `48`

### 3. Label loading

`src/dataset.py` loads targets from `kappa_train.mat` using `h5py`.

Important detail:

- the code expects the target file to contain `9` values per structure
- labels are reshaped from a flat vector into `(num_structures, 9)`
- labels are transformed with `log1p`
- training uses per-dimension normalization based only on the training split

### 4. Dataset preparation

`MicrostructureDataset`:

- reads grayscale PNGs
- converts them into 3 channels for ViT input
- applies ImageNet normalization
- matches each image back to its parent structure id
- returns:
  - image tensor of shape `(3, 224, 224)`
  - target tensor of shape `(9,)`
  - `structure_id`

### 5. Training

`src/train.py` trains a ViT regression model with:

- pretrained backbone: `google/vit-base-patch16-224`
- output dimension: `9`
- loss: `SmoothL1Loss`
- optimizer: `AdamW`
- mixed precision on CUDA
- cosine decay learning rate schedule with warmup
- gradient clipping
- structure-level validation split to avoid leakage between slices of the same structure

Validation is done by:

- predicting on all validation slices
- averaging predictions for each structure
- computing structure-level MAE after reversing target normalization

Checkpoints saved to `models/`:

- `vit_best.pth`
- `vit_last.pth`

### 6. Prediction

`src/predict.py`:

- loads the best checkpoint
- restores the saved target normalizer
- predicts on the image dataset
- averages slice predictions for each structure
- prints the first few structure-level predictions vs targets

## Requirements

Install dependencies from `requirements.txt`:

```text
torch
torchvision
transformers
numpy
opencv-python
scipy
h5py
```

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## How to Run

Run all commands from the project root:

```powershell
cd c:\College\6th-Sem\MiniProject\Project2
```

### Step 1. Generate training images

```powershell
python convert_mat_to_images.py
```

This reads:

- `data/train/*.mat`

and writes:

- `processed_data/train_images/*.png`

### Step 2. Train the model

```powershell
python src/train.py
```

This will:

- load `processed_data/train_images`
- load labels from `data/kappa_train.mat`
- create an internal train/validation split by structure id
- train the ViT regressor
- save checkpoints in `models/`

### Step 3. Run prediction

```powershell
python src/predict.py
```

This will:

- load `models/vit_best.pth`
- run inference on `processed_data/train_images`
- print structure-level predicted vs actual values

## Notes About Validation and Test Data

The repository contains:

- `data/val`
- `data/test`
- `data/kappa_val.mat`
- `data/kappa_test.mat`

But the current preprocessing script only generates images for `data/train`.

So right now:

- training uses `processed_data/train_images`
- validation is created internally from the training structures
- `processed_data/val_images` and `processed_data/test_images` are not populated by default

If you want separate validation or test image folders, update `convert_mat_to_images.py` to point to:

- `data/val` -> `processed_data/val_images`
- `data/test` -> `processed_data/test_images`

and run the script again for each split.

## Example: Generate Images for Validation or Test

The easiest way is to edit these two lines in `convert_mat_to_images.py`:

```python
INPUT_DIR = "data/train"
OUTPUT_DIR = "processed_data/train_images"
```

For validation:

```python
INPUT_DIR = "data/val"
OUTPUT_DIR = "processed_data/val_images"
```

For test:

```python
INPUT_DIR = "data/test"
OUTPUT_DIR = "processed_data/test_images"
```

Then rerun:

```powershell
python convert_mat_to_images.py
```

## Key Files

- [convert_mat_to_images.py](</c:/College/6th-Sem/MiniProject/Project2/convert_mat_to_images.py:1>) converts 3D `.mat` microstructures into 2D PNG slices
- [src/dataset.py](</c:/College/6th-Sem/MiniProject/Project2/src/dataset.py:18>) loads labels, normalizes targets, and prepares image tensors
- [src/model.py](</c:/College/6th-Sem/MiniProject/Project2/src/model.py:4>) builds the pretrained ViT regression model
- [src/train.py](</c:/College/6th-Sem/MiniProject/Project2/src/train.py:116>) trains the model and saves checkpoints
- [src/predict.py](</c:/College/6th-Sem/MiniProject/Project2/src/predict.py:18>) restores the trained model and prints predictions

## Current Assumptions

- each structure has exactly 9 target values in `kappa_*.mat`
- the structure id in an image filename matches the corresponding row in the target matrix
- the raw `.mat` volume uses the key `BW`
- the PNG images are already stored at `224 x 224`

## Troubleshooting

### `No PNG files found`

Generate the images first with:

```powershell
python convert_mat_to_images.py
```

### `Expected label count ... to be divisible by 9`

The current dataset code assumes each structure has 9 regression targets. If your label format changes, update the reshape logic in [src/dataset.py](</c:/College/6th-Sem/MiniProject/Project2/src/dataset.py:18>).

### CUDA not available

The code will fall back to CPU automatically, but training will be much slower.

## Recommended Run Order

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python convert_mat_to_images.py
python src/train.py
python src/predict.py
```
