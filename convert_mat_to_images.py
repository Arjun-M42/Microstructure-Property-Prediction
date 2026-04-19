import os
import cv2
import numpy as np
from scipy.io import loadmat

INPUT_DIR = "data/train"
OUTPUT_DIR = "processed_data/train_images"

SLICE_STEP = 24
IMG_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(os.listdir(INPUT_DIR))

for idx, file in enumerate(files):

    data = loadmat(os.path.join(INPUT_DIR, file))
    volume = data["BW"]

    structure_id = file.replace(".mat", "")
    size = volume.shape[0]

    for z in range(0, size, SLICE_STEP):
        img = volume[:, :, z]
        img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(f"{OUTPUT_DIR}/{structure_id}_XY_{z}.png", img)

    for y in range(0, size, SLICE_STEP):
        img = volume[:, y, :]
        img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(f"{OUTPUT_DIR}/{structure_id}_XZ_{y}.png", img)

    for x in range(0, size, SLICE_STEP):
        img = volume[x, :, :]
        img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(f"{OUTPUT_DIR}/{structure_id}_YZ_{x}.png", img)

    print(f"Processed {idx+1}/{len(files)}")