import os
import shutil
from pathlib import Path
from PIL import Image
import re
from tqdm import tqdm
import random

folder_path = [
    "data/ICPR_train/Scenario-A/Brazilian",
    "data/ICPR_train/Scenario-A/Mercosur",
    "data/ICPR_train/Scenario-B/Brazilian",
    "data/ICPR_train/Scenario-B/Mercosur"
]


DEST_INPUTS_TRAIN = "data_converted_1000_112x56/train/inputs"
DEST_GT_TRAIN = "data_converted_1000_112x56/train/gt"
DEST_INPUTS_VAL = "data_converted_1000_112x56/val/inputs"
DEST_GT_VAL = "data_converted_1000_112x56/val/gt"

# Target dimensions from config
TARGET_WIDTH = 112
TARGET_HEIGHT = 56

track_folder_paths = []
for start_folder in folder_path:
    for track_folder in tqdm(os.listdir(start_folder)):
        if track_folder.startswith('.'):
            continue
        track_folder_paths.append(os.path.join(start_folder, track_folder))

# random.shuffle(track_folder_paths)
track_folder_paths = sorted(track_folder_paths)

track_folder_paths = track_folder_paths[:1000]
track_train_folder_paths = track_folder_paths[:800]
track_val_folder_paths = track_folder_paths[800:]


def convert_data(track_folder_paths, dest_inputs, dest_gt):
    for track_folder_path in track_folder_paths:
        # Extract the track folder name from the full path
        track_folder = os.path.basename(track_folder_path)

        dest_folder_path = os.path.join(dest_inputs, track_folder)
        os.makedirs(dest_folder_path, exist_ok=True)

        dest_folder_path = os.path.join(dest_gt, track_folder)
        os.makedirs(dest_folder_path, exist_ok=True)

        for idx in range(1, 6):
            # Try both png and jpg extensions
            for ext in ['.png', '.jpg']:
                source_lr_file = os.path.join(
                    track_folder_path, f"lr-00{idx}{ext}")
                if os.path.exists(source_lr_file):
                    # Resize and convert to JPG
                    img = Image.open(source_lr_file)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # Convert RGBA/LA/P to RGB
                        img = img.convert('RGB')
                    img_resized = img.resize(
                        (TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                    dest_lr_file = os.path.join(
                        dest_inputs, track_folder, f"lr-00{idx}.jpg")
                    img_resized.save(dest_lr_file, 'JPEG', quality=95)
                    break  # Found and processed the file, no need to try other extensions

        for ext in ['.png', '.jpg']:
            source_hr_file = os.path.join(track_folder_path, f"hr-005{ext}")
            if os.path.exists(source_hr_file):
                # Resize and convert to JPG
                img = Image.open(source_hr_file)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Convert RGBA/LA/P to RGB
                    img = img.convert('RGB')
                img_resized = img.resize(
                    (TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                dest_hr_file = os.path.join(
                    dest_gt, track_folder, f"hr-005.jpg")
                img_resized.save(dest_hr_file, 'JPEG', quality=95)
                break  # Found and processed the file, no need to try other extensions


convert_data(track_train_folder_paths, DEST_INPUTS_TRAIN, DEST_GT_TRAIN)
convert_data(track_val_folder_paths, DEST_INPUTS_VAL, DEST_GT_VAL)
