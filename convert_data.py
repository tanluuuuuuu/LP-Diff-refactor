import os
import shutil
from pathlib import Path
from PIL import Image
import re
from tqdm import tqdm

folder_path = [
    "data_input/train/Scenario-A/Brazilian",
    "data_input/train/Scenario-A/Mercosur",
    "data_input/train/Scenario-B/Brazilian",
    "data_input/train/Scenario-B/Mercosur"
]

DEST_INPUTS = "data_converted/train/inputs"
DEST_GT = "data_converted/train/gt"

# Target dimensions from config
TARGET_WIDTH = 224
TARGET_HEIGHT = 112


for start_folder in folder_path:
    for track_folder in tqdm(os.listdir(start_folder)):
        if track_folder.startswith('.'):
           continue 
        dest_folder_path = os.path.join(DEST_INPUTS, track_folder)
        os.makedirs(dest_folder_path, exist_ok=True)
        
        dest_folder_path = os.path.join(DEST_GT, track_folder)
        os.makedirs(dest_folder_path, exist_ok=True)

        for idx in range(1, 6):
            # Try both png and jpg extensions
            for ext in ['.png', '.jpg']:
                source_lr_file = os.path.join(start_folder, track_folder, f"lr-00{idx}{ext}")
                if os.path.exists(source_lr_file):
                    # Resize and convert to JPG
                    img = Image.open(source_lr_file)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # Convert RGBA/LA/P to RGB
                        img = img.convert('RGB')
                    img_resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                    dest_lr_file = os.path.join(DEST_INPUTS, track_folder, f"lr-00{idx}.jpg")
                    img_resized.save(dest_lr_file, 'JPEG', quality=95)
                    
        for ext in ['.png', '.jpg']:
            source_hr_file = os.path.join(start_folder, track_folder, f"hr-005{ext}")
            if os.path.exists(source_hr_file):
                # Resize and convert to JPG
                img = Image.open(source_hr_file)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Convert RGBA/LA/P to RGB
                    img = img.convert('RGB')
                img_resized = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
                dest_hr_file = os.path.join(DEST_GT, track_folder, f"hr-005.jpg")
                img_resized.save(dest_hr_file, 'JPEG', quality=95)