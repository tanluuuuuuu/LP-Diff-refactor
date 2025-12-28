# train_lp_diff.py
import modal

# Create Modal app
app = modal.App("lp-diff-training")

# Create volumes for data and checkpoints
data_volume = modal.Volume.from_name("icpr2026-data", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(
    "icpr2026", create_if_missing=True)

# Define the image with all dependencies
# image = (
#     modal.Image.debian_slim(python_version="3.10")
#     .apt_install("git", "unzip")
#     .pip_install(
#         "albumentations==1.3.1",
#         "glog==0.3.1",
#         "matplotlib==3.8.0",
#         "numpy==1.24.4",
#         "opencv_contrib_python==4.7.0.72",
#         "opencv_python==4.11.0.86",
#         "opencv_python_headless==4.8.1.78",
#         "Pillow==11.2.1",
#         "tensorboardX==2.6.4",
#         "torch>=2.4.0",
#         "torchvision==0.19.0",
#         "tqdm==4.66.5"
#     )
#     .run_commands("cd /root && git clone https://github.com/tanluuuuuuu/LP-Diff-refactor.git || true")
# )
pytorch_image = modal.Image.from_registry("pytorch/pytorch").apt_install(
    "git",
    "libgl1",
    "libglib2.0-0"
).pip_install(
    "albumentations==1.3.1",
    "glog==0.3.1",
    "matplotlib==3.8.0",
    "numpy==1.24.4",
    "opencv_contrib_python==4.7.0.72",
    "opencv_python==4.11.0.86",
    "opencv_python_headless==4.8.1.78",
    "Pillow==11.2.1",
    "tqdm==4.66.5"
)


@app.function(
    image=pytorch_image,
    gpu="A100",  # or "A10G", "T4", "L4", etc.
    volumes={
        "/mnt/icpr2026": data_volume,
        "/root/checkpoints": checkpoint_volume,
    },
    timeout=86400,  # 24 hours
)
def train_model():
    """Long-running training function for LP-Diff model"""
    import torch
    import sys
    import subprocess
    import os

    # Clone LP-Diff repository if not exists
    lp_diff_path = "/root/LP-Diff-refactor"
    if not os.path.exists(lp_diff_path):
        print("Cloning LP-Diff-refactor repository...")
        subprocess.run([
            "git", "clone",
            "https://github.com/tanluuuuuuu/LP-Diff-refactor.git",
            lp_diff_path
        ], check=True)
        print("Repository cloned successfully")
    else:
        print("LP-Diff-refactor already exists, skipping clone")

    # Change to LP-Diff directory and add to path
    os.chdir(lp_diff_path)
    sys.path.insert(0, lp_diff_path)

    # Unzip data if not already extracted
    data_zip = "/mnt/icpr2026/data_converted_1000.zip"
    data_dir = "/mnt/icpr2026/data_converted_1000"

    if os.path.exists(data_zip) and not os.path.exists(data_dir):
        print(f"Unzipping {data_zip}...")
        subprocess.run(["unzip", data_zip, "-d", "/mnt/icpr2026/"], check=True)
        data_volume.commit()  # Save the extracted files to volume
        print("Data unzipped successfully")
    elif os.path.exists(data_dir):
        print("Data already unzipped, skipping extraction")
    else:
        raise FileNotFoundError(f"Data zip file not found at {data_zip}")

    import data as Data
    import model as Model
    import logging
    import core.logger as Logger
    import core.metrics as Metrics
    import numpy as np
    import warnings
    from tqdm import tqdm
    warnings.filterwarnings("ignore")

    # Configuration from your notebook
    opt = {
        "name": "LP-Diff",
        "phase": "train",
        "gpu_ids": 0,
        "distributed": False,
        "path": {
            "log": "/root/checkpoints/logs",
            "results": "/root/checkpoints/results",
            "checkpoint": "/root/checkpoints/checkpoints",
            "resume_state": "/root/checkpoints/checkpoints/I10800_E153",
        },
        "datasets": {
            "train": {
                "name": "ICPR",
                "mode": "LRHR",
                "dataroot": "/mnt/icpr2026/data_converted_1000",
                "width": 224,
                "height": 112,
                "batch_size": 40,
                "num_workers": 0,
                "use_shuffle": True,
            },
            "val": {
                "name": "ICPR",
                "mode": "LRHR",
                "dataroot": "/mnt/icpr2026/data_converted_1000",
                "width": 224,
                "height": 112,
                "batch_size": 1,
                "num_workers": 1,
                "use_shuffle": False,
                "data_len": -1,
            }
        },
        "model": {
            "finetune_norm": False,
            "unet": {
                "in_channel": 6,
                "out_channel": 3,
                "inner_channel": 64,
                "channel_multiplier": [
                    1,
                    2,
                    4,
                    8,
                    8
                ],
                "attn_res": [
                    16
                ],
                "res_blocks": 2,
                "dropout": 0.1
            },
            "beta_schedule": {
                "train": {
                    "schedule": "linear",
                    "n_timestep": 1000,
                    "linear_start": 1e-6,
                    "linear_end": 1e-2
                },
                "val": {
                    "schedule": "linear",
                    "n_timestep": 1000,
                    "linear_start": 1e-6,
                    "linear_end": 1e-2
                }
            },
            "diffusion": {
                "image_size": 72,
                "channels": 3,
                "conditional": True
            }
        },
        "train": {
            "use_prerain_MTA": False,
            "resume_training": True,
            "MTA": "./best_377.pt",
            "n_iter": 1000000,
            "val_freq": 1000001,
            "save_checkpoint_freq": 200,
            "print_freq": 200,
            "optimizer": {
                "type": "adam",
                "lr": 5e-3
            },
            "ema_scheduler": {
                "step_start_ema": 5000,
                "update_ema_every": 1,
                "ema_decay": 0.9999
            }
        }
    }

    config = {
        'phase': 'train',
        'debug': False,
    }

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and config['phase'] != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    print('Initial Dataset Finished')

    # Create model
    diffusion = Model.create_model(opt)
    print('Model created')

    # Training loop
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        print('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            print(f'Epoch {current_epoch}, Step {current_step}')
            for _, train_data in tqdm(enumerate(train_loader)):
                current_step += 1
                if current_step > n_iter:
                    break

                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()

                # Log training info
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                    print(message)

                # Validation
                # if current_step % opt['train']['val_freq'] == 0:
                #     print(
                #         'Begin validation at step: {}'.format(current_step))
                #     diffusion.set_new_noise_schedule(
                #         opt['model']['beta_schedule']['val'], schedule_phase='val')

                #     avg_psnr = 0.0
                #     idx = 0
                #     for val_data in val_loader:
                #         idx += 1
                #         diffusion.feed_data(val_data)
                #         diffusion.test(continous=False)
                #         visuals = diffusion.get_current_visuals()

                #         sr_img = Metrics.tensor2img(visuals['SR'])
                #         hr_img = Metrics.tensor2img(visuals['HR'])

                #         # Calculate metrics
                #         avg_psnr += Metrics.calculate_psnr(sr_img, hr_img)

                #     avg_psnr = avg_psnr / idx
                #     print('# Validation # PSNR: {:.4e}'.format(avg_psnr))

                #     # Switch back to training schedule
                #     diffusion.set_new_noise_schedule(
                #         opt['model']['beta_schedule']['train'], schedule_phase='train')

                # Save checkpoint
                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    print(
                        'Saving models and training states at step: {}'.format(current_step))
                    diffusion.save_network(current_epoch, current_step)
                    # Commit checkpoints to volume
                    checkpoint_volume.commit()
                    print('Checkpoint saved and committed to volume')

        print('Training completed!')
        # Final save
        print('Saving final model...')
        diffusion.save_network(current_epoch, current_step)
        checkpoint_volume.commit()

    return {
        "status": "completed",
        "final_epoch": current_epoch,
        "final_step": current_step,
        "total_iterations": n_iter
    }


@app.local_entrypoint()
def main():
    """Entry point for local execution"""
    print("Starting training on Modal...")
    result = train_model.remote()
    print(f"Training result: {result}")
