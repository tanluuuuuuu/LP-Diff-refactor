import torch
import data as Data
import model as Model
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
import json

import warnings
warnings.filterwarnings("ignore")

opt = {
    "name": "LP-Diff",
    "phase": "train",
    "gpu_ids": 0,
    "distributed": False,
    "path": {
        "log": "logs",
        "results": "results",
        "checkpoint": "checkpoint",
        "resume_state": None,
    },
    "datasets": {
        "train": {
            "name": "ICPR",
            "mode": "LRHR",
            "dataroot": "data_converted_1000_112x56",
            "width": 112,
            "height": 56,
            "batch_size": 20,
            "num_workers": 0,
            "use_shuffle": True,
        },
        "val": {
            "name": "MDLP",
            "mode": "LRHR",
            "width": 112,
            "height": 56,
            "dataroot": "data_converted_1000_112x56",
            "use_shuffle": False,
        }
    },
    "model": {
        "finetune_norm": False,
        "unet": {
            "in_channel": 6,
            "out_channel": 3,
            "inner_channel": 32,
            "channel_multiplier": [
                1,
                2,
                4,
            ],
            "attn_res": [
                4
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
        "resume_training": False,
        "MTA": "./best_377.pt",
        "n_iter": 1000000,
        "val_freq": 2e4,
        "save_checkpoint_freq": 1e4,
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
    },
    "wandb": {
        "project": "LP-Diff"
    }
}

if __name__ == "__main__":
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
        # elif phase == 'val':
        #     val_set = Data.create_dataset(dataset_opt, phase)
        #     val_loader = Data.create_dataloader(
        #         val_set, dataset_opt, phase)
    print('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    print('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            print(f'Epoch {current_epoch}, Step {current_step}')
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                    print(message)

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    print('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

        # save model
        print('End of training.')
