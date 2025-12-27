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
    "gpu_ids": [
        0
    ],
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
            "dataroot": "data_converted",
            "width": 224,
            "height": 112,
            "batch_size": 20,
            "num_workers": 0,
            "use_shuffle": True,
        },
        "val": {
            "name": "MDLP",
            "mode": "LRHR",
            "width": 224,
            "height": 112,
            "dataroot": "data_converted",
            "use_shuffle": False,
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
        'gpu_ids': 0,
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

                # validation
                # if current_step % opt['train']['val_freq'] == 0:
                # # if current_step % 1 == 0:
                #     avg_psnr = 0.0
                #     idx = 0
                #     avg_val_loss = 0.0

                #     diffusion.set_new_noise_schedule(
                #         opt['model']['beta_schedule']['val'], schedule_phase='val')
                #     for _,  val_data in enumerate(val_loader):
                #         idx += 1
                #         diffusion.feed_data(val_data)
                #         loss = diffusion.test(continous=False)
                #         visuals = diffusion.get_current_visuals()
                #         sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                #         hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                #         lr1_img = Metrics.tensor2img(visuals['LR1'])  # uint8
                #         lr2_img = Metrics.tensor2img(visuals['LR2'])  # uint8
                #         lr3_img = Metrics.tensor2img(visuals['LR3'])  # uint8

                #         avg_psnr += Metrics.calculate_psnr(
                #             sr_img, hr_img)
                #         avg_val_loss += loss

                #     avg_psnr = avg_psnr / idx
                #     avg_val_loss = avg_val_loss / idx
                #     if avg_val_loss < best_loss and avg_psnr < best_psnr:
                #         best_loss = avg_val_loss
                #         diffusion.save_best_loss(current_epoch, current_step)
                #     elif avg_psnr > best_psnr and avg_val_loss > best_loss:
                #         best_psnr = avg_psnr
                #         diffusion.save_best_psnr(current_epoch, current_step)
                #     elif avg_val_loss <= best_loss and avg_psnr >= best_psnr:
                #         best_psnr = avg_psnr
                #         best_loss = avg_val_loss
                #         diffusion.save_best_both(current_epoch, current_step)
                        
                #     diffusion.set_new_noise_schedule(
                #         opt['model']['beta_schedule']['train'], schedule_phase='train')
                #     # log
                #     print('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                #     logger_val = logging.getLogger('val')  # validation logger
                #     logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} loss: {:.4e}'.format(
                #         current_epoch, current_step, avg_psnr, avg_val_loss))
                    

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    print('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

        # save model
        print('End of training.')
    else:
        print('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()
            
            # print('SR ', visuals['SR'])
            
            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr1_img = Metrics.tensor2img(visuals['LR1'])  # uint8
            lr2_img = Metrics.tensor2img(visuals['LR2'])  # uint8
            lr3_img = Metrics.tensor2img(visuals['LR3'])  # uint8
            # middle_img = Metrics.tensor2img(visuals['middle'])  # uint8
            
            filename = os.path.basename(os.path.split(diffusion.data['path'][0])[0])
            
            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_sr.png'.format(result_path, filename))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                Metrics.save_img(
                    sr_img, '{}/{}_sr_process.png'.format(result_path, filename))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_sr.png'.format(result_path, filename))
            
            Metrics.save_img(
                hr_img, '{}/{}_hr.png'.format(result_path, filename))
            Metrics.save_img(
                lr1_img, '{}/{}_lr1.png'.format(result_path, filename))
            Metrics.save_img(
                lr2_img, '{}/{}_lr2.png'.format(result_path, filename))
            Metrics.save_img(
                lr2_img, '{}/{}_lr3.png'.format(result_path, filename))
            # Metrics.save_img(
            #     middle_img, '{}/{}_middle.png'.format(result_path, filename))
            
            # folder_path = os.path.join(r'', filename)
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)
            # Metrics.save_img(
            #     Metrics.tensor2img(visuals['SR'][-1]), os.path.join(folder_path, filename + '.jpg'))
            

            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        print('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        print('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))