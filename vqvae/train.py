import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger

from ffcv.fields import RGBImageField
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
from ffcv.transforms import ToTensor, ToTorchImage

from ffcv_pl.data_loading import FFCVDataModule
from ffcv_pl.ffcv_utils.augmentations import DivideImage255
from ffcv_pl.ffcv_utils.decoders import FFCVDecoders

from vqvae.model import VQVAE
from data.datamodules import ImageDataModule

import argparse
import math
import os.path
import wandb
import yaml


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--params_file', type=str, required=True, help='path to yaml file with model params')
    parser.add_argument('--dataloader', type=str, choices=['standard', 'ffcv'], default='standard',
                        help='defines what type of dataloader to use.')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='path to a dataset folder containing two sub-folders (validation / train) or beton files '
                             '(train.beton / validation.beton).')
    parser.add_argument('--save_path', type=str, required=True, help='path for checkpointing the model')
    parser.add_argument('--save_every_n_epochs', type=int, default=1, help='how often to save a new checkpoint')
    parser.add_argument('--run_name', type=str, required=True,
                        help='name of the run, for wandb logging and checkpointing')
    parser.add_argument('--seed', type=int, required=True, help='global random seed for reproducibility')
    parser.add_argument('--loading_path', type=str,
                        help='if passed, will load and continue training of an existing checkpoint', default=None)
    parser.add_argument('--logging', help='if passed, wandb logger is used', action='store_true')
    parser.add_argument('--wandb_project', type=str, help='project name for wandb logger', default='vqvae')
    parser.add_argument('--wandb_id', type=str,
                        help='wandb id of the run. Useful for resuming logging of a model', default=None)
    parser.add_argument('--workers', type=int, help='num of parallel workers', default=1)

    return parser.parse_args()


def get_model_conf(filepath: str):
    # load params
    with open(filepath, 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)

    return params


def get_datamodule(loader_type: str, dirpath: str, image_size: int, batch_size: int, workers: int, seed: int,
                   is_dist: bool):
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(f"dataset path not found: {dirpath}")

    else:

        if loader_type == 'standard':

            train_folder = f'{dirpath}train/'
            val_folder = f'{dirpath}validation/'
            return ImageDataModule(image_size, batch_size, workers, train_folder, val_folder)

        elif loader_type == 'ffcv':

            train_file = f'{dirpath}train.beton'
            val_file = f'{dirpath}validation.beton'
            train_decoder = FFCVDecoders(image_transforms=[CenterCropRGBImageDecoder((image_size, image_size), ratio=1),
                                                           ToTensor(), ToTorchImage(),
                                                           DivideImage255(dtype=torch.float32)])
            val_decoder = FFCVDecoders(image_transforms=[CenterCropRGBImageDecoder((image_size, image_size), ratio=1),
                                                         ToTensor(), ToTorchImage(),
                                                         DivideImage255(dtype=torch.float32)])

            return FFCVDataModule(batch_size, workers, is_dist, (RGBImageField,), train_file, val_file,
                                  train_decoders=train_decoder, val_decoders=val_decoder, seed=seed)

        else:
            raise ValueError(f"loader type not recognized: {loader_type}")


def set_matmul_precision():
    """
    If using Ampere Gpus enable using tensor cores
    """

    gpu_cores = os.popen('nvidia-smi -L').readlines()[0]

    if 'A100' in gpu_cores.lower():
        torch.set_float32_matmul_precision('high')
        print('[INFO] set matmul precision "high"')


def main():

    # only for A100
    set_matmul_precision()

    args = parse_args()
    conf = get_model_conf(args.params_file)

    # configuration params (assumes some env variables in case of multi-node setup)
    num_nodes = int(os.getenv('NODES')) if os.getenv('NODES') is not None else 1
    gpus = torch.cuda.device_count()
    rank = int(os.getenv('NODE_RANK')) if os.getenv('NODE_RANK') is not None else 0
    is_dist = gpus > 1 or num_nodes > 1

    workers = int(args.workers)
    seed = int(args.seed)

    cumulative_batch_size = int(conf['training']['cumulative_bs'])
    batch_size_per_device = cumulative_batch_size // (num_nodes * gpus)

    base_learning_rate = float(conf['training']['base_lr'])
    learning_rate = base_learning_rate * math.sqrt(cumulative_batch_size / 256)

    max_epochs = int(conf['training']['max_epochs'])

    pl.seed_everything(seed, workers=True)

    # logging stuff, checkpointing and resume
    log_to_wandb = bool(args.logging)
    project_name = str(args.wandb_project)
    wandb_id = args.wandb_id

    run_name = str(args.run_name)
    save_checkpoint_dir = f'{args.save_path}{run_name}/'
    save_every_n_epochs = int(args.save_every_n_epochs)

    load_checkpoint_path = args.loading_path
    resume = load_checkpoint_path is not None

    if rank == 0:  # prevents from logging multiple times
        logger = WandbLogger(project=project_name, name=run_name, offline=not log_to_wandb, id=wandb_id,
                             resume='must' if resume else None)
    else:
        logger = WandbLogger(project=project_name, name=run_name, offline=True)

    # model params
    image_size = int(conf['image_size'])
    ae_conf = conf['autoencoder']
    q_conf = conf['quantizer']
    l_conf = conf['loss'] if 'loss' in conf.keys() else None
    t_conf = {'lr': learning_rate,
              'betas': conf['training']['betas'],
              'eps': conf['training']['eps'],
              'weight_decay': conf['training']['weight_decay'],
              'warmup_epochs': conf['training']['warmup_epochs'] if 'warmup_epochs' in conf['training'].keys() else None,
              'decay_epochs': conf['training']['decay_epochs'] if 'decay_epochs' in conf['training'].keys() else None,
              }

    # get model
    if resume:
        # image_size: int, ae_conf: dict, q_conf: dict, l_conf: dict, t_conf: dict, init_cb: bool = True,
        #                  load_loss: bool = True
        model = VQVAE.load_from_checkpoint(load_checkpoint_path, strict=False,
                                           image_size=image_size, ae_conf=ae_conf, q_conf=q_conf, l_conf=l_conf,
                                           t_conf=t_conf, init_cb=False, load_loss=True)
    else:
        model = VQVAE(image_size=image_size, ae_conf=ae_conf, q_conf=q_conf, l_conf=l_conf, t_conf=t_conf,
                      init_cb=True, load_loss=True)

    # data loading (standard pytorch lightning or ffcv)
    datamodule = get_datamodule(args.dataloader, args.dataset_path, image_size, batch_size_per_device,
                                workers, seed, is_dist)

    # callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=save_checkpoint_dir, filename='{epoch:02d}', save_last=True,
                                          save_top_k=-1, every_n_epochs=save_every_n_epochs)

    callbacks = [LearningRateMonitor(), checkpoint_callback]

    # trainer
    trainer = pl.Trainer(strategy='ddp', accelerator='gpu', num_nodes=num_nodes, devices=gpus,
                         callbacks=callbacks, deterministic=True, logger=logger, max_epochs=max_epochs)

    print(f"[INFO] workers: {workers}")
    print(f"[INFO] batch size per device: {batch_size_per_device}")
    print(f"[INFO] cumulative batch size (all devices): {cumulative_batch_size}")
    print(f"[INFO] final learning rate: {learning_rate}")

    trainer.fit(model, datamodule, ckpt_path=load_checkpoint_path)

    # ensure wandb has stopped logging
    wandb.finish()


if __name__ == '__main__':
    main()
