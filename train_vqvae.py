import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from src.models.vqvae import VQVAE
from datasets.datamodules import ImageDataModule

import wandb

import argparse
import yaml


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--params_file', type=str, help='path to yaml file with model params')
    parser.add_argument('--dataset_path', type=str, help='path to dataset folder containing two folders (test / train)')
    parser.add_argument('--checkpoint_path', type=str, help='path to checkpoint folder')
    parser.add_argument('--run_name', type=str, help='name of the run, for wandb logging and checkpointing')
    parser.add_argument('--logging', type=bool, help='if False, wandb logger runs offline', default=True)
    parser.add_argument('--seed', type=int, help='global random seed for reproducibility', default=1111)
    parser.add_argument('--epochs', type=int, help='total num epochs', default=-1)
    parser.add_argument('--gpus', type=int, help='num of requested gpus', default=1)
    parser.add_argument('--workers', type=int, help='num of parallel workers', default=1)

    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    # load params
    with open(args.params_file, 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)

    # load dataset
    image_size = params['dataset']['image_size']
    batch_size = params['dataset']['batch_size']
    train_folder = f'{args.dataset_path}train/'
    test_folder = f'{args.dataset_path}test/'

    datamodule = ImageDataModule(train_folder, test_folder, image_size, batch_size, args.workers)

    # define model
    model_params = params['model']
    lr = float(params['hparams']['lr'])

    vq_vae = VQVAE(num_downsamples=model_params['num_downsamples'], latent_size=model_params['latent_size'],
                   num_residual_layers=model_params['num_residual_layers'], image_size=image_size,
                   num_embeddings=model_params['num_embeddings'], lr=lr)

    run_name = args.run_name
    wandb_logger = WandbLogger(project='vq-vae', name=run_name, offline=not args.logging)

    # callbacks
    early_stop_callback = EarlyStopping(monitor='validation_mse', patience=params['hparams']['early_stopping_patience'])

    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_path, filename=run_name + '-{epoch:02d}',
                                          monitor='validation_mse', save_top_k=1)

    trainer = pl.Trainer(callbacks=[early_stop_callback, checkpoint_callback], accelerator='gpu',
                         strategy=DDPStrategy(find_unused_parameters=False), deterministic=True,
                         logger=wandb_logger, devices=args.gpus, max_epochs=args.epochs)

    trainer.fit(vq_vae, datamodule=datamodule)

    wandb.finish()
