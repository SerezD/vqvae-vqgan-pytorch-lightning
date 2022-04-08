import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models.vqvae import VqVae

import wandb

if __name__ == '__main__':

    SEED = 4219

    batch_size = 128
    workers = 8
    gpus = 1

    pl.seed_everything(SEED, workers=True)

    # init dataset (CIFAR10 in this example)
    image_size = (32, 32)
    training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                     transform=transforms.Compose(
                                         [transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))]))

    validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                       transform=transforms.Compose(
                                           [transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))]))

    train_dl = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=workers)
    val_dl = DataLoader(validation_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=workers)

    # define model
    num_downsamples = 2  # encoding layers (final encodings will have size: image_size / 2**num_downsamples
    hiddens_size = 256   # final channels = dimension of final vectors
    num_residual_layers = 2
    num_embeddings = 512       # total size of the codebook

    vq_vae = VqVae(num_downsamples=num_downsamples, hiddens_size=hiddens_size, num_residual_layers=num_residual_layers,
                   image_size=image_size, num_embeddings=num_embeddings)

    run_name = 'CB=' + str(num_embeddings) + '-DIM=' + str(hiddens_size)
    wandb_logger = WandbLogger(project='vq-vae', name=run_name, offline=True)

    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10)

    checkpoint_callback = ModelCheckpoint(dirpath='./runs/vq-vae/', filename=run_name + '-{epoch:02d}',
                                          monitor='val_loss', save_top_k=1)

    trainer = pl.Trainer(callbacks=[early_stop_callback, checkpoint_callback],
                         strategy=DDPPlugin(find_unused_parameters=False), deterministic=True,
                         logger=wandb_logger, gpus=gpus)

    trainer.fit(vq_vae, train_dl, val_dl)

    wandb.finish()
