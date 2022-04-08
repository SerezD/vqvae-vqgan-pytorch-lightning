import pytorch_lightning as pl

import torch
from torch.nn import functional as F
from torchvision.utils import make_grid

from modules.encoders import VqVaeEncoder
from modules.decoders import VqVaeDecoder
from modules.vector_quantizers import VectorQuantizer

from utils.plotting import codebook_usage_barplot, codebook_multidimensional_scaling, codebook_tsne_proj, \
                           close_all_plots

import wandb


class VqVae(pl.LightningModule):
    """
    General VQ-VAE class, using EMA updating for codebook with decay=0.95
    """

    def __init__(self, num_downsamples, hiddens_size, num_residual_layers, image_size, num_embeddings, lr=2e-4):
        """
        :param num_downsamples: final image size will be reduced by a factor of 2**x, where x is this parameter.
        :param hiddens_size: num of channels after the encoding stage, equal to num of features in codebook vectors.
        :param num_residual_layers: num of residual layers to apply in both Encoder and Decoder.
        :param image_size: initial input_image size.
        :param num_embeddings: desired codebook dimension, aka number of vectors in the codebook.
        """

        super().__init__()

        self.hparams.lr = lr

        # init size parameters
        self.image_size = image_size
        self.hidden_image_size = (int(image_size[0] / (2 ** num_downsamples)),
                                  int(image_size[1] / (2 ** num_downsamples)))
        self.num_hiddens = self.hidden_image_size[0] * self.hidden_image_size[1]  # vectors per image
        self.num_embeddings = num_embeddings  # codebook vectors

        # network structure
        self.encoder = VqVaeEncoder(num_downsamples, hiddens_size, num_residual_layers)
        self.vec_quant = VectorQuantizer(num_embeddings, hiddens_size)
        self.decoder = VqVaeDecoder(hiddens_size, num_downsamples, hiddens_size, num_residual_layers)

        # sum of the used codebook indices at each epoch (resets with epoch)
        self.register_buffer('used_indices', torch.zeros(num_embeddings), persistent=False)

    def forward(self, x, x_idx):

        device = x.device

        z = self.encoder(x)

        e_loss, quantized, used_indices = self.vec_quant(z)

        # get reconstruction and loss
        x_recon = self.decoder(quantized)
        x_loss = F.mse_loss(x_recon, x)

        loss = e_loss.to(device) + x_loss

        return loss, e_loss.detach(), x_loss.detach(), x_recon, used_indices

    def training_step(self, batch, batch_idx):

        batch = batch[0] # for CIFAR10 example remove labels
        device = batch.device

        loss, e_loss, x_loss, x_recon, used_indices = self.forward(batch, batch_idx)

        # add this batch indices to epoch_probs. (scaled to avoid large integers - result does not change)
        self.used_indices = self.used_indices.to(device) + (used_indices / 1000.0).to(device)

        # logs
        if batch_idx == 0:
            train_samples = batch[:8]
            train_reconstructions = x_recon[:8]
            grid = make_grid(torch.cat((train_samples, train_reconstructions)))
            self.logger.experiment.log({'train/reconstructions': wandb.Image(grid)})

        self.log("train/e_loss", e_loss.item())
        self.log("train/rec_loss", x_loss.item())
        self.log("train/loss", loss.item())

        return {"loss": loss}

    def training_epoch_end(self, _):

        # probability that every vector in dict has to be chosen in last epoch, reset used indices
        indices_sum = torch.sum(self.used_indices, dim=0)
        avg_probs = self.used_indices / indices_sum
        self.used_indices = torch.zeros(self.num_embeddings)

        # perplexity is the exponential of entropy (the amount of information used in the encodings)
        # theoretically, it should be better to use a lot of information (high perplexity)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1)).sum()

        # get the percentage of used codebook
        used_codebook = (torch.count_nonzero(avg_probs).item() * 100) / self.num_embeddings

        # log results
        self.log("train/used_codebook", used_codebook)
        self.log("train/perplexity", perplexity)

        # create and log plots
        codebook_usage_bar = codebook_usage_barplot(y=avg_probs.cpu().numpy())
        codebook_mds, euc_matrix = codebook_multidimensional_scaling(self.get_codebook().weight)
        codebook_tsne = codebook_tsne_proj(self.get_codebook().weight)

        self.logger.experiment.log({'train_plots/codebook_probs': wandb.Image(codebook_usage_bar),
                                    'train_plots/codebook_mds': wandb.Image(codebook_mds),
                                    'train_plots/codebook_tsne': wandb.Image(codebook_tsne),
                                    'train_plots/codebook_euc_dist': wandb.Image(euc_matrix)
                                    })

        close_all_plots()

    def validation_step(self, batch, batch_idx):

        batch = batch[0]  # for CIFAR10 example remove labels

        loss, e_loss, x_loss, x_recon, _ = self.forward(batch, batch_idx)

        # save samples
        if batch_idx == 0:
            val_samples = batch[:8]
            val_reconstructions = x_recon[:8]
            grid = make_grid(torch.cat((val_samples, val_reconstructions)))
            self.logger.experiment.log({'validation/reconstructions': wandb.Image(grid)})

        # logs
        self.log("validation/e_loss", e_loss.item())
        self.log("validation/rec_loss", x_loss.item())
        self.log('val_loss', loss)

        return {"val_loss": loss}

    @torch.no_grad()
    def encode(self, x, quantize=False):
        """
        :param x: an image_batch to encode.
        :param quantize: quantize image or not.
        :return: the encoded or quantized version of x, and optionally its indices .
        """

        x = self.encoder(x)

        if quantize:
            return self.vec_quant.quantize(x)

        return x

    @torch.no_grad()
    def decode(self, z):
        """
        :param z: a quantized image_batch.
        :return: the reconstruction.
        """
        return self.decoder(z)

    @torch.no_grad()
    def get_codebook(self):
        """
        :return: the codebook parameter.
        """
        return self.vec_quant.get_codebook()

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
