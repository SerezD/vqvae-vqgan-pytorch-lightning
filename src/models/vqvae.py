import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid
from torchmetrics.functional import multiscale_structural_similarity_index_measure as mssim

from src.modules.encoders import VqVaeEncoder
from src.modules.decoders import VqVaeDecoder
from src.modules.vector_quantizers import EMAVectorQuantizer
from src.utils.metrics import get_codebook_usage
from src.utils.plotting import close_all_plots, codebook_usage_barplot

import wandb


class VQVAE(pl.LightningModule):

    def __init__(self, num_downsamples, latent_size, num_residual_layers, image_size, num_embeddings, lr=2e-4):
        """
        :param num_downsamples: final image size will be reduced by a factor of 2**x, where x is this parameter.
        :param latent_size: num of channels after the encoding stage, equal to num of features in codebook vectors.
        :param num_residual_layers: num of residual layers to apply in both Encoder and Decoder.
        :param image_size: initial input_image size.
        :param num_embeddings: desired codebook dimension, aka number of vectors in the codebook.
        """

        super().__init__()

        self.hparams.lr = lr

        # init size parameters
        self.image_size = image_size
        self.hidden_image_size = image_size // (2 ** num_downsamples)
        self.num_hiddens = self.hidden_image_size**2  # vectors per image
        self.num_embeddings = num_embeddings  # codebook vectors

        # network structure
        self.encoder = VqVaeEncoder(num_downsamples, latent_size, num_residual_layers)
        self.vec_quant = EMAVectorQuantizer(num_embeddings, latent_size)
        self.decoder = VqVaeDecoder(latent_size, num_downsamples, latent_size, num_residual_layers)

    def forward(self, x):

        device = x.device

        z = self.encoder(x)

        e_loss, quantized, used_indices = self.vec_quant(z)

        # get reconstruction and loss
        x_recon = self.decoder(quantized)
        x_loss = F.mse_loss(x_recon, x)

        loss = e_loss.to(device) + x_loss

        return loss, e_loss.detach(), x_loss.detach(), x_recon, used_indices

    def training_step(self, batch, batch_index):

        loss, e_loss, rec_loss, x_recon, _ = self.forward(batch)

        self.log_reconstructions(batch_index, batch, x_recon, t_or_v='t')

        return {'loss': loss, 'rec_loss': rec_loss, 'e_loss': e_loss}

    def training_epoch_end(self, outputs):

        device = outputs[0]['loss'].device

        # get losses
        loss = torch.empty((0, 1), device=device)
        rec_loss = torch.empty((0, 1), device=device)
        e_loss = torch.empty((0, 1), device=device)

        for out in outputs:
            loss = torch.cat((loss, out['loss'].view(1, 1)), dim=0)
            rec_loss = torch.cat((rec_loss, out['rec_loss'].view(1, 1)), dim=0)
            e_loss = torch.cat((e_loss, out['e_loss'].view(1, 1)), dim=0)

        loss = torch.mean(loss.unsqueeze(1))
        rec_loss = torch.mean(rec_loss.unsqueeze(1))
        e_loss = torch.mean(e_loss.unsqueeze(1))

        self.log('train_loss', loss, sync_dist=True)
        self.log('train/r_loss', rec_loss, sync_dist=True)
        self.log('train/l_loss', e_loss, sync_dist=True)

        return

    def validation_step(self, batch, batch_index):

        loss, e_loss, rec_loss, x_recon, used_indices = self.forward(batch)

        self.log_reconstructions(batch_index, batch, x_recon, t_or_v='v')

        # compute SSIM and MSE
        mssim_score = mssim(batch.to(torch.float32), x_recon.to(torch.float32)).cpu()
        mse_score = F.mse_loss(x_recon, batch).cpu()

        # use non-deterministic for this operation
        torch.use_deterministic_algorithms(False)
        used_indices = torch.bincount(used_indices.view(-1), minlength=self.num_embeddings)
        torch.use_deterministic_algorithms(True)

        return {'loss': loss, 'rec_loss': rec_loss, 'e_loss': e_loss, 'mssim': mssim_score, 'mse': mse_score,
                'idx_counts': used_indices}

    def validation_epoch_end(self, outputs):

        device = outputs[0]['idx_counts'][0].device

        # get losses
        loss = torch.empty((0, 1), device=device)
        rec_loss = torch.empty((0, 1), device=device)
        e_loss = torch.empty((0, 1), device=device)

        # get metrics
        mssim_score = torch.empty((0, 1))
        mse_score = torch.empty((0, 1))

        # initialize epoch used_index count for all sequences
        epoch_count = torch.zeros(self.num_embeddings, device=device)

        for out in outputs:
            loss = torch.cat((loss, out['loss'].view(1, 1)), dim=0)
            rec_loss = torch.cat((rec_loss, out['rec_loss'].view(1, 1)), dim=0)
            e_loss = torch.cat((e_loss, out['e_loss'].view(1, 1)), dim=0)

            mssim_score = torch.cat((mssim_score, out['mssim'].view(1, 1)), dim=0)
            mse_score = torch.cat((mse_score, out['mse'].view(1, 1)), dim=0)

            epoch_count = epoch_count + out['idx_counts']

        loss = torch.mean(loss.unsqueeze(1))
        rec_loss = torch.mean(rec_loss.unsqueeze(1))
        e_loss = torch.mean(e_loss.unsqueeze(1))
        mssim_score = torch.mean(mssim_score.unsqueeze(1))
        mse_score = torch.mean(mse_score.unsqueeze(1))

        self.log('validation_loss', loss, sync_dist=True)
        self.log('validation_mssim', mssim_score, sync_dist=True)
        self.log('validation_mse', mse_score, sync_dist=True)
        self.log('validation/r_loss', rec_loss, sync_dist=True)
        self.log('validation/l_loss', e_loss, sync_dist=True)

        used_idx, perplexity, cb_usage = get_codebook_usage(epoch_count)

        # log results
        self.log(f"val_metrics/used_codebook", cb_usage, sync_dist=True)
        self.log(f"val_metrics/perplexity", perplexity, sync_dist=True)

        # create and log plots
        codebook_usage_bar = codebook_usage_barplot(y=used_idx.cpu().numpy())
        self.logger.experiment.log({f'val_metrics/codebook_histo': wandb.Image(codebook_usage_bar)})
        close_all_plots()

        return

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @torch.no_grad()
    def log_reconstructions(self, batch_index, ground_truths, reconstructions, t_or_v='t'):

        if batch_index == 2 and self.current_epoch % 5 == 1:
            b, _, _, _ = ground_truths.shape
            panel_name = 'train_imgs' if t_or_v == 't' else 'validation_imgs'

            grid = make_grid(ground_truths, nrow=min(b, 8))
            self.logger.experiment.log({f'{panel_name}/ground_truths': wandb.Image(grid)})

            grid = make_grid(reconstructions, nrow=min(b, 8))
            self.logger.experiment.log({f'{panel_name}/reconstructions': wandb.Image(grid)})

    @torch.no_grad()
    def get_indices(self, images: torch.Tensor):
        """
        :param images: B, 3, H, W
        :return B, S batch of codebook indices
        """

        return self(images)[-1]

    @torch.no_grad()
    def get_reconstructions(self, images: torch.Tensor):
        """
        :param images: B, 3, H, W
        :return (N, B, 3, H, W)
        """

        return self(images)[-2]

    @torch.no_grad()
    def reconstruct_from_indices(self, indices: list):
        """
        :param indices: N, B, S where N is the number of sequences and S the single sequence len (from coarse to fine)
                        if N is < than max number of sequences, reconstructs only for the first specified N steps
        :return N, B, 3, H, W.
        """
        pass
