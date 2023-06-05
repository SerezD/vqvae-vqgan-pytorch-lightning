from typing import Any

import pytorch_lightning as pl
import torch
from einops import rearrange, pack
from scheduling_utils.schedulers import LinearScheduler, CosineScheduler, LinearCosineScheduler
from torchvision.utils import make_grid
import wandb

from vqvae.modules.abstract_modules.base_autoencoder import BaseVQVAE
from vqvae.modules.autoencoder import Encoder, Decoder
from vqvae.modules.loss.loss import VQLPIPSWithDiscriminator
from vqvae.modules.vector_quantizers import VectorQuantizer, EMAVectorQuantizer, GumbelVectorQuantizer, \
    EntropyVectorQuantizer


class VQVAE(BaseVQVAE, pl.LightningModule):

    def __init__(self, image_size: int, ae_conf: dict, q_conf: dict, l_conf: dict or None, t_conf: dict,
                 init_cb: bool = True, load_loss: bool = True):
        """
        :param image_size: resolution of (squared) input images

        :param ae_conf: encoder/decoder parameters
                        channels: int
                        num_res_blocks: int
                        channel_multipliers: tuple (for each downsampling level)

        :param q_conf: quantizer type and configuration.
                       num_embeddings: int,
                       embedding_dim: int,
                       type: choice ['standard', 'ema', 'gumbel', 'entropy']
                       params: dict depending on type
                                standard --> {commitment_cost: float}
                                ema --> {commitment_cost: float, decay: float, epsilon: float}
                                gumbel --> {straight_through: bool, temp: float, kl_cost: float,
                                            kl_warmup_epochs: int or None,
                                            temp_decay_epochs: int or None,
                                            temp_final: float or None}
                                entropy --> {ent_loss_ratio: float, ent_temperature: float,
                                             ent_loss_type: str = 'softmax' or 'argmax',
                                             commitment_cost: float}

        :param l_conf: if None will use standard mse loss.
                       Otherwise (VQGAN case), must specify:
                          l1_weight: float --> weight for log_laplace loss
                          l2_weight: float --> weight for standard l2 loss
                          perc_weight: float --> weight for perceptual loss
                          adversarial_params: dict
                              start_epoch: int --> suggestion is to wait at least one epoch.
                              loss_type: str = "hinge" or "non-saturating"
                              adaptive_weight: float --> weight for generator loss, modulated according to gradients
                              r1_reg_weight: float --> weight for R1 regularization of Discriminator
                              r1_reg_every: int --> R1 regularization to be applied every n steps

        :param t_conf: training parameters, containing:
                        lr: float,
                        betas: tuple[float, ...],
                        eps: float,
                        weight_decay: float,
                        warmup_epoch: int or None,
                        decay_epoch: int or None,

                       E.G. warmup_epoch = 5 will apply linear warmup of learning rate from epoch 0 to 5,
                       while decay_epoch = 100 will apply cosine decay from epoch 0 (or warmup_epoch) to epoch 100.

        :param init_cb: pass False when loading the model
        :param load_loss: may pass False in inference mode, will allow overall parameter reduction
        """

        super().__init__(image_size=image_size)

        # training parameters
        self.t_conf = t_conf

        # quantizer
        self.cb_size = q_conf['num_embeddings']
        self.latent_dim = q_conf['embedding_dim']

        if q_conf['type'] == 'standard':

            self.quantizer = VectorQuantizer(self.cb_size, self.latent_dim, float(q_conf['params']['commitment_cost']))
            self.kl_warmup_epochs = None
            self.temp_decay_epochs = None
            self.temp_final = None

        elif q_conf['type'] == 'ema':
            self.quantizer = EMAVectorQuantizer(self.cb_size, self.latent_dim,
                                                float(q_conf['params']['commitment_cost']),
                                                float(q_conf['params']['decay']), float(q_conf['params']['epsilon']))
            self.kl_warmup_epochs = None
            self.temp_decay_epochs = None
            self.temp_final = None

        elif q_conf['type'] == 'gumbel':

            self.quantizer = GumbelVectorQuantizer(self.cb_size, self.latent_dim,
                                                   bool(q_conf['params']['straight_through']),
                                                   float(q_conf['params']['temp']), float(q_conf['params']['kl_cost']))
            self.kl_warmup_epochs = q_conf['params']['kl_warmup_epochs']
            self.temp_decay_epochs = q_conf['params']['temp_decay_epochs']
            self.temp_final = q_conf['params']['temp_final']

        elif q_conf['type'] == 'entropy':

            self.quantizer = EntropyVectorQuantizer(self.cb_size, self.latent_dim,
                                                    float(q_conf['params']['ent_loss_ratio']),
                                                    float(q_conf['params']['ent_temperature']),
                                                    str(q_conf['params']['ent_loss_type']),
                                                    float(q_conf['params']['commitment_cost']))
            self.kl_warmup_epochs = None
            self.temp_decay_epochs = None
            self.temp_final = None
        else:
            raise ValueError(f'unrecognized quantizer: {q_conf["type"]}')

        # Autoencoder
        channels = ae_conf['channels']
        num_res_blocks = ae_conf['num_res_blocks']
        channel_multipliers = ae_conf['channel_multipliers']
        final_conv_channels = self.cb_size if q_conf['type'] == 'gumbel' else self.latent_dim
        self.encoder = Encoder(channels, num_res_blocks, channel_multipliers, final_conv_channels)
        self.decoder = Decoder(channels, num_res_blocks, channel_multipliers, final_conv_channels)

        # Loss
        if load_loss:
            if l_conf is None:
                self.criterion = torch.nn.MSELoss()
            else:
                self.criterion = VQLPIPSWithDiscriminator(image_size, l_conf['l1_weight'], l_conf['l2_weight'],
                                                          l_conf['perc_weight'], l_conf['adversarial_params'])
        else:
            self.criterion = None

        # initialization
        if init_cb:
            self.quantizer.init_codebook()

    def forward(self, x: torch.Tensor):
        """
        :param x: (B, C, H, W)
        :return reconstructions: (B, C, H, W), quantizer_loss: float, used_indices: (B, S)
        """

        z = self.encoder(x)
        quantized, used_indices, e_loss = self.quantizer(z)
        x_recon = self.decoder(quantized)

        return x_recon, e_loss, used_indices

    def on_train_start(self):
        """
        Initialize warmup or decay of the learning rate (if specified).
        Initialize const warmup and decay if using Gumbel Softmax.
        """
        # init warmup/decay lr
        lr = float(self.t_conf['lr'])
        if self.t_conf['warmup_epoch'] is not None and self.t_conf['decay_epoch'] is not None:

            warmup_step_start = 0
            warmup_step_end = self.t_conf['warmup_epoch'] * self.trainer.num_training_batches
            decay_step_end = self.t_conf['decay_epoch'] * self.trainer.num_training_batches
            self.warmup_then_decay_lr = LinearCosineScheduler(warmup_step_start, decay_step_end,
                                                              lr, lr / 10, warmup_step_end)

        elif self.t_conf['warmup_epoch'] is not None:

            warmup_step_start = 0
            warmup_step_end = self.t_conf['warmup_epoch'] * self.trainer.num_training_batches
            self.warmup_lr = LinearScheduler(warmup_step_start, warmup_step_end, 1e-20, lr)

        elif self.t_conf['decay_epoch'] is not None:

            decay_step_start = 0
            decay_step_end = self.t_conf['decay_epoch'] * self.trainer.num_training_batches
            self.decay_lr = CosineScheduler(decay_step_start, decay_step_end, lr, lr / 10)

        # if quantizer is gumbel
        if isinstance(self.quantizer, GumbelVectorQuantizer):
            temp, kl = self.quantizer.get_consts()
            if self.kl_warmup_epochs is not None:
                kl_start = 0
                kl_stop = self.kl_warmup_epochs * self.trainer.num_training_batches
                self.quantizer.kl_warmup = LinearScheduler(kl_start, kl_stop, 0.0, kl)

            if self.temp_decay_epochs is not None and self.temp_final is not None:
                temp_start = 0
                temp_stop = self.temp_decay_epochs * self.trainer.num_training_batches
                self.quantizer.temp_decay = CosineScheduler(temp_start, temp_stop, temp, self.temp_final)

    def on_train_batch_start(self, _: Any, batch_index: int):
        """
        Update lr and gumbel quant values according to current epoch/batch index
        """
        current_step = (self.current_epoch * self.trainer.num_training_batches) + batch_index

        # lr update
        if self.warmup_then_decay_lr is not None:
            step_lr = self.warmup_then_decay_lr.step(current_step)
        elif self.warmup_lr is not None:
            step_lr = self.warmup_lr.step(current_step)
        elif self.decay_lr is not None:
            step_lr = self.decay_lr.step(current_step)
        else:
            step_lr = self.t_conf['lr']

        for optimizer in self.trainer.optimizers:
            for g in optimizer.param_groups:
                g['lr'] = step_lr

        # gumbel update and logging
        if isinstance(self.quantizer, GumbelVectorQuantizer):
            this_temp, this_kl = self.quantizer.get_consts()
            if self.quantizer.kl_warmup is not None:
                this_kl = self.quantizer.kl_warmup.step(current_step)
            if self.quantizer.temp_decay is not None:
                this_temp = self.quantizer.temp_decay.step(current_step)
            self.quantizer.set_consts(this_temp, this_kl)
        else:
            this_temp, this_kl = 0.0, 0.0

        self.log('gumbel_quantizer/temperature', this_temp, sync_dist=True)
        self.log('gumbel_quantizer/kl_constant', this_kl, sync_dist=True)

    def training_step(self, batch: Any, batch_index: int):
        """
        :param batch: images B C H W, or tuple if ffcv loader
        :param batch_index: used for logging reconstructions only once per epoch
        """
        images = self.preprocess_batch(batch[0] if isinstance(batch, tuple) else batch)
        x_recon, q_loss, used_indices = self.forward(images)

        # log reconstructions
        self.log_reconstructions(batch_index, images, x_recon, t_or_v='t')

        if isinstance(self.criterion, VQLPIPSWithDiscriminator):
            ae_opt, disc_opt = self.optimizers()

            # Autoencoder Optimization
            res = self.criterion.forward_autoencoder(q_loss, images, x_recon, self.current_epoch,
                                                     last_layer=self.decoder.conv_out.weight)
            loss, l1_loss, l2_loss, p_loss, g_loss, adaptive_weight = res

            ae_opt.zero_grad()
            self.manual_backward(loss)
            ae_opt.step()

            # Discriminator Optimization
            step = (self.current_epoch * self.trainer.num_training_batches) + batch_index
            disc_loss, r1_penalty = self.criterion.forward_discriminator(images, x_recon, self.current_epoch, step)

            disc_opt.zero_grad()
            self.manual_backward(disc_loss)
            disc_opt.step()

        else:
            l2_loss = self.criterion(x_recon, images)
            l1_loss, g_loss, p_loss, disc_loss = torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)
            adaptive_weight, r1_penalty = 0., 0.
            loss = q_loss + l2_loss

        self.log('adaptive_weight', adaptive_weight, sync_dist=True, on_step=False, on_epoch=True)
        self.log('r1_penalty', r1_penalty, sync_dist=True, on_step=False, on_epoch=True)

        self.log('train/loss', loss.detach().cpu().item(), sync_dist=True, on_step=False, on_epoch=True)
        self.log('train/l1_loss', l1_loss.detach().cpu().item(), sync_dist=True, on_step=False, on_epoch=True)
        self.log('train/l2_loss', l2_loss.detach().cpu().item(), sync_dist=True, on_step=False, on_epoch=True)
        self.log('train/quant_loss', q_loss.detach().cpu().item(), sync_dist=True, on_step=False, on_epoch=True)
        self.log('train/perc_loss', p_loss.detach().cpu().item(), sync_dist=True, on_step=False, on_epoch=True)
        self.log('train/gen_loss', g_loss.detach().cpu().item(), sync_dist=True, on_step=False, on_epoch=True)
        self.log('train/disc_loss', disc_loss.detach().cpu().item(), sync_dist=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Any, batch_index: int):
        """
        :param batch: images B C H W, or tuple if ffcv loader
        :param batch_index: used for logging reconstructions only once per epoch
        """

        images = self.preprocess_batch(batch[0] if isinstance(batch, tuple) else batch)
        x_recon, q_loss, used_indices = self.forward(images)

        # log reconstructions
        self.log_reconstructions(batch_index, images, x_recon, t_or_v='v')

        # batch index count (use non-deterministic for this operation)
        torch.use_deterministic_algorithms(False)
        used_indices = torch.bincount(used_indices.view(-1), minlength=self.cb_size)
        torch.use_deterministic_algorithms(True)

        if isinstance(self.criterion, VQLPIPSWithDiscriminator):

            # Autoencoder part
            res = self.criterion.forward_autoencoder(q_loss, images, x_recon, self.current_epoch,
                                                     last_layer=self.decoder.conv_out.weight)
            loss, l1_loss, l2_loss, p_loss, g_loss, _ = res

            # Discriminator part
            step = (self.current_epoch * self.trainer.num_training_batches) + batch_index
            disc_loss, _ = self.criterion.forward_discriminator(images, x_recon, self.current_epoch, step)

        else:
            l2_loss = self.criterion(x_recon, images)
            l1_loss, g_loss, p_loss, disc_loss = torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)
            loss = q_loss + l2_loss

        self.log('validation/loss', loss.cpu().item(), sync_dist=True, on_step=False, on_epoch=True)
        self.log('validation/l1_loss', l1_loss.cpu().item(), sync_dist=True, on_step=False, on_epoch=True)
        self.log('validation/l2_loss', l2_loss.cpu().item(), sync_dist=True, on_step=False, on_epoch=True)
        self.log('validation/quant_loss', q_loss.cpu().item(), sync_dist=True, on_step=False, on_epoch=True)
        self.log('validation/perc_loss', p_loss.cpu().item(), sync_dist=True, on_step=False, on_epoch=True)
        self.log('validation/gen_loss', g_loss.cpu().item(), sync_dist=True, on_step=False, on_epoch=True)
        self.log('validation/disc_loss', disc_loss.cpu().item(), sync_dist=True, on_step=False, on_epoch=True)

        self.val_epoch_usage_count = used_indices if self.val_epoch_usage_count is None else + used_indices
        return loss

    def on_validation_epoch_end(self):
        """
        Compute and log metrics on codebook usage
        """
        perplexity, cb_usage = self.quantizer.get_codebook_usage(self.val_epoch_usage_count)

        # log results
        self.log(f"val_metrics/used_codebook", cb_usage, sync_dist=True)
        self.log(f"val_metrics/perplexity", perplexity, sync_dist=True)

        self.val_epoch_usage_count = None

        return

    def configure_optimizers(self):
        def split_decay_groups(named_modules: list, named_parameters: list,
                               whitelist_weight_modules: tuple[torch.nn.Module, ...],
                               blacklist_weight_modules: tuple[torch.nn.Module, ...],
                               wd: float):
            """
            reference https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/vqvae.py
            separate out all parameters to those that will and won't experience regularizing weight decay
            """

            decay = set()
            no_decay = set()
            for mn, m in named_modules:
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in named_parameters}
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
            assert len(
                param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params),)

            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": wd},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups

        # parameters
        lr = float(self.t_conf['lr'])
        betas = [float(b) for b in self.t_conf['betas']]
        eps = float(self.t_conf['eps'])
        weight_decay = float(self.t_conf['weight_decay'])

        # autoencoder optimizer
        ae_params = split_decay_groups(
            named_modules=list(self.encoder.named_modules()) + list(self.decoder.named_modules()) + list(
                self.quantizer.named_modules()),
            named_parameters=list(self.encoder.named_parameters()) + list(self.decoder.named_parameters()) + list(
                self.quantizer.named_parameters()),
            whitelist_weight_modules=(torch.nn.Conv2d,),
            blacklist_weight_modules=(torch.nn.GroupNorm, torch.nn.Embedding),
            wd=weight_decay
        )
        ae_optimizer = torch.optim.AdamW(ae_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        # discriminator optimizer
        if isinstance(self.criterion, VQLPIPSWithDiscriminator):
            # all have weight decay (conv2d and Linear)
            disc_optimizer = torch.optim.AdamW(self.criterion.discriminator.parameters(), lr=lr, betas=betas, eps=eps,
                                               weight_decay=weight_decay)

            # turn off automatic optimization!
            self.automatic_optimization = False
            return [ae_optimizer, disc_optimizer], []

        return ae_optimizer

    @torch.no_grad()
    def log_reconstructions(self, batch_index, ground_truths, reconstructions, t_or_v='t'):
        """
        log reconstructions every 5 epochs
        """
        if batch_index == 2 and self.current_epoch % 5 == 0:
            b = min(ground_truths.shape[0], 8)
            panel_name = 'train' if t_or_v == 't' else 'validation'

            display, _ = pack([self.preprocess_visualization(ground_truths[:b]),
                               self.preprocess_visualization(reconstructions[:b])], '* c h w')

            display = make_grid(display, nrow=b)
            display = wandb.Image(display)
            self.logger.experiment.log({f'{panel_name}/reconstructions': display})

    def get_tokens(self, images: torch.Tensor) -> torch.IntTensor:
        """
        :param images: B, 3, H, W
        :return B, S batch of codebook indices
        """

        return self.quantizer.vec_to_codes(self.encoder(images))

    def quantize(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: B, 3, H, W
        :return B, S, D batch of quantized
        """
        return rearrange(self.quantizer(self.encoder(images))[0], 'b d h w -> b (h w) d')

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        """
        :param images: B, 3, H, W
        :return reconstructions (B, 3, H, W)
        """
        return self(images)[0]

    def reconstruct_from_tokens(self, tokens: torch.IntTensor) -> torch.Tensor:
        """
        :param tokens: B, S where S is the sequence len
        :return (B, 3, H, W) reconstructed images
        """
        return self.decoder(self.quantizer.codes_to_vec(tokens))