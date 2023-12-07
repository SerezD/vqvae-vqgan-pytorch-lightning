import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as functional

from vqvae.modules.loss.stylegan2_discriminator.discriminator import Discriminator
from vqvae.modules.loss.lpips_pytorch import LPIPS
from vqvae.modules.loss.stylegan2_discriminator.utils.ops import conv2d_gradfix


def generator_loss(logits: torch.Tensor, loss_type: str = "hinge"):
    """
    :param logits: discriminator output in the generator phase (fake_logits)
    :param loss_type: which loss to apply between 'hinge' and 'non-saturating'
    """
    if loss_type == 'hinge':
        loss = -torch.mean(logits)
    elif loss_type == 'non-saturating':
        # Torch docs for bce with logits:
        # This loss combines a Sigmoid layer and the BCELoss in one single class.
        # This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as,
        # by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability
        loss = functional.binary_cross_entropy_with_logits(logits, target=torch.ones_like(logits))
    else:
        raise ValueError(f'unknown loss_type: {loss_type}')
    return loss


def discriminator_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor, loss_type: str = 'hinge'):
    """
    :param logits_real: discriminator output when input is the original image
    :param logits_fake: discriminator output when input is the reconstructed image
    :param loss_type: which loss to apply between 'hinge' and 'non-saturating'
    """

    if loss_type == 'hinge':
        real_loss = functional.relu(1.0 - logits_real)
        fake_loss = functional.relu(1.0 + logits_fake)
    elif loss_type == 'non-saturating':
        # Torch docs for bce with logits:
        # This loss combines a Sigmoid layer and the BCELoss in one single class.
        # This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as,
        # by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability
        real_loss = functional.binary_cross_entropy_with_logits(logits_real,
                                                                target=torch.ones_like(logits_real), reduction='none')
        fake_loss = functional.binary_cross_entropy_with_logits(logits_fake,
                                                                target=torch.zeros_like(logits_fake), reduction='none')
    else:
        raise ValueError(f'unknown loss_type: {loss_type}')

    return torch.mean(real_loss + fake_loss)


class VQLPIPSWithDiscriminator(nn.Module):

    def __init__(self, image_size: int, l1_weight: float, l2_weight: float, perc_weight: float, adversarial_conf: dict):

        super().__init__()

        self.l1_loss = lambda rec, tar: (tar - rec).abs().mean()
        self.l1_weight = l1_weight

        self.l2_loss = lambda rec, tar: (tar - rec).pow(2).mean()
        self.l2_weight = l2_weight

        self.perceptual_loss = LPIPS(net_type='vgg')
        self.perceptual_weight = perc_weight

        self.discriminator = Discriminator(image_size)

        self.adversarial_start_epoch = adversarial_conf['start_epoch']
        self.adversarial_loss_type = adversarial_conf['loss_type']

        self.generator_weight = adversarial_conf['g_weight']
        self.use_adaptive_g_weight  =adversarial_conf['use_adaptive']

        self.r1_regularization_cost = adversarial_conf['r1_reg_weight']
        self.r1_regularization_every = adversarial_conf['r1_reg_every']

    def calculate_adaptive_weight(self, nll_loss: float, g_loss: float, last_layer: torch.nn.Parameter):
        """
        From Taming Transformers for High-Resolution Image Synthesis paper, Patrick Esser, Robin Rombach, Bjorn Ommer:

        "we compute the adaptive weight λ according to λ = ∇GL[Lrec] / (∇GL[LGAN] + δ)
         where Lrec is the perceptual reconstruction loss, ∇GL[·] denotes the gradient of its input w.r.t. the last
         layer L of the decoder, and δ = 10−6 is used for numerical stability"

        """
        nll_grads = grad(nll_loss, last_layer, grad_outputs=torch.ones_like(nll_loss), retain_graph=True)[0].detach()
        g_grads = grad(g_loss, last_layer, grad_outputs=torch.ones_like(g_loss), retain_graph=True)[0].detach()

        adaptive_weight = torch.norm(nll_grads, p=2) / (torch.norm(g_grads, p=2) + 1e-8)
        adaptive_weight = torch.clamp(adaptive_weight, 0.0, 1e4).detach()
        adaptive_weight = adaptive_weight * self.generator_weight

        return adaptive_weight

    def calculate_r1_regularization_term(self, logits_real: torch.Tensor, images: torch.Tensor, compute_r1: bool):
        """
        r1 term calculation: https://github.com/rosinality/stylegan2-pytorch/blob/master/train.py
        """
        if compute_r1:

            # gradient
            with conv2d_gradfix.no_weight_gradients():
                gradients = torch.autograd.grad(outputs=logits_real.sum(), inputs=images, create_graph=True)[0]

            r1_term = self.r1_regularization_cost * gradients.pow(2).view(gradients.shape[0], -1).sum(1).mean()
        else:
            r1_term = 0.

        return r1_term

    def forward_autoencoder(self, quantizer_loss: float, images: torch.Tensor, reconstructions: torch.Tensor,
                            current_epoch: int, last_layer: torch.nn.Parameter):

        # reconstruction losses
        l1_loss = self.l1_loss(reconstructions.contiguous(), images.contiguous())
        l2_loss = self.l2_loss(reconstructions.contiguous(), images.contiguous())
        p_loss = self.perceptual_loss(images.contiguous(), reconstructions.contiguous())

        nll_loss = l1_loss * self.l1_weight + l2_loss * self.l2_weight + p_loss * self.perceptual_weight

        # adversarial loss
        logits_fake = self.discriminator(reconstructions.contiguous())
        g_loss = generator_loss(logits_fake, loss_type=self.adversarial_loss_type)

        if (self.training and current_epoch >= self.adversarial_start_epoch):
            if self.use_adaptive_g_weight:
                g_weight = self.calculate_adaptive_weight(p_loss, g_loss, last_layer=last_layer)
            else:
                g_weight = self.generator_weight
        else:
            g_weight = 0.  # case: disc not started yet or testing

        loss = nll_loss + g_loss * g_weight + quantizer_loss

        return loss, l1_loss, l2_loss, p_loss, g_loss, g_weight

    def forward_discriminator(self, images: torch.Tensor, reconstructions: torch.Tensor,
                              current_epoch: int, current_step: int):

        # discriminator update
        d_weight = 1. if current_epoch >= self.adversarial_start_epoch else 0.  # if disc as started
        compute_r1 = (self.training and current_step % self.r1_regularization_every == 0 and d_weight == 1. and
                      self.r1_regularization_cost is not None)

        images = images.contiguous().requires_grad_(compute_r1)
        logits_real = self.discriminator(images)
        logits_fake = self.discriminator(reconstructions.contiguous().detach())
        d_loss = discriminator_loss(logits_real, logits_fake, loss_type=self.adversarial_loss_type)

        r1_term = self.calculate_r1_regularization_term(logits_real, images, compute_r1)

        loss = d_weight * (d_loss + r1_term)
        return loss, d_loss, r1_term


class VQLPIPS(nn.Module):

    def __init__(self, l1_weight: float, l2_weight: float, perc_weight: float):
        """
        VQGAN Loss without discriminator. Used just for ablation.
        """

        super().__init__()

        self.l1_loss = lambda rec, tar: (tar - rec).abs().mean()
        self.l1_weight = l1_weight

        self.l2_loss = lambda rec, tar: (tar - rec).pow(2).mean()
        self.l2_weight = l2_weight

        self.perceptual_loss = LPIPS(net_type='alex')
        self.perceptual_weight = perc_weight

    def forward(self, quantizer_loss: float, images: torch.Tensor, reconstructions: torch.Tensor):
        """
        :returns quant + nll loss, l1 loss, l2 loss, perceptual loss
        """

        # reconstruction losses
        l1_loss = self.l1_loss(reconstructions.contiguous(), images.contiguous())
        l2_loss = self.l2_loss(reconstructions.contiguous(), images.contiguous())
        p_loss = self.perceptual_loss(images.contiguous(), reconstructions.contiguous())

        nll_loss = l1_loss * self.l1_weight + l2_loss * self.l2_weight + p_loss * self.perceptual_weight

        loss = quantizer_loss + nll_loss

        return loss, l1_loss, l2_loss, p_loss
