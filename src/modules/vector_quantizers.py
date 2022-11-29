"""
Scheme of basic Quantization
                                     ┌────────────┐
                                     │            │
                                     │  CODEBOOK  │
   Input                   QUANTIZE─►│   K,D=C    ├──────┐                      Output
     │                         │     │            │      │                        ▲
     ▼                         │     └────────────┘      │                        │
┌─────────┐                    │                         ▼                    ┌───┴─────┐
│ ENCODER ├─► (B,C=D,H,W)─►(B*H*W,C=D)              (B*H*W,C=D)─►(B,C=D,H,W)─►│ DECODER │
└─────────┘                                                                   └─────────┘

EMA ALGORITHM
Each codebook entry is updated according to the encoder outputs who selected it.
The important thing is that the codebook updating is not a loss term anymore.
Specifically, for every codebook item wi, the mean code mi and usage count Ni are tracked:
Ni ← Ni · γ + ni(1 − γ),
mi ← mi · γ + Xnij e(xj )(1 − γ),
wi ← mi Ni
where γ is a discount factor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class EMAVectorQuantizer(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.95, epsilon=1e-5):

        """
        :param num_embeddings: size of the latent dictionary (num of embedding vectors).
        :param embedding_dim: size of a single tensor in dict.
        :param commitment_cost: scaling factor for e_loss
        :param decay: decay for EMA updating
        :param epsilon: smoothing parameters for EMA weights
        """

        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # create the codebook of the desired size
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # uniformly init codebook
        self.codebook.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.codebook.requires_grad_(False)

        # ema parameters
        # ema usage count: total count of each embedding trough epochs
        self.register_buffer('ema_count', torch.zeros(num_embeddings))

        # same size as dict, initialized with normal
        # the updated means
        self.register_buffer('ema_weight', torch.empty((self.num_embeddings, self.embedding_dim)))
        self.ema_weight.data.normal_()

        self.decay = decay
        self.epsilon = epsilon

    def forward(self, x):
        """
        :param x: a series of tensors (output of the Encoder - B,C,H,W).
        """
        b, c, h, w = x.shape

        inputs = rearrange(x, 'b c h w -> b h w c').contiguous()
        input_shape = inputs.shape

        # Flat input to vectors of embedding dim = C.
        flat_input = rearrange(inputs, 'b h w c -> (b h w) c')

        # Calculate distances of each vector w.r.t the dict
        # distances is a matrix (B*H*W, codebook_size)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebook.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebook.weight.t()))

        # Get indices of the closest vector in dict, and create a mask on the correct indexes
        # encoding_indices = (num_vectors_in_batch, 1)
        # Mask = (num_vectors_in_batch, codebook_dim)
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # Quantize and un-flat
        quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        # Update a codebook vector as the mean of the encoder outputs that are closer to it
        # Calculate the usage count of codes and the mean code, then update the codebook vector dividing the two
        if self.training:
            with torch.no_grad():
                ema_count = self.get_buffer('ema_count') * self.decay + (1 - self.decay) * torch.sum(encodings, 0)

                # Laplace smoothing of the ema count
                n = flat_input.shape[0]  # tensors in batch
                self.ema_count = (ema_count + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n

                dw = torch.matmul(encodings.t(), flat_input)
                self.ema_weight = self.get_buffer('ema_weight') * self.decay + (1 - self.decay) * dw

                self.codebook.weight.data = self.get_buffer('ema_weight') / self.get_buffer('ema_count').unsqueeze(1)

        # Loss function (only the inputs are updated)
        e_loss = self.commitment_cost * F.mse_loss(quantized.detach(), inputs)

        # during backpropagation quantized = inputs (copy gradient trick)
        quantized = inputs + (quantized - inputs).detach()

        encoding_indices = rearrange(encoding_indices, '(b h w)-> b (h w)', b=b, h=h, w=w)

        return e_loss, rearrange(quantized, 'b h w c -> b c h w').contiguous(), encoding_indices

    @torch.no_grad()
    def get_codebook(self):
        return self.codebook

    @torch.no_grad()
    def quantize(self, inputs):
        """
        :param inputs: batch to quantize
        :return: the quantized version of x
        """
        return self.forward(inputs)[1]
