import torch
import torch.nn.functional as F

from einops import rearrange, einsum
from vqvae.modules.abstract_modules.base_quantizer import BaseVectorQuantizer


class VectorQuantizer(BaseVectorQuantizer):

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):

        """
        Original VectorQuantizer with straight through gradient estimator (loss is optimized on inputs and codebook)
        :param num_embeddings: size of the latent dictionary (num of embedding vectors).
        :param embedding_dim: size of a single tensor in dict.
        :param commitment_cost: scaling factor for e_loss
        """

        super().__init__(num_embeddings, embedding_dim)

        self.commitment_cost = commitment_cost

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.IntTensor, float):
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return quantized_x (B, D, H, W), detached codes (B, H*W), latent_loss
        """

        b, c, h, w = x.shape
        device = x.device

        # Flat input to vectors of embedding dim = C.
        flat_x = rearrange(x, 'b c h w -> (b h w) c')

        # Calculate distances of each vector w.r.t the dict
        # distances is a matrix (B*H*W, codebook_size)
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebook.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.codebook.weight.t()))

        # Get indices of the closest vector in dict, and create a mask on the correct indexes
        # encoding_indices = (num_vectors_in_batch, 1)
        # Mask = (num_vectors_in_batch, codebook_dim)
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # Quantize and un-flat
        quantized = torch.matmul(encodings, self.codebook.weight)

        # Loss functions
        e_loss = self.commitment_cost * F.mse_loss(quantized.detach(), flat_x)
        q_loss = F.mse_loss(quantized, flat_x.detach())

        # during backpropagation quantized = inputs (copy gradient trick)
        quantized = flat_x + (quantized - flat_x).detach()

        quantized = rearrange(quantized, '(b h w) c -> b c h w', b=b, h=h, w=w)
        encoding_indices = rearrange(encoding_indices, '(b h w)-> b (h w)', b=b, h=h, w=w).detach()

        return quantized, encoding_indices, q_loss + e_loss

    @torch.no_grad()
    def vec_to_codes(self, x: torch.Tensor) -> torch.IntTensor:
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return flat codebook indices (B, H * W)
        """
        b, c, h, w = x.shape

        # Flat input to vectors of embedding dim = C.
        flat_x = rearrange(x, 'b c h w -> (b h w) c')

        # Calculate distances of each vector w.r.t the dict
        # distances is a matrix (B*H*W, codebook_size)
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebook.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.codebook.weight.t()))

        # Get indices of the closest vector in dict
        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = rearrange(encoding_indices, '(b h w) -> b (h w)', b=b, h=h, w=w)

        return encoding_indices


class EMAVectorQuantizer(BaseVectorQuantizer):

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25, decay: float = 0.95,
                 epsilon: float = 1e-5):

        """
        EMA ALGORITHM
        Each codebook entry is updated according to the encoder outputs who selected it.
        The important thing is that the codebook updating is not a loss term anymore.
        Specifically, for every codebook item wi, the mean code mi and usage count Ni are tracked:
        Ni ← Ni · γ + ni(1 − γ),
        mi ← mi · γ + Xnij e(xj )(1 − γ),
        wi ← mi Ni
        where γ is a discount factor

        :param num_embeddings: size of the latent dictionary (num of embedding vectors).
        :param embedding_dim: size of a single tensor in dictionary
        :param commitment_cost: scaling factor for e_loss
        :param decay: decay for EMA updating
        :param epsilon: smoothing parameters for EMA weights
        """

        super().__init__(num_embeddings, embedding_dim)

        self.commitment_cost = commitment_cost

        # EMA does not require grad
        self.codebook.requires_grad_(False)

        # ema parameters
        # ema usage count: total count of each embedding trough epochs
        self.register_buffer('ema_count', torch.zeros(num_embeddings))

        # same size as dict, initialized as codebook
        # the updated means
        self.register_buffer('ema_weight', torch.empty((self.num_embeddings, self.embedding_dim)))
        self.ema_weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

        self.decay = decay
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.IntTensor, float):
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return quantized_x (B, D, H, W), detached codes (B, H*W), latent_loss
        """

        b, c, h, w = x.shape
        device = x.device

        # Flat input to vectors of embedding dim = C.
        flat_x = rearrange(x, 'b c h w -> (b h w) c')

        # Calculate distances of each vector w.r.t the dict
        # distances is a matrix (B*H*W, codebook_size)
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebook.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.codebook.weight.t()))

        # Get indices of the closest vector in dict, and create a mask on the correct indexes
        # encoding_indices = (num_vectors_in_batch, 1)
        # Mask = (num_vectors_in_batch, codebook_dim)
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # Quantize and un-flat
        quantized = torch.matmul(encodings, self.codebook.weight)

        # Use EMA to update the embedding vectors
        # Update a codebook vector as the mean of the encoder outputs that are closer to it
        # Calculate the usage count of codes and the mean code, then update the codebook vector dividing the two
        if self.training:
            with torch.no_grad():
                ema_count = self.get_buffer('ema_count') * self.decay + (1 - self.decay) * torch.sum(encodings, 0)

                # Laplace smoothing of the ema count
                self.ema_count = (ema_count + self.epsilon) / (b + self.num_embeddings * self.epsilon) * b

                dw = torch.matmul(encodings.t(), flat_x)
                self.ema_weight = self.get_buffer('ema_weight') * self.decay + (1 - self.decay) * dw

                self.codebook.weight.data = self.get_buffer('ema_weight') / self.get_buffer('ema_count').unsqueeze(1)

        # Loss function (only the inputs are updated)
        e_loss = self.commitment_cost * F.mse_loss(quantized.detach(), flat_x)

        # during backpropagation quantized = inputs (copy gradient trick)
        quantized = flat_x + (quantized - flat_x).detach()

        quantized = rearrange(quantized, '(b h w) c -> b c h w', b=b, h=h, w=w)
        encoding_indices = rearrange(encoding_indices, '(b h w)-> b (h w)', b=b, h=h, w=w).detach()

        return quantized, encoding_indices, e_loss

    @torch.no_grad()
    def vec_to_codes(self, x: torch.Tensor) -> torch.IntTensor:
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return flat codebook indices (B, H * W)
        """
        b, c, h, w = x.shape

        # Flat input to vectors of embedding dim = C.
        flat_x = rearrange(x, 'b c h w -> (b h w) c')

        # Calculate distances of each vector w.r.t the dict
        # distances is a matrix (B*H*W, codebook_size)
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebook.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.codebook.weight.t()))

        # Get indices of the closest vector in dict
        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = rearrange(encoding_indices, '(b h w) -> b (h w)', b=b, h=h, w=w)

        return encoding_indices


class GumbelVectorQuantizer(BaseVectorQuantizer):
    def __init__(self, num_embeddings: int, embedding_dim: int, straight_through: bool = False, temp: float = 1.0,
                 kl_cost: float = 5e-4):
        """
        :param num_embeddings: size of the latent dictionary (num of embedding vectors).
        :param embedding_dim: size of a single tensor in dict.
        :param straight_through: if True, will one-hot quantize, but still differentiate as if it is the soft sample
        :param temp: temperature parameter for gumbel softmax
        :param kl_cost: cost for kl divergence
        """
        super().__init__(num_embeddings, embedding_dim)

        self.x_to_logits = torch.nn.Conv2d(num_embeddings, num_embeddings, 1)
        self.straight_through = straight_through
        self.temp = temp
        self.kl_cost = kl_cost

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.IntTensor, float):
        """
        :param x: tensors (output of the Encoder - B,N,H,W). Note that N = number of embeddings in dict!
        :return quantized_x (B, D, H, W), detached codes (B, H*W), latent_loss
        """

        # deterministic quantization during inference
        hard = self.straight_through if self.training else True

        logits = self.x_to_logits(x)
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temp, dim=1, hard=hard)
        quantized = einsum(soft_one_hot, self.get_codebook(), 'b n h w, n d -> b d h w')

        # + kl divergence to the prior (uniform) loss, increase cb usage
        # Note:
        #       KL(P(x), Q(x)) = sum_x (P(x) * log(P(x) / Q(x)))
        #       in this case: P(x) is qy, Q(x) is uniform distribution (1 / num_embeddings)
        qy = F.softmax(logits, dim=1)
        kl_loss = self.kl_cost * torch.sum(qy * torch.log(qy * self.num_embeddings + 1e-10), dim=1).mean()

        encoding_indices = soft_one_hot.argmax(dim=1).detach()

        return quantized, encoding_indices, kl_loss

    def get_consts(self) -> (float, float):
        """
        return temp, kl_cost
        """
        return self.temp, self.kl_cost

    def set_consts(self, temp: float = None, kl_cost: float = None) -> None:
        """
        update values for temp, kl_cost
        :param temp: new value for temperature (if not None)
        :param kl_cost: new value for kl_cost (if not None)
        """
        if temp is not None:
            self.temp = temp

        if kl_cost is not None:
            self.kl_cost = kl_cost

    @torch.no_grad()
    def vec_to_codes(self, x: torch.Tensor) -> torch.IntTensor:
        """
        :param x: tensors (output of the Encoder - B,N,H,W). Note that N = number of embeddings in dict!
        :return flat codebook indices (B, H * W)
        """

        soft_one_hot = F.gumbel_softmax(x, tau=1.0, dim=1, hard=True)
        encoding_indices = soft_one_hot.argmax(dim=1)
        return encoding_indices


class EntropyVectorQuantizer(BaseVectorQuantizer):

    def __init__(self, num_embeddings: int, embedding_dim: int, ent_loss_ratio: float = 0.1,
                 ent_temperature: float = 0.01, ent_loss_type: str = 'softmax', commitment_cost: float = 0.25):

        super().__init__(num_embeddings, embedding_dim)

        # hparams
        self.ent_loss_ratio = ent_loss_ratio
        self.ent_temperature = ent_temperature
        self.ent_loss_type = ent_loss_type
        self.commitment_cost = commitment_cost

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.IntTensor, float):
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return quantized_x (B, D, H, W), detached codes (B, H*W), latent_loss
        """

        def entropy_loss(affinity: torch.Tensor, temperature: float, loss_type: str = 'softmax'):
            """
            Increase codebook usage by maximizing entropy

            affinity: 2D tensor of size Dim, n_classes
            """

            n_classes = affinity.shape[-1]

            affinity = torch.div(affinity, temperature)
            probs = F.softmax(affinity, dim=-1)
            log_probs = F.log_softmax(affinity + 1e-5, dim=-1)

            if loss_type == "softmax":
                target_probs = probs
            elif loss_type == "argmax":
                codes = torch.argmax(affinity, dim=-1)
                one_hots = F.one_hot(codes, n_classes).to(codes)
                one_hots = probs - (probs - one_hots).detach()
                target_probs = one_hots
            else:
                raise ValueError("Entropy loss {} not supported".format(loss_type))

            avg_probs = torch.mean(target_probs, dim=0)
            avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
            sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
            return sample_entropy - avg_entropy

        batch_size, c, h, w = x.shape

        # compute distances
        flat_x = rearrange(x, 'b c h w -> (b h w) c')
        transposed_cb_weights = self.get_codebook().T

        # final distance vector is (B * Latent_Dim, Codebook Dim)
        a2 = torch.sum(flat_x ** 2, dim=1, keepdim=True)
        b2 = torch.sum(transposed_cb_weights ** 2, dim=0, keepdim=True)
        ab = torch.matmul(flat_x, transposed_cb_weights)
        distances = a2 - 2 * ab + b2

        # get indices and quantized
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.codebook(encoding_indices)
        quantized = rearrange(quantized, '(b h w) c -> b c h w', b=batch_size, h=h, w=w)
        encoding_indices = rearrange(encoding_indices, '(b h w)-> b (h w)', b=batch_size, h=h, w=w).detach()

        # compute_loss
        e_latent_loss = torch.mean((quantized.detach() - x) ** 2) * self.commitment_cost
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)
        ent_loss = entropy_loss(-distances, self.ent_temperature, self.ent_loss_type) * self.ent_loss_ratio
        loss = e_latent_loss + q_latent_loss + ent_loss

        quantized = x + (quantized - x).detach()

        return quantized, encoding_indices, loss

    @torch.no_grad()
    def vec_to_codes(self, x: torch.Tensor) -> torch.IntTensor:
        """
        :param x: tensors (output of the Encoder - B,D,H,W).
        :return flat codebook indices (B, H * W)
        """

        batch_size, c, h, w = x.shape

        # compute distances
        flat_x = rearrange(x, 'b c h w -> (b h w) c')
        transposed_cb_weights = self.get_codebook().T

        # final distance vector is (B * Latent_Dim, Codebook Dim)
        a2 = torch.sum(flat_x ** 2, dim=1, keepdim=True)
        b2 = torch.sum(transposed_cb_weights ** 2, dim=0, keepdim=True)
        ab = torch.matmul(flat_x, transposed_cb_weights)
        distances = a2 - 2 * ab + b2

        # get indices and quantized
        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = rearrange(encoding_indices, '(b h w)-> b (h w)', b=batch_size, h=h, w=w)

        return encoding_indices
