from torch import nn
import torch.nn.functional as F


class VqVaeResidual(nn.Module):

    """
    From VQ-VAE paper: https://arxiv.org/abs/1711.00937
    residual block (implemented as ReLU, 3x3 conv, ReLU, 1x1 conv), all having 256 hidden units.
    """
    def __init__(self, input_channels=256, hidden_channels=256):
        """
        :param input_channels: number of input channels
        :param hidden_channels: intermediate and final number of hiddens
        """
        super().__init__()

        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=input_channels,
                      out_channels=hidden_channels,
                      kernel_size=(3, 3),
                      stride=(1, 1),
                      padding=1,
                      bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hidden_channels,
                      out_channels=hidden_channels,
                      kernel_size=(1, 1),
                      stride=(1, 1),
                      bias=False)
        )

    def forward(self, x):
        return x + self.res_block(x)


class VqVaeResidualStack(nn.Module):

    """
    create a stack of n residual blocks (2 blocks with 256 hiddens were used in the original paper)
    """
    def __init__(self, in_channels=256, num_hiddens=256, num_residual_layers=2):

        """
        :param in_channels: input channels (256 in the original paper)
        :param num_hiddens: output hiddens (256 in the original paper)
        :param num_residual_layers: number of residual blocks (2 in the original paper)
        """
        super().__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([VqVaeResidual(in_channels, num_hiddens) for _ in range(self.num_residual_layers)])

    def forward(self, x):
        for i in range(self.num_residual_layers):
            x = self.layers[i](x)
        return F.relu(x)
