from torch import nn
import torch.nn.functional as F

from modules.residuals import VqVaeResidualStack


class VqVaeEncoder(nn.Module):

    """
    From VQ-VAE paper: https://arxiv.org/abs/1711.00937
    The original encoder consists of 2 strided convolutional layers with stride 2 and window size 4 x 4,
    followed by two residual 3 x 3 blocks (implemented as ReLU, 3x3 conv, ReLU, 1x1 conv), all having 256 hidden units.

    DALL-E paper: first encoder layer has 7x7 convs and last has 1x1 convs
    """

    def __init__(self, num_downsamples=2, out_channels=128, num_residual_layers=2):
        """
        :param num_downsamples: the number of Conv layers to apply, causing a down-sample of the original image equal
                                to 2 ** num_downsamples
                                Ex: from 32 to 8 if there are 2 downsamples or from 32 to 4 if there are 2 downsamples.
        :param out_channels: default to 128
        :param num_residual_layers: default to 2 (like the original paper).
        """

        super().__init__()

        assert num_downsamples >= 2, 'the down sample must be at least two !'

        self.hidden_convs = num_downsamples - 2  # real num of hidden convs

        # first conv
        self.conv_first = nn.Conv2d(in_channels=3,
                                    out_channels=out_channels // 2,
                                    kernel_size=(4, 4),
                                    stride=(2, 2),
                                    padding=1
                                    )

        # internal convs
        if self.hidden_convs > 0:

            self.hidden_convs_list = nn.ModuleList(
                [nn.Conv2d(in_channels=out_channels // 2, out_channels=out_channels // 2,
                           kernel_size=(4, 4), stride=(2, 2), padding=1)
                 for _ in range(self.hidden_convs)]
            )

        else:
            self.hidden_conv_list = None
        
        # last conv
        self.conv_last = nn.Conv2d(in_channels=out_channels // 2,
                                   out_channels=out_channels,
                                   kernel_size=(4, 4),
                                   stride=(2, 2),
                                   padding=1
                                   )
        
        self.residual_stack = VqVaeResidualStack(in_channels=out_channels,
                                                 num_hiddens=out_channels,
                                                 num_residual_layers=num_residual_layers
                                                 )

    def forward(self, inputs):

        x = self.conv_first(inputs)
        x = F.relu(x)

        for i in range(self.hidden_convs):
            x = self.hidden_convs_list[i](x)
            x = F.relu(x)

        x = self.conv_last(x)

        return self.residual_stack(x)
