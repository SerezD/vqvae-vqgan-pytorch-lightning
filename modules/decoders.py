from torch import nn
import torch.nn.functional as F

from modules.residuals import VqVaeResidualStack


class VqVaeDecoder(nn.Module):

    """
    From VQ-VAE paper: https://arxiv.org/abs/1711.00937
    The original decoder has two residual 3 x 3 blocks, followed by two transposed convolutions with stride
    2 and window size 4 x 4.
    Reminder: Transposed convolution interleaves zeros in the input 2d vector, and then applies convolution on
    the new input, in order to obtain a higher dimensional output.
    """

    def __init__(self, input_channels, num_upsamples, hidden_channels=128, num_residual_layers=2):
        """
        :param input_channels: must be equal to the number of output channels in the encoder
        :param num_upsamples: this value should be the same that the encoder num of downsamples
        :param hidden_channels: default is 128
        :param num_residual_layers: default is 2
        """
        super().__init__()

        assert num_upsamples >= 2, 'the down sample must be at least two !'

        self.hidden_convs = num_upsamples - 2  # real num of hidden convs

        self.residual_stack = VqVaeResidualStack(in_channels=input_channels,
                                                 num_hiddens=hidden_channels,
                                                 num_residual_layers=num_residual_layers
                                                 )

        # convs
        self.conv_trans_first = nn.ConvTranspose2d(in_channels=hidden_channels,
                                                   out_channels=hidden_channels // 2,
                                                   kernel_size=(4, 4),
                                                   stride=(2, 2),
                                                   padding=(1, 1)
                                                   )

        # internal convs
        if self.hidden_convs > 0:

            self.hidden_convs_list = nn.ModuleList(
                [nn.ConvTranspose2d(in_channels=hidden_channels // 2, out_channels=hidden_channels // 2,
                                    kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                 for _ in range(self.hidden_convs)]
            )

        else:
            self.hidden_conv_list = None

        self.conv_trans_last = nn.ConvTranspose2d(in_channels=hidden_channels // 2,
                                                  out_channels=3,
                                                  kernel_size=(4, 4),
                                                  stride=(2, 2),
                                                  padding=(1, 1)
                                                  )

    def forward(self, inputs):

        x = self.residual_stack(inputs)
        x = self.conv_trans_first(x)
        x = F.relu(x)

        for i in range(self.hidden_convs):
            x = self.hidden_convs_list[i](x)
            x = F.relu(x)

        return self.conv_trans_last(x)
