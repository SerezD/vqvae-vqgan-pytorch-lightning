# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from vqvae.modules.loss.stylegan2_discriminator.utils import misc, persistence
from vqvae.modules.loss.stylegan2_discriminator.utils.ops import conv2d_resample, upfirdn2d, bias_act, fma


# ----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


# ----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
        x,  # Input tensor of shape [batch_size, in_channels, in_height, in_width].
        weight,  # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
        styles,  # Modulation coefficients of shape [batch_size, in_channels].
        noise=None,  # Optional noise tensor to add to the output activations.
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
        padding=0,  # Padding with respect to the upsampled image.
        resample_filter=None,
        # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
        demodulate=True,  # Apply weight demodulation?
        flip_weight=True,  # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
        fused_modconv=True,  # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    misc.assert_shape(weight, [out_channels, in_channels, kh, kw])  # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None])  # [NIHW]
    misc.assert_shape(styles, [batch_size, in_channels])  # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1, 2, 3],
                                                                            keepdim=True))  # max_Ikk
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter, up=up, down=down,
                                            padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings():  # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter, up=up, down=down, padding=padding,
                                        groups=batch_size, flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
                 in_features,  # Number of input features.
                 out_features,  # Number of output features.
                 bias=True,  # Apply additive bias before the activation function?
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 lr_multiplier=1,  # Learning rate multiplier.
                 bias_init=0,  # Initial value for the additive bias.
                 ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t()).to(w.dtype)
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 kernel_size,  # Width and height of the convolution kernel.
                 bias=True,  # Apply additive bias before the activation function?
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 up=1,  # Integer upsampling factor.
                 down=1,  # Integer downsampling factor.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
                 channels_last=False,  # Expect the input to have memory_format=channels_last?
                 trainable=True,  # Update the weights of this layer during training?
                 ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = (self.weight * self.weight_gain).to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1)  # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w, f=self.resample_filter, up=self.up, down=self.down,
                                            padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x.to(w.dtype), b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels, 0 = first block.
                 tmp_channels,  # Number of intermediate channels.
                 out_channels,  # Number of output channels.
                 resolution,  # Resolution of this block.
                 img_channels,  # Number of input color channels.
                 first_layer_idx,  # Index of the first layer.
                 architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
                 activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 use_fp16=False,  # Use FP16 for this block?
                 fp16_channels_last=False,  # Use channels-last memory format with FP16?
                 freeze_layers=0,  # Freeze-D: Number of layers to freeze.
                 ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0

        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable

        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation,
                                       trainable=next(trainable_iter), conv_clamp=conv_clamp,
                                       channels_last=self.channels_last)

        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation,
                                 trainable=next(trainable_iter), conv_clamp=conv_clamp,
                                 channels_last=self.channels_last)

        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
                                 trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp,
                                 channels_last=self.channels_last)

        if architecture == 'resnet':
            self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
                                    trainable=next(trainable_iter), resample_filter=resample_filter,
                                    channels_last=self.channels_last)

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == 'skip' else None

        # Main layers.
        if self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img


# ----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings():  # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H,
                      W)  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 cmap_dim,  # Dimensionality of mapped conditioning label, 0 = no label.
                 resolution,  # Resolution of this block.
                 img_channels,  # Number of input color channels.
                 architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
                 mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, None = entire minibatch.
                 mbstd_num_channels=1,  # Number of features for the minibatch standard deviation layer, 0 = disable.
                 activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture

        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size,
                                       num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation,
                                conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])  # [NCHW]
        _ = force_fp32  # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x).to(dtype=dtype, memory_format=memory_format)

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x


# ----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
                 img_resolution,  # Input resolution.
                 c_dim=0,  # Conditioning label (C) dimensionality.
                 img_channels=3,  # Number of input color channels.
                 architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'.
                 channel_base=32768,  # Overall multiplier for the number of channels.
                 channel_max=512,  # Maximum number of channels in any layer.
                 num_fp16_res=0,  # Use FP16 for the N highest resolutions.
                 conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
                 cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
                 block_kwargs={},  # Arguments for DiscriminatorBlock.
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 epilogue_kwargs={},  # Arguments for DiscriminatorEpilogue.
                 ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                                       first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs,
                                       **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs,
                                        **common_kwargs)

    def forward(self, img, **block_kwargs):
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        x = self.b4(x, img, cmap)
        return x

# ----------------------------------------------------------------------------
