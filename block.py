import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1,
               groups=1):
    """
    Creates a convolution layer.

    Parameters
    ----------
    in_channels : int
        The number of channels to take in
    out_channels : int
        The number of output channels
    kernel_size : int
        The kernel size of the convolution kernel
    stride : int, optional
        The stride to move the kernel by, by default 1
    dilation : int, optional
        The dilation of the kernel, by default 1
    groups : int, optional
        How many groups to use in the convolution process, by default 1

    Returns
    -------
    nn.Conv2d
        The desired convolution kernel
    """
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                     padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    """
    Creates and returns the normalization process to use

    Parameters
    ----------
    norm_type : str
        The type of normalization procedure, must be one of 'batch' or
        'instance'
    nc : int
        The number of channels

    Returns
    -------
    nn.Module
        The normalization procedure

    Raises
    ------
    NotImplementedError
        If norm_type specified incorrectly
    """
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'
                                  .format(norm_type))
    return layer


def pad(pad_type, padding):
    """
    Generates and returns an object for padding

    Parameters
    ----------
    pad_type : str
        The type of desired padding, one of 'reflect', 'replicate'
    padding : int
        The amount of padding to use

    Returns
    -------
    nn.Module
        The padding procedure

    Raises
    ------
    NotImplementedError
        If pad_type specified incorrectly
    """
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'
                                  .format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    """
    Returns the appropriate amount of padding to use

    Parameters
    ----------
    kernel_size : int
        The size of the kernel
    dilation : int
        The dilation of the kernel

    Returns
    -------
    int
        The amount of padding to use to ensure that the output size stays
        consistent
    """
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1,
               bias=True, pad_type='zero', norm_type=None, act_type='relu'):
    """
    Generates a convolution block for the RFDN block

    Parameters
    ----------
    in_channels : int
        The number of channels to take in
    out_channels : int
        The number of output channels
    kernel_size : int
        The kernel size of the convolution kernel
    stride : int, optional
        The stride to move the kernel by, by default 1
    dilation : int, optional
        The dilation of the kernel, by default 1
    groups : int, optional
        How many groups to use in the convolution process, by default 1
    bias : bool, optional
        Whether or not to include the bias parameter, by default True
    pad_type : str, optional
        The type of padding to use, by default 'zero'
    norm_type : str, optional
        The type of norm to use, by default None
    act_type : str, optional
        The type of activation to use, by default 'relu'

    Returns
    -------
    nn.Sequential
        The Sequential object storing the network block
    """
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Generates and returns the activation function to use

    Parameters
    ----------
    act_type : str
        The type of activation function, must be one of 'relu', 'lre;u' or
        'prelu'
    inplace : bool, optional
        Whether to perform the activation in place or not, by default True
    neg_slope : float, optional
        The initial slope for the parameterized ReLU, by default 0.05
    n_prelu : int, optional
        The number of parameters for the parameterized ReLU, by default 1

    Returns
    -------
    nn.Module
        The activation function to use

    Raises
    ------
    NotImplementedError
        If act_type specified incorrectly
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'
                                  .format(act_type))
    return layer


# class ShortcutBlock(nn.Module):
#     def __init__(self, submodule):
#         super(ShortcutBlock, self).__init__()
#         self.sub = submodule

#     def forward(self, x):
#         output = x + self.sub(x)
#         return output

# def mean_channels(F):
#     assert(F.dim() == 4)
#     spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
#     return spatial_sum / (F.size(2) * F.size(3))

# def stdv_channels(F):
#     assert(F.dim() == 4)
#     F_mean = mean_channels(F)
    # F_variance = (F - F_mean).pow(2).sum(3, keepdim=True)
    # .sum(2, keepdim=True) / (F.size(2) * F.size(3))
#     return F_variance.pow(0.5)


def sequential(*args):
    """
    Generates a Sequential object that holds all functionality for the
    convolutional block.

    Returns
    -------
    nn.Sequential
        The convolutional block

    Raises
    ------
    NotImplementedError
        If given an OrderedDict as the padding
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support ' +
                                      'OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2,
                       kernel_size=3, stride=1):
    """
    Generates the pixel shuffling operation for the end of the network

    Parameters
    ----------
    in_channels : int
        The number of channels to take in
    out_channels : int
        The number of channels in the final output image
    upscale_factor : int, optional
        The amount of upsampling in the final image, by default 2
    kernel_size : int, optional
        The size of kernels to use, by default 3
    stride : int, optional
        The stride of kernels to use, by default 1

    Returns
    -------
    nn.Sequential
        The image upsampling block for the end of the network
    """
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2),
                      kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class ESA(nn.Module):
    """
    Class ESA is the enhanced spatial attention block as outlined in:

    https://openaccess.thecvf.com/content_CVPR_2020/html/
    Liu_Residual_Feature_Aggregation_Network_for_Image_Super-
    Resolution_CVPR_2020_paper.html

    This block is placed at the end of an RFDB and is used to bring attention
    to important feature channels
    """
    def __init__(self, n_feats, conv):
        """
        Constructor, see class documentation for more details

        Parameters
        ----------
        n_feats : int
            The number of input features
        conv : nn.Module
            The convolution module to apply to the features within the block
        """
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        The forward pass through the ESA block

        Parameters
        ----------
        x : torch.Tensor
            The input to the ESA

        Returns
        -------
        torch.Tensor
            The block output
        """
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear',
                           align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)

        return x * m


class RFDB(nn.Module):
    """
    Class RFDB is the residual feature distillation block for the residual
    feature distillation network. It is the basic building block that makes up
    most of the network.
    """
    def __init__(self, in_channels):
        """
        Constructor, see class documentation for more details

        Parameters
        ----------
        in_channels : int
            The number of channels for the input tensors
        """
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        """
        The forward pass through the RFDB block

        Parameters
        ----------
        x : torch.Tensor
            The input to the ESA

        Returns
        -------
        torch.Tensor
            The block output
        """
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3+r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4],
                        dim=1)
        out_fused = self.esa(self.c5(out))

        return out_fused


class RFDB2(nn.Module):
    """
    Class RFDB2 is the residual feature distillation block for the two layer
    residual feature distillation network. It is the basic building block
    that makes up most of the network.
    """
    def __init__(self, in_channels):
        """
        Constructor, see class documentation for more details

        Parameters
        ----------
        in_channels : int
            The number of channels for the input tensors
        """
        super(RFDB2, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3 = conv_layer(self.remaining_channels, self.dc, 3)

        self.act = activation('lrelu', neg_slope=0.05)
        self.c4 = conv_layer(self.dc*3, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        """
        The forward pass through the RFDB2 block

        Parameters
        ----------
        x : torch.Tensor
            The input to the ESA

        Returns
        -------
        torch.Tensor
            The block output
        """
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2+r_c1)

        r_c3 = self.act(self.c3(r_c2))

        out = torch.cat([distilled_c1, distilled_c2, r_c3], dim=1)
        out_fused = self.esa(self.c4(out))

        return out_fused


class RFDB1(nn.Module):
    """
    Class RFDB1 is the residual feature distillation block for the single layer
    residual feature distillation network. It is the basic building block
    that makes up most of the network.
    """
    def __init__(self, in_channels):
        """
        Constructor, see class documentation for more details

        Parameters
        ----------
        in_channels : int
            The number of channels for the input tensors
        """
        super(RFDB1, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2 = conv_layer(self.remaining_channels, self.dc, 3)

        self.act = activation('lrelu', neg_slope=0.05)
        self.c3 = conv_layer(self.dc*2, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        """
        The forward pass through the RFDB1 block

        Parameters
        ----------
        x : torch.Tensor
            The input to the ESA

        Returns
        -------
        torch.Tensor
            The block output
        """
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1+input)

        r_c2 = self.act(self.c2(r_c1))

        out = torch.cat([distilled_c1, r_c2], dim=1)
        out_fused = self.esa(self.c3(out))

        return out_fused


class FDCB(nn.Module):
    """
    Class FDCB is the feature distillation connection block for the
    feature distillation connection network. It is the basic building block
    that makes up most of the network.
    """
    def __init__(self, in_channels):
        """
        Constructor, see class documentation for more details

        Parameters
        ----------
        in_channels : int
            The number of channels for the input tensors
        """
        super(FDCB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc*4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        """
        The forward pass through the FDCB block

        Parameters
        ----------
        x : torch.Tensor
            The input to the ESA

        Returns
        -------
        torch.Tensor
            The block output
        """
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4],
                        dim=1)
        out_fused = self.esa(self.c5(out))

        return out_fused


# class SRB(nn.Module):
#     # the number of channels rc/dc need to be modified accordingly!!!
#     def __init__(self, in_channels):
#         super(SRB, self).__init__()
#         self.dc = self.distilled_channels = in_channels//2
#         self.rc = self.remaining_channels = in_channels
#         self.c1_r = conv_layer(in_channels, self.rc, 3)
#         self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
#         self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
#         self.c4_r = conv_layer(self.remaining_channels, self.rc, 3)
#         self.act = activation('lrelu', neg_slope=0.05)
#         self.c5 = conv_layer(self.rc, in_channels, 1)
#         self.esa = ESA(in_channels, nn.Conv2d)

#     def forward(self, input):
#         r_c1 = (self.c1_r(input))
#         r_c1 = self.act(r_c1 + input)

#         r_c2 = (self.c2_r(r_c1))
#         r_c2 = self.act(r_c2 + r_c1)

#         r_c3 = (self.c3_r(r_c2))
#         r_c3 = self.act(r_c3 + r_c2)

#         r_c4 = self.c4_r(r_c3)
#         r_c4 = self.act(r_c4 + r_c3)

#         out = r_c4
#         out_fused = self.esa(self.c5(out))

#         return out_fused


class BaseB(nn.Module):
    """
    Class BaseB is the base block for the base network. It is the basic
    building block that makes up most of the network.
    """
    def __init__(self, in_channels):
        """
        Constructor, see class documentation for more details

        Parameters
        ----------
        in_channels : int
            The number of channels for the input tensors
        """
        super(BaseB, self).__init__()
        self.dc = self.distilled_channels = in_channels//2
        self.rc = self.remaining_channels = in_channels
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        """
        The forward pass through the base block

        Parameters
        ----------
        x : torch.Tensor
            The input to the ESA

        Returns
        -------
        torch.Tensor
            The block output
        """
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1)

        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2)

        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3)

        r_c4 = self.act(self.c4(r_c3))

        out = r_c4
        out_fused = self.esa(self.c5(out))

        return out_fused
