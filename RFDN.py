# Import modules
import torch
import torch.nn as nn
import block as B


class RFDN(nn.Module):
    """
    Class RFDN is the residual feature distillation network which we want to
    evaluate and reduce. This class was taken from the RFDN class which can
    be found at:

    https://arxiv.org/abs/2009.11551
    https://github.com/njulj/RFDN

    Parameters
    ----------
    in_nc : int
        The number of input channels to the network, by default 3
    nf : int
        The number of input feature channels per block of the network
    num_modules : int
        The number of blocks in the network, by default 4
    out_nc : int
        The number of channels which should be output by the network, by
        default 3
    upscale : int
        The amount of upscaling to be done, by default 4
    """
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        """
        Constructor, see class documentation for more details.

        Parameters
        ----------
        in_nc : int
            The number of input channels to the network, by default 3
        nf : int
            The number of input feature channels per block of the network
        num_modules : int
            The number of blocks in the network, by default 4
        out_nc : int
            The number of channels which should be output by the network, by
            default 3
        upscale : int
            The amount of upscaling to be done, by default 4
        """
        super(RFDN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = B.RFDB(in_channels=nf)
        self.B2 = B.RFDB(in_channels=nf)
        self.B3 = B.RFDB(in_channels=nf)
        self.B4 = B.RFDB(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0

    def forward(self, input):
        """
        Computes the forward pass of the network

        Parameters
        ----------
        input : torch.Tensor
            The input image tensor to compute the super-resolution of, which
            should be of the form (N, C, W, H).

        Returns
        -------
        torch.Tensor
            The output tensor, which is the super-resoluted image tensor of
            the input argument.
        """
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def __repr__(self):
        return "RFDN"


class RFDN1(nn.Module):
    """
    Class RFDN1 is the residual feature distillation network which with number
    of layers in the residual feature distillation blocks reduced to 1. This
    class is necessary to evaluate the efficacy of the layered
    architecture in the residual feature distillation blocks.
    This class was adapted from the RFDN class which can be found at:

    https://arxiv.org/abs/2009.11551
    https://github.com/njulj/RFDN

    Parameters
    ----------
    in_nc : int
        The number of input channels to the network, by default 3
    nf : int
        The number of input feature channels per block of the network
    num_modules : int
        The number of blocks in the network, by default 4
    out_nc : int
        The number of channels which should be output by the network, by
        default 3
    upscale : int
        The amount of upscaling to be done, by default 4
    """
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=4):
        """
        Constructor, see class documentation for more details.

        Parameters
        ----------
        in_nc : int
            The number of input channels to the network, by default 3
        nf : int
            The number of input feature channels per block of the network
        num_modules : int
            The number of blocks in the network, by default 4
        out_nc : int
            The number of channels which should be output by the network, by
            default 3
        upscale : int
            The amount of upscaling to be done, by default 4
        """
        super(RFDN1, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = B.RFDB1(in_channels=nf)
        self.B2 = B.RFDB1(in_channels=nf)
        self.B3 = B.RFDB1(in_channels=nf)
        self.B4 = B.RFDB1(in_channels=nf)
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0

    def forward(self, input):
        """
        Computes the forward pass of the network

        Parameters
        ----------
        input : torch.Tensor
            The input image tensor to compute the super-resolution of, which
            should be of the form (N, C, W, H).

        Returns
        -------
        torch.Tensor
            The output tensor, which is the super-resoluted image tensor of
            the input argument.
        """
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def __repr__(self):
        return "RFDN1"
