import torch
import torch.nn as nn
from utils import *


class RecurrentConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='SYMMETRIC', activation=nn.ELU(inplace=True), normalization=False, device=None):
        """
        Define Recurrent Convolution module
        This module must be satisfy the condition : input size = output size

        :param in_channels: input channel size
        :param out_channels: output channel size
        :param kernel_size:  kernel size
        :param stride:  Convolution srtide
        :param padding:  VALID, SAME, SYMMETRIC. default is SYMMETRIC
        :param normalization: default False
        :param activation: default ELU
        :return: Torch
        """
        super(RecurrentConv, self).__init__()
        self.buffer = None
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.normalization = normalization
        self.device = device
        self.N = 0
        self.output_size = 0

        # Padding Process
        assert padding in ['VALID', 'SAME', 'SYMMETRIC']
        if padding == 'VALID':
            self.p = 0
            self.pad = None
        elif padding == 'SAME':
            self.p = int((kernel_size - 1) / 2)
            self.pad = nn.ConstantPad2d(padding=self.p, value=0)
        elif padding == 'SYMMETRIC':
            self.p = int((kernel_size - 1) / 2)
            self.pad = nn.ReflectionPad2d(padding=self.p)
        else:
            raise ValueError("padding only available 'SAME' or 'SYMMETRIC'")

        # other layers
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, bias=True)
        self.buffer_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if normalization:
            self.norm = nn.InstanceNorm2d(num_features=out_channels, track_running_stats=False)
        self.activation = activation

        # Initializaion
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
                if not normalization:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, X):
        # initial buffer setting
        N, C, H, W = X.shape
        if self.buffer is None:
            self.N = N
            self.output_size = (H + 2 * self.p - self.kernel_size) // self.stride + 1
            self.buffer = torch.zeros(size=(self.N, self.out_channels, self.output_size, self.output_size), device=self.device)

        # Buffer convolution
        buffer = self.buffer_conv(self.buffer)

        # padding & convolution
        if self.pad is not None:
            X = self.pad(X)
        X = self.conv(X)

        X = X + buffer

        # normalization
        if self.normalization:
            X = self.norm(X)

        # Activation
        if self.activation is not None:
            X = self.activation(X)

        self.buffer = X
        return X

    def zero_buffer(self):
        """
        make self.buffer to None
        iteration 후에 self.buffer를 다시 0으로 만들기 위해.
        recurrent model에서 항상 맨 처음은 buffer에 0의 값이 존재해야 한다.
        """
        self.buffer = torch.zeros(size=(self.N, self.out_channels, self.output_size, self.output_size), device=self.device)


class RecurrentDeConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, kernel_size=3, stride=1,
                 padding='SYMMETRIC', activation=nn.ELU(inplace=True), normalization=False, device=None):
        """
        Define Recurrent Deconvolution module

        :param in_channels: Input channel size
        :param out_channels:  Output channel size
        :param kernel_size:  kernel size
        :param stride:  Convolution srtide
        :param padding:  SAME, VALID, SYMMETRIC. default is SYMMETRIC
        :param normalization: default False
        :param activation: default ELU
        :param device: cuda or cpu device
        :return: Torch
        """
        super(RecurrentDeConv, self).__init__()

        self.upscale = nn.UpsamplingNearest2d(scale_factor=scale)
        self.recurrent_conv = RecurrentConv(in_channels, out_channels, kernel_size, stride, padding, activation, normalization, device)

    def forward(self, X):
        X = self.upscale(X)
        X= self.recurrent_conv(X)
        return X

    def zero_buffer(self):
        self.recurrent_conv.zero_buffer()


class DisConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, dilation=1, padding='SAME',
                 bias=True, activation=nn.LeakyReLU(negative_slope=0.2, inplace=True)):
        """
        Define Disriminator Convolution
        The VanillaConv2d is for discriminator convolution layer.
        Activation is set to leaky_relu with negative slope 0.2
        Using Spectral Normalization

        :param in_channels: Input channel size
        :param out_channels:  Output channel size
        :param kernel_size: int. kernel_size
        :param stride: int. stride
        :param dilation: dilation rate
        :param bias: boolean
        :param activation: nn module or None

        :return: Torch
        """
        super(DisConv, self).__init__()
        self.padding = padding

        # Padding
        # confirm padding is 'SYMMETRIC' or 'SAME' or 'VALID', if not, raise error.
        assert padding in ['SYMMETRIC', 'SAME', 'VALID']
        if padding == 'VALID':
            self.pad = None
        elif padding == 'SAME':
            p = int(dilation * (kernel_size - 1) / 2)
            self.pad = nn.ConstantPad2d(padding=p, value=0)
        else:
            # SYMMETRIC
            p = int(dilation * (kernel_size - 1) / 2)
            self.pad = nn.ReflectionPad2d(padding=p)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation, bias=bias)

        # Here use pytorch spectral normalization, just for efficiency.
        self.spec_norm = nn.utils.spectral_norm(self.conv)

        # Activation
        self.activation = activation

        # Initializing
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if bias:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, X):
        if self.padding == 'SAME' or self.padding == 'SYMMETRIC':
            # padding process
            X = self.pad(X)
        X = self.spec_norm(X)

        if self.activation is not None:
            X = self.activation(X)
        return X


class ContextualAttention(nn.Module):
    """
    Contextual Attention layer

    :param kernel_size: (int) kernel size for extracting patches
    :param stride: (int) stride for extracting patches
    :param dilation: (int) dilation for extracting patches
    :param fuse_k: (int) Attention Propagation fusion degree
    :param softmax_scale: (int or float) scaling for attention score
    :param fuse: (boolean) fusion

    :return A Tensor
    """
    def __init__(self, kernel_size=3, stride=1, dilation=2, fuse_k=3, softmax_scale=10, fuse=False, device=None):
        super(ContextualAttention, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.device = device

    def forward(self, X, mask=None):
        """
        forward process has 3 steps.
        1. Extract patches for conv, deconv
        2. Convolution / Attention score
        4. Deconvolution

        :param X: (4-D Tensor) input feature
        :param mask: mask for background. 1 means that this patch is not available
        :return:
        """
        # --------------------    Step 1    --------------------------
        # Extract and Generate filter/weight for deconvolution
        # raw_ means 'original'
        # get shapes
        raw_X_shape = list(X.size())

        # Extract patches from input feature only use stride and dilation value.
        # stride is stride * dilation
        # W is extracted patches, and now it is kernel/weight for deconvolution
        # Shape of raw_W is  (N, C*K*K, L) where L is total kernel sliding blocks.
        # ksize = 4, stride = 2, dilation = 1, padding = 'SAME'
        # deconv_W size -> (N, 128*4*4, 32*32)
        deconv_W_ksize = 2 * self.dilation    # 4
        deconv_W = extract_image_patches(X, kernel_size=deconv_W_ksize, stride=self.stride*self.dilation, dilation=1, padding='SAME')

        # deconv_W shape : (N, C*K*K, L).  Reshape deconv_W into (N, C, K, K, L)
        deconv_W = deconv_W.view(raw_X_shape[0], raw_X_shape[1], deconv_W_ksize, deconv_W_ksize, -1)

        # transpose to (N, L, C, K, K)  i.e.  (N, 32*32, 128, 4, 4)
        # N: batch size,  32*32: in_channels,  128: out_channels,  4: kernel size
        deconv_W = deconv_W.permute(0, 4, 1, 2, 3)

        # Split deconv_W along batch axis  -> output is tuple with len() = batch_size
        # each element has shape of (1, 32*32, 128, 4, 4)
        deconv_W_groups = torch.split(deconv_W, 1, dim=0)

        # Downscaling foreground and background for convolution and
        # efficiency of computation.
        # This X is the input of convolution step.
        # X shape :  (N, 128, 32, 32)
        X = F.interpolate(X, scale_factor=1.0 / self.dilation, mode='nearest')

        # get size
        X_shape = list(X.size())

        # Split foreground along batch axis
        # each element has shape of (1, 128, 32, 32)
        X_groups = torch.split(X, 1, dim=0)

        # Extract patches.
        # conv_W is extracted patches, and now it is kernel/weight for convolution
        # Shape of W is  (N, C*K*K, L) where L is total kernel sliding blocks.
        # ksize = 3, stride = 1, dilation = 1, padding = 'SAME'
        # deconv_W size -> (N, 128*3*3, 32*32)
        conv_W = extract_image_patches(X, kernel_size=self.kernel_size, stride=self.stride, dilation=1, padding='SAME')

        # conv_W shape : (N, C*K*K, L).  Reshape conv_W into (N, C, K, K, L)
        conv_W = conv_W.view(X_shape[0], X_shape[1], self.kernel_size, self.kernel_size, -1)

        # transpose to (N, L, C, K, K)  i.e.  (N, 32*32, 128, 3, 3)
        # N: batch size,  32*32: out_channels,  128: in_channels,  3: kernel size
        conv_W = conv_W.permute(0, 4, 1, 2, 3)

        # Split conv_W along batch axis  -> output is tuple with len() = batch_size
        # each element has shape of (1, 32*32, 128, 3, 3)
        conv_W_groups = torch.split(conv_W, 1, dim=0)

        # Mask Processing
        if mask is None:
            mask = torch.zeros([X_shape[0], 1, X_shape[2], X_shape[3]]).to(self.device)    # (N, 1, 32, 32)
        else:
            # input mask size is 256. so downscaling
            mask = F.interpolate(mask, scale_factor=1.0 / (4*self.dilation), mode='nearest')

        mask_shape = list(mask.size())

        # extract mask patches and setting for masking.
        # if a patch is invalid, mask is 0
        # So, after convolution, we multiply mask mm
        m = extract_image_patches(mask, kernel_size=self.kernel_size, stride=self.stride, dilation=1, padding='SAME')
        m = m.view(mask_shape[0], mask_shape[1], self.kernel_size, self.kernel_size, -1)    # (N, 1, 3, 3, 32*32)
        m = m.permute(0, 4, 1, 2, 3)    # (N, 32*32, 1, 3, 3)
        m = m[0]    # (32*32, 1, 3, 3)
        mm = (torch.mean(m, dim=(1, 2, 3), keepdim=True) == 0).float()  # (32*32, 1, 1, 1)
        mm = mm.permute(1, 0, 2, 3)     # (1, 32*32, 1, 1)

        # --------------------- Convolution / Attention score step ----------------------
        y = []  # for final output
        k = self.fuse_k
        fuse_weight = torch.eye(k).view(1, 1, k, k).to(self.device)    # (1, 1, k, k)

        for xi, conv_wi, deconv_wi in zip(X_groups, conv_W_groups, deconv_W_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            conv_wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            deconv_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            eps = torch.Tensor([1e-4]).to(self.device)  # Prevent for zero dividing
            conv_wi = conv_wi[0]    # (32*32, 128, 3, 3)
            max_wi = torch.max(torch.sqrt(torch.sum(torch.pow(conv_wi, 2), dim=(1, 2, 3), keepdim=True)), eps)
            conv_wi_normed = conv_wi / max_wi

            p = int(self.kernel_size / 2)     # for 'SAME' padding
            yi = F.conv2d(xi, conv_wi_normed, stride=1, padding=p)  # (1, L, H, W) = (1, 32*32, 32, 32)

            # fusing
            if self.fuse:
                # reshape (1, 1, L, H*W)
                yi = yi.view(1, 1, X_shape[2] * X_shape[3], X_shape[2] * X_shape[3])    # (1, 1, 32*32, 32*32)

                fuse_pad = int(k / 2)   # for 'SAME' padding
                # apply convolution along some direction
                yi = F.conv2d(yi, fuse_weight, stride=1, padding=fuse_pad)    # (1, 1, 32*32, 32*32)
                yi = yi.contiguous().view(1, X_shape[2], X_shape[3], X_shape[2], X_shape[3])  # (1, 32, 32, 32, 32)
                # for convolution with another direction, switch height and width
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, X_shape[2] * X_shape[3], X_shape[2] * X_shape[3])  # (1, 1, 32*32, 32*32)

                yi = F.conv2d(yi, fuse_weight, stride=1, padding=fuse_pad)    # (1, 1, 32*32, 32*32)
                yi = yi.contiguous().view(1, X_shape[3], X_shape[2], X_shape[3], X_shape[2])  # (1, 32, 32, 32, 32)

                # switch height and width again
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, X_shape[2] * X_shape[3], X_shape[2], X_shape[3])    # (1, 32*32, 32, 32)
            yi = yi * mm
            # softmax along L axis
            yi = F.softmax(yi * self.softmax_scale, dim=1)
            yi = yi * mm

            # ------------------- Deconvolution step -------------------------
            yi = F.conv_transpose2d(yi, deconv_wi[0], stride=self.dilation, padding=1) / 4.0    # (B=1, C=128, H=64, W=64)
            y.append(yi)

        y = torch.cat(y, dim=0)
        y.contiguous().view(raw_X_shape)

        return y


class DMFBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Dense Multi-scale Fusion Block

        :param in_channels: input channel number
        :param out_channels: output channel number
        """
        super(DMFBlock, self).__init__()
        block_channels = in_channels // 4

        self.conv1 = nn.Conv2d(in_channels, block_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(num_features=block_channels, track_running_stats=False)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2_1 = nn.Conv2d(block_channels, block_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2_2 = nn.Conv2d(block_channels, block_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv2_4 = nn.Conv2d(block_channels, block_channels, kernel_size=3, stride=1, padding=4, dilation=4)
        self.conv2_8 = nn.Conv2d(block_channels, block_channels, kernel_size=3, stride=1, padding=8, dilation=8)

        self.conv3_1 = nn.Conv2d(block_channels, block_channels, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(block_channels, block_channels, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(block_channels, block_channels, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
        self.norm4 = nn.InstanceNorm2d(num_features=out_channels, track_running_stats=False)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, X):
        # For residual
        res_X = X

        # First column convolution
        conv1 = self.relu1(self.norm1(self.conv1(X)))

        # Combination
        conv2_1 = self.conv2_1(conv1)
        conv2_2 = self.conv2_2(conv1)
        conv2_4 = self.conv2_4(conv1)
        conv2_8 = self.conv2_8(conv1)

        K2 = self.conv3_1(conv2_1 + conv2_2)
        K3 = self.conv3_2(K2 + conv2_4)
        K4 = self.conv3_3(K3 + conv2_8)

        # Concatenation along channel axis
        concat = torch.cat((conv2_1, K2, K3, K4), dim=1)

        # conv1x1
        out = self.relu4(self.norm4(self.conv4(concat)))

        return out + res_X


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation),
            nn.InstanceNorm2d(num_features=dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1),
            nn.InstanceNorm2d(num_features=dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out




