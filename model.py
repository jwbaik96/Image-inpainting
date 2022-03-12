import torch
import torch.nn as nn
from model_ops import *


class Discriminator(nn.Module):
    def __init__(self, device):
        super(Discriminator, self).__init__()
        cnum = 48
        self.conv1 = DisConv(in_channels=4, out_channels=cnum)  # (cnum, 128, 128)
        self.conv2 = DisConv(in_channels=cnum, out_channels=2*cnum)  # (2*cnum, 64, 64)
        self.conv3 = DisConv(in_channels=2*cnum, out_channels=4*cnum)  # (4*cnum, 32, 32)
        self.conv4 = DisConv(in_channels=4*cnum, out_channels=4*cnum)  # (4*cnum, 16, 16)
        self.conv5 = DisConv(in_channels=4*cnum, out_channels=4*cnum)  # (4*cnum, 8, 8)
        self.conv6 = DisConv(in_channels=4*cnum, out_channels=8*cnum)  # (8*cnum, 4, 4)

        self.edge1 = DisConv(in_channels=4, out_channels=cnum//2, stride=4)  # (cnum/2, 64, 64)
        self.edge2 = DisConv(in_channels=cnum//2, out_channels=cnum)  # (cnum, 32, 32)
        self.edge3 = DisConv(in_channels=cnum, out_channels=2*cnum)  # (2*cnum, 16, 16)
        self.edge4 = DisConv(in_channels=2*cnum, out_channels=4*cnum)  # (4*cnum, 8, 8)
        self.edge5 = DisConv(in_channels=4*cnum, out_channels=4*cnum)  # (4*cnum, 4, 4)

        self.concat = DisConv(in_channels=12*cnum, out_channels=12*cnum, stride=1, activation=None)

        self.filter_h = torch.Tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]]).to(device)
        self.filter_v = torch.Tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).to(device)

    def forward(self, X):
        edge = self.sobel_edge_detector(X)

        edge = self.edge1(edge)
        edge = self.edge2(edge)
        edge = self.edge3(edge)
        edge = self.edge4(edge)
        edge = self.edge5(edge)

        f1 = self.conv1(X)
        f2 = self.conv2(f1)
        X = self.conv3(f2)
        X = self.conv4(X)
        X = self.conv5(X)
        X = self.conv6(X)

        X = torch.cat([X, edge], dim=1)
        X = self.concat(X)

        return X, (f1, f2)

    def sobel_edge_detector(self, imgs):
        # R,G,B channel split
        R_imgs, G_imgs, B_imgs, mask = torch.split(imgs, 1, dim=1)

        R_edge_h = F.conv2d(R_imgs, self.filter_h, stride=1, padding=1)
        G_edge_h = F.conv2d(G_imgs, self.filter_h, stride=1, padding=1)
        B_edge_h = F.conv2d(B_imgs, self.filter_h, stride=1, padding=1)
        R_edge_v = F.conv2d(R_imgs, self.filter_v, stride=1, padding=1)
        G_edge_v = F.conv2d(G_imgs, self.filter_v, stride=1, padding=1)
        B_edge_v = F.conv2d(B_imgs, self.filter_v, stride=1, padding=1)

        R_edge = R_edge_h ** 2 + R_edge_v ** 2
        G_edge = G_edge_h ** 2 + G_edge_v ** 2
        B_edge = B_edge_h ** 2 + B_edge_v ** 2

        return torch.cat([R_edge, G_edge, B_edge, mask], dim=1)


class WrappingModule(nn.Module):
    def __init__(self, cnum, device):
        super(WrappingModule, self).__init__()

        self.device = device

        self.main_module = RecurrentModule(cnum=cnum, device=device)
        self.coarse_module = ResModule(cnum=cnum)
        self.refine_module = AttentionModule(cnum=cnum, device=device)

    def forward(self, X, mask):
        masked_image = X * (1.0 - mask) + mask
        input_data = torch.cat((masked_image, mask), dim=1)

        # LP route
        LP_coarse = self.main_module(input_data, mask, self.coarse_module)

        # Refine route
        coarse_img = X * (1.0 - mask) + LP_coarse * mask
        input_data = torch.cat((coarse_img, mask), dim=1)
        refine = self.main_module(input_data, mask, self.refine_module, mask)

        return LP_coarse, refine

    def zero_buffer(self):
        self.main_module.zero_buffer()


class RecurrentModule(nn.Module):
    def __init__(self, cnum=16, device=None):
        """
        Recurrent main module for Inpainting network
        :param cnum: basic channel number
        :param device: device for cuda or cpu
        """
        super(RecurrentModule, self).__init__()

        # Gated Recurrent Convolution Encoder
        self.encoder_layer1 = RecurrentConv(in_channels=4, out_channels=cnum, kernel_size=5, stride=1, device=device)     # (cnum, 256, 256)   # Information Transfer 1
        self.encoder_layer2 = RecurrentConv(in_channels=cnum, out_channels=2*cnum, kernel_size=3, stride=2, device=device)    # (2*cnum, 128, 128)
        self.encoder_layer3 = RecurrentConv(in_channels=2*cnum, out_channels=2*cnum, kernel_size=3, stride=1, device=device)    # (2*cnum, 128, 128)  # Information Transfer 2
        self.encoder_layer4 = RecurrentConv(in_channels=2*cnum, out_channels=4*cnum, kernel_size=3, stride=2, normalization=True, device=device)    # (4*cnum, 64, 64)
        self.encoder_layer5 = RecurrentConv(in_channels=4*cnum, out_channels=4*cnum, kernel_size=3, stride=1, normalization=True, device=device)    # (4*cnum, 64, 64)
        self.encoder_layer6 = RecurrentConv(in_channels=4*cnum, out_channels=4*cnum, kernel_size=3, stride=1, normalization=True, device=device)    # (4*cnum, 64, 64)

        # Gated Recurrent Deconvolution Decoder
        self.decoder_layer1 = RecurrentConv(in_channels=4*cnum, out_channels=4*cnum, kernel_size=3, stride=1, normalization=True, device=device)  # (4*cnum, 64, 64)
        self.decoder_layer2 = RecurrentConv(in_channels=4*cnum, out_channels=4*cnum, kernel_size=3, stride=1, normalization=True, device=device)  # (4*cnum, 64, 64)
        self.decoder_layer3 = RecurrentDeConv(in_channels=4*cnum, out_channels=2*cnum, normalization=True, device=device)     # (2*cnum, 128, 128)
        self.decoder_layer4 = RecurrentConv(in_channels=4*cnum, out_channels=2*cnum, kernel_size=3, stride=1, device=device)  # (2*cnum, 128, 128)    # Information Receiver 2
        self.decoder_layer5 = RecurrentDeConv(in_channels=2*cnum, out_channels=cnum, device=device)       # (cnum, 256, 256)
        self.decoder_layer6 = RecurrentConv(in_channels=2*cnum+1, out_channels=cnum, kernel_size=3, stride=1, device=device)  # (cnum, 256, 256)    # Information Receiver 1
        self.decoder_layer7 = RecurrentConv(in_channels=cnum, out_channels=cnum//2, kernel_size=3, stride=1, device=device)  # (cnum/2, 256, 256)
        self.decoder_layer8 = RecurrentConv(in_channels=cnum//2, out_channels=3, kernel_size=3, stride=1, activation=None, device=device)  # (3, 256, 256)

    def forward(self, X, mask, embedded_module=None, *args):
        # Encoding Part
        transfer_1 = self.encoder_layer1(X)
        X = self.encoder_layer2(transfer_1)
        transfer_2 = self.encoder_layer3(X)
        X = self.encoder_layer4(transfer_2)
        X = self.encoder_layer5(X)
        X = self.encoder_layer6(X)

        if embedded_module is not None:
            X = embedded_module(X, *args)

        # Decoding Part
        X = self.decoder_layer1(X)
        X = self.decoder_layer2(X)
        X = self.decoder_layer3(X)
        X = torch.cat((X, transfer_2), dim=1)
        X = self.decoder_layer4(X)
        X = self.decoder_layer5(X)
        X = torch.cat((X, transfer_1, (1.0-mask)), dim=1)
        X = self.decoder_layer6(X)
        X = self.decoder_layer7(X)
        X = self.decoder_layer8(X)
        X = (torch.tanh(X) + 1.0) / 2.0
        return X

    def zero_buffer(self):
        self.encoder_layer1.zero_buffer()
        self.encoder_layer2.zero_buffer()
        self.encoder_layer3.zero_buffer()
        self.encoder_layer4.zero_buffer()
        self.encoder_layer5.zero_buffer()
        self.encoder_layer6.zero_buffer()

        self.decoder_layer1.zero_buffer()
        self.decoder_layer2.zero_buffer()
        self.decoder_layer3.zero_buffer()
        self.decoder_layer4.zero_buffer()
        self.decoder_layer5.zero_buffer()
        self.decoder_layer6.zero_buffer()
        self.decoder_layer7.zero_buffer()
        self.decoder_layer8.zero_buffer()


class DMFBlockModule(nn.Module):
    def __init__(self, cnum=16):
        super(DMFBlockModule, self).__init__()

        self.dmfb = nn.Sequential(
            DMFBlock(in_channels=4*cnum, out_channels=4*cnum),
            DMFBlock(in_channels=4*cnum, out_channels=4*cnum),
            DMFBlock(in_channels=4*cnum, out_channels=4*cnum),
            DMFBlock(in_channels=4*cnum, out_channels=4*cnum)
        )

    def forward(self, X):
        return self.dmfb(X)


class AttentionModule(nn.Module):
    def __init__(self, cnum=16, device=None):
        super(AttentionModule, self).__init__()
        self.res_module = ResModule(cnum)
        self.dmfb_module = DMFBlockModule(cnum)
        self.attention = ContextualAttention(device=device)
        self.concatenation = nn.Sequential(
            nn.Conv2d(in_channels=12*cnum, out_channels=8*cnum, kernel_size=3, stride=1, padding=1),   # (8*cnum, 64, 64)
            nn.InstanceNorm2d(num_features=8*cnum, track_running_stats=False),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=8*cnum, out_channels=4*cnum, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(num_features=4*cnum, track_running_stats=False),
            nn.ELU(inplace=True)
        )

    def forward(self, X, mask):
        res = self.res_module(X)
        attention = self.attention(X, mask)
        dmfb = self.dmfb_module(X)

        # Concatenation along channel axis
        concat = torch.cat((res, attention, dmfb), dim=1)
        concat = self.concatenation(concat)

        return concat


class ResModule(nn.Module):
    def __init__(self, cnum):
        super(ResModule, self).__init__()

        blocks = []
        for _ in range(8):
            block = ResnetBlock(4*cnum, 2)
            blocks.append(block)

        self.res_module = nn.Sequential(*blocks)

    def forward(self, X):
        return self.res_module(X)
