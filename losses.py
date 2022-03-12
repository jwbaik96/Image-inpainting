import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import VGG19


class L1Loss(nn.Module):
    def __init__(self, weight=1.):
        """
        L1 Reconstruction Loss for two images.
        """
        super(L1Loss, self).__init__()
        self.weight = weight

    def forward(self, ground_truth, predict):
        return self.weight * torch.mean(torch.abs(ground_truth - predict))


class L1LossWithMask(nn.Module):
    def __init__(self, weight=1.):
        super(L1LossWithMask, self).__init__()
        self.weight = weight

    def forward(self, ground_truth, predict, mask):
        return self.weight * torch.mean(torch.abs(ground_truth - predict) * mask)


class SNPatchGenLoss(nn.Module):
    def __init__(self, weight=1.):
        """
        SN-Patch discriminator Generator loss
        :param weight: weight
        """
        super(SNPatchGenLoss, self).__init__()
        self.weight = weight

    def forward(self, DGz):
        """
        :param DGz_list: Discriminator output of fake image and features list
        :return: Tensor
        """
        return -1.0 * self.weight * torch.mean(DGz)


class SNPatchDisLoss(nn.Module):
    def __init__(self, weight=1.):
        """
        SN-Patch discriminator Discriminator loss
        :param weight: weight
        """
        super(SNPatchDisLoss, self).__init__()
        self.weight = weight

    def forward(self, Dx, DGz):
        """
        :param Dx: Discriminator output of real image
        :param DGz: Discriminator output of fake image
        :return: Tensor
        """
        real = torch.mean(F.relu((1 - Dx), inplace=True))
        fake = torch.mean(F.relu((1 + DGz), inplace=True))

        return self.weight * (real + fake)


class VGGLoss(nn.Module):
    def __init__(self, perceptual_loss_weight, style_loss_weight, device=None):
        super(VGGLoss, self).__init__()
        self.perceptual_loss_weight = perceptual_loss_weight
        self.style_loss_weight = style_loss_weight
        self.vgg = VGG19().to(device)
        self.p_L1_loss = nn.L1Loss()
        self.s_L1_loss = nn.L1Loss()

    def forward(self, GT, pred):
        GT_vgg = self.vgg(GT)
        pred_vgg = self.vgg(pred)

        p_loss = 0.0
        p_loss += self.p_L1_loss(pred_vgg['relu1_1'], GT_vgg['relu1_1'])
        p_loss += self.p_L1_loss(pred_vgg['relu2_1'], GT_vgg['relu2_1'])
        p_loss += self.p_L1_loss(pred_vgg['relu3_1'], GT_vgg['relu3_1'])
        p_loss += self.p_L1_loss(pred_vgg['relu4_1'], GT_vgg['relu4_1'])
        p_loss += self.p_L1_loss(pred_vgg['relu5_1'], GT_vgg['relu5_1'])

        s_loss = 0.0
        s_loss += self.s_L1_loss(self.gram_matrix(pred_vgg['relu2_2']), self.gram_matrix(GT_vgg['relu2_2']))
        s_loss += self.s_L1_loss(self.gram_matrix(pred_vgg['relu3_4']), self.gram_matrix(GT_vgg['relu3_4']))
        s_loss += self.s_L1_loss(self.gram_matrix(pred_vgg['relu4_4']), self.gram_matrix(GT_vgg['relu4_4']))
        s_loss += self.s_L1_loss(self.gram_matrix(pred_vgg['relu5_2']), self.gram_matrix(GT_vgg['relu5_2']))

        return self.perceptual_loss_weight * p_loss, self.style_loss_weight * s_loss

    def gram_matrix(self, X):
        N, C, H, W = X.size()

        f = X.view(N, C, H * W)
        f_T = f.transpose(1, 2)

        # bmm -> batch matrix multiplication
        gram = torch.bmm(f, f_T) / (C * H * W)   # (N, C, C)

        return gram


class PerceptualLoss(nn.Module):
    def __init__(self, weight=1.0, device=None):
        super(PerceptualLoss, self).__init__()
        self.weight = weight
        self.vgg = VGG19().to(device)
        self.L1loss = nn.L1Loss()

    def forward(self, GT, pred):
        GT_vgg = self.vgg(GT)
        pred_vgg = self.vgg(pred)

        loss = 0.0
        loss += self.L1loss(pred_vgg['relu1_1'], GT_vgg['relu1_1'])
        loss += self.L1loss(pred_vgg['relu2_1'], GT_vgg['relu2_1'])
        loss += self.L1loss(pred_vgg['relu3_1'], GT_vgg['relu3_1'])
        loss += self.L1loss(pred_vgg['relu4_1'], GT_vgg['relu4_1'])
        loss += self.L1loss(pred_vgg['relu5_1'], GT_vgg['relu5_1'])
        return self.weight * loss
#
#
# class StyleLoss(nn.Module):
#     def __init__(self, weight=1.0, device=None):
#         super(StyleLoss, self).__init__()
#         self.weight = weight
#         self.vgg = VGG19().to(device)
#         self.L1loss = nn.L1Loss()
#
#     def forward(self, GT, pred):
#         GT_vgg = self.vgg(GT)
#         pred_vgg = self.vgg(pred)
#
#         loss = 0.0
#         loss += self.L1loss(self.gram_matrix(pred_vgg['relu2_2']), self.gram_matrix(GT_vgg['relu2_2']))
#         loss += self.L1loss(self.gram_matrix(pred_vgg['relu3_4']), self.gram_matrix(GT_vgg['relu3_4']))
#         loss += self.L1loss(self.gram_matrix(pred_vgg['relu4_4']), self.gram_matrix(GT_vgg['relu4_4']))
#         loss += self.L1loss(self.gram_matrix(pred_vgg['relu5_2']), self.gram_matrix(GT_vgg['relu5_2']))
#         return self.weight * loss
#
#     def gram_matrix(self, X):
#         N, C, H, W = X.size()
#
#         f = X.view(N, C, H * W)
#         f_T = f.transpose(1, 2)
#
#         # bmm -> batch matrix multiplication
#         gram = torch.bmm(f, f_T) / (C * H * W)   # (N, C, C)
#
#         return gram


class SobelEdgeLoss(nn.Module):
    def __init__(self, weight=1.0, device=None):
        """
        Sobel Edge Loss
        :param weight: weight
        :param threshold: threshold for sobel edge detector
        :param device: torch device for cuda or cpu
        """
        super(SobelEdgeLoss, self).__init__()

        self.weight = weight
        self.NaN_escaper = 1e-6

        # horizon and vertical edge detection kernel, '4' is normalization.
        self.filter_h = torch.Tensor([[[[3, 10, 3], [0, 0, 0], [-3, -10, -3]]]]).to(device) / 16
        self.filter_v = torch.Tensor([[[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]]]).to(device) / 16

    def forward(self, GT, pred):
        GT_R, GT_G, GT_B = self.sobel_edge_detector(GT)
        pred_R, pred_G, pred_B = self.sobel_edge_detector(pred)

        R_loss = torch.mean(torch.abs(pred_R - GT_R))
        G_loss = torch.mean(torch.abs(pred_G - GT_G))
        B_loss = torch.mean(torch.abs(pred_B - GT_B))

        return self.weight * (R_loss + G_loss + B_loss)

    def sobel_edge_detector(self, imgs):
        # R,G,B channel split
        R_imgs, G_imgs, B_imgs = torch.split(imgs, 1, dim=1)
        R_imgs = F.pad(R_imgs, [1, 1, 1, 1], 'reflect')
        G_imgs = F.pad(G_imgs, [1, 1, 1, 1], 'reflect')
        B_imgs = F.pad(B_imgs, [1, 1, 1, 1], 'reflect')

        R_edge_h = F.conv2d(R_imgs, self.filter_h, stride=1)
        G_edge_h = F.conv2d(G_imgs, self.filter_h, stride=1)
        B_edge_h = F.conv2d(B_imgs, self.filter_h, stride=1)
        R_edge_v = F.conv2d(R_imgs, self.filter_v, stride=1)
        G_edge_v = F.conv2d(G_imgs, self.filter_v, stride=1)
        B_edge_v = F.conv2d(B_imgs, self.filter_v, stride=1)

        R_edge = torch.abs(R_edge_h) + torch.abs(R_edge_v)
        G_edge = torch.abs(G_edge_h) + torch.abs(G_edge_v)
        B_edge = torch.abs(B_edge_h) + torch.abs(B_edge_v)

        R_edge_min, R_edge_max = torch.min(R_edge), torch.max(R_edge)
        G_edge_min, G_edge_max = torch.min(G_edge), torch.max(G_edge)
        B_edge_min, B_edge_max = torch.min(B_edge), torch.max(B_edge)

        R_edge = (R_edge - R_edge_min) / (R_edge_max - R_edge_min + self.NaN_escaper)
        G_edge = (G_edge - G_edge_min) / (G_edge_max - G_edge_min + self.NaN_escaper)
        B_edge = (B_edge - B_edge_min) / (B_edge_max - B_edge_min + self.NaN_escaper)

        return R_edge, G_edge, B_edge


class FeatureLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(FeatureLoss, self).__init__()
        self.weight = weight

    def forward(self, GT_features, pred_features):
        l1 = torch.mean(torch.abs(GT_features[0] - pred_features[0]))
        l2 = torch.mean(torch.abs(GT_features[1] - pred_features[1]))

        return self.weight * (l1 + l2)

