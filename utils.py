import torch
import torch.nn.functional as F
from torchvision import models


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features

        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for index in range(2):
            self.relu1_1.add_module(str(index), features[index])

        for index in range(2, 4):
            self.relu1_2.add_module(str(index), features[index])

        for index in range(4, 7):
            self.relu2_1.add_module(str(index), features[index])

        for index in range(7, 9):
            self.relu2_2.add_module(str(index), features[index])

        for index in range(9, 12):
            self.relu3_1.add_module(str(index), features[index])

        for index in range(12, 14):
            self.relu3_2.add_module(str(index), features[index])

        for index in range(14, 16):
            self.relu3_3.add_module(str(index), features[index])

        for index in range(16, 18):
            self.relu3_4.add_module(str(index), features[index])

        for index in range(18, 21):
            self.relu4_1.add_module(str(index), features[index])

        for index in range(21, 23):
            self.relu4_2.add_module(str(index), features[index])

        for index in range(23, 25):
            self.relu4_3.add_module(str(index), features[index])

        for index in range(25, 27):
            self.relu4_4.add_module(str(index), features[index])

        for index in range(27, 30):
            self.relu5_1.add_module(str(index), features[index])

        for index in range(30, 32):
            self.relu5_2.add_module(str(index), features[index])

        for index in range(32, 34):
            self.relu5_3.add_module(str(index), features[index])

        for index in range(34, 36):
            self.relu5_4.add_module(str(index), features[index])

        # Don't need to gradient update
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        relu1_1 = self.relu1_1(X)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


class LowPassFilter(torch.nn.Module):
    def __init__(self, device):
        super(LowPassFilter, self).__init__()
        self.filter = torch.ones((1, 1, 5, 5), device=device) / 25.0

    def forward(self, imgs):
        # R,G,B channel split
        R_imgs, G_imgs, B_imgs = torch.split(imgs, 1, dim=1)

        R_imgs = F.conv2d(R_imgs, self.filter, stride=1)
        G_imgs = F.conv2d(G_imgs, self.filter, stride=1)
        B_imgs = F.conv2d(B_imgs, self.filter, stride=1)

        return torch.cat((R_imgs, G_imgs, B_imgs), dim=1)


class HighPassFilter(torch.nn.Module):
    def __init__(self, device):
        super(HighPassFilter, self).__init__()
        self.filter = torch.Tensor([[[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]]]).to(device)

    def forward(self, imgs):
        # R,G,B channel split
        R_imgs, G_imgs, B_imgs = torch.split(imgs, 1, dim=1)

        R_imgs = F.conv2d(R_imgs, self.filter, stride=1, padding=1)
        G_imgs = F.conv2d(G_imgs, self.filter, stride=1, padding=1)
        B_imgs = F.conv2d(B_imgs, self.filter, stride=1, padding=1)

        return torch.cat((R_imgs, G_imgs, B_imgs), dim=1)


def extract_image_patches(X, kernel_size, stride, padding, dilation):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding: 'SAME', 'VALID', 'SYMMETRIC'
    :param X: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param kernel_size: The size of the sliding window for each dimension of images
    :param stride: stride
    :param dilation: dilation rate
    :return: A Tensor
    """
    assert len(X.size()) == 4

    # padding process
    assert padding in ['SAME', 'VALID', 'SYMMETRIC']
    if padding == 'VALID':
        pass
    elif padding == 'SAME':
        p = int(dilation * (kernel_size - 1.0) / 2.0)
        X = F.pad(X, pad=[p, p, p, p], mode='constant', value=0)
    elif padding == 'SYMMETRIC':
        p = int(dilation * (kernel_size - 1.0) / 2.0)
        X = F.pad(X, pad=[p, p, p, p], mode='reflect')
    else:
        raise NotImplementedError("padding mode is only 'SAME', 'VALID', 'SYMMETRIC'. {} is not available!".format(padding))

    # Extract patches
    unfold = torch.nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)

    patches = unfold(X)
    return patches   # (N, C*K*K, L) ,  L is the total number of such blocks


def save_parameter(gen_network_state_dict, dis_network_state_dict, config, epoch):
    # model parameter save
    if epoch is not None:
        torch.save(gen_network_state_dict, config['GEN_PARAM_PATH']+'{}_epoch'.format(epoch+1))
        torch.save(dis_network_state_dict, config['DIS_PARAM_PATH']+'{}_epoch'.format(epoch+1))
        print("current epoch: {}, Parameter saved.".format(epoch+1))
    else:
        torch.save(gen_network_state_dict, config['GEN_PARAM_PATH'])
        torch.save(dis_network_state_dict, config['DIS_PARAM_PATH'])
        print("Parameter saved!!!.")

