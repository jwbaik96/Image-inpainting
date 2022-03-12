from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import cv2
import numpy as np


class CelebaDataset(Dataset):
    """
    Celeba Dataset processing for image inpainting
    """
    def __init__(self, config):
        # Data folder directory
        self.folder_dir = config['celeba_data_folder_dir']

        # Image directory
        file_list = os.listdir(self.folder_dir)

        # Save all images directory
        self.imgs_dir = []
        for img_name in file_list:
            self.imgs_dir.append(self.folder_dir + '/' + img_name)

    def __len__(self):
        # return total number of images
        return len(self.imgs_dir)

    def __getitem__(self, index):
        rotation_degree = np.random.randint(60)
        affine_degree = np.random.randint(15)
        transform = transforms.Compose([transforms.RandomRotation(degrees=rotation_degree), transforms.RandomAffine(degrees=affine_degree), transforms.ToTensor()])
        # transform = transforms.Compose([transforms.ToTensor()])

        img = transform(self.read_image(self.imgs_dir[index]))
        return img

    def read_image(self, path):
        # Read image from path
        img = Image.open(path).convert('RGB')
        return img


class PlaceDataset(Dataset):
    def __init__(self, config):
        self.config = config

        # Data list txt
        txt_file = config['place_data_list_txt'] + 'train.txt'

        with open(txt_file, 'r', encoding='utf8') as f:
            self.data_list = f.read().split('\n')

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.transform(self.read_image(index))

    def read_image(self, index):
        img_dir = self.config['place_data_list_txt'] + self.data_list[index]
        img = Image.open(img_dir).convert('RGB')
        return img


class GenerateMask(object):
    """
    Generate one free-form or bbox mask randomly

    :param random_bbox_shape: rectangular bax shape
    :param random_bbox_margin: Least distance between bboxes
    :param random_bbox_number: bbox number in one mask
    :param random_ff_setting: setting for generating free form mask
                        config -> {'img_shape', 'mv', 'ma', 'ml', 'mbw'}
    """
    def __init__(self, config):
        self.config = config
        self.transform = transforms.ToTensor()

    def random_ff_mask(self):
        """Generate a random free form mask with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

            default config : {'img_shape': [256, 256], 'mv': 5, 'ma': 4.0, 'ml': 40, 'mbw': 10}
        Returns:
            tuple: (top, left, height, width)
        """
        config = self.config['random_ff_setting']
        h, w = config['img_shape']
        mv = np.random.randint(config['mv']) + 1
        ma = np.random.randint(config['ma']) + 1
        ml = np.random.randint(config['ml']) + 1
        mbw = np.random.randint(config['mbw']) + 1

        mask = np.zeros((h, w))
        num_v = 12 + np.random.randint(mv)

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(ma)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(ml)
                brush_w = 10 + np.random.randint(mbw)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y

        return mask.reshape(mask.shape + (1,)).astype(np.float32)

    def random_bbox(self):
        """Generate a random tlhw with configuration.
        Returns:
            tuple: (top, left, height, width)
        """
        img_height, img_width = self.config['img_shape']  # image shape
        max_mask_number = self.config['max_mask_number']
        max_mask_shape = self.config['max_mask_shape']
        margin_height, margin_width = self.config['min_margin']  # margin

        # if mask size is big, generate small number of masks
        threshold_low = self.config['threshold_low']
        threshold_medium = self.config['threshold_medium']
        threshold_high = self.config['threshold_high']

        bbox_list = []
        index = 0
        while index < max_mask_number:
            h, w = np.random.randint(5, max_mask_shape, size=2)

            threshold = h + w
            if threshold > threshold_high:
                index = max_mask_number
            elif threshold > threshold_medium:
                index += max_mask_number // 2
            elif threshold > threshold_low:
                index += max_mask_number // 4

            maxt = img_height - margin_height - h
            maxl = img_width - margin_width - w

            t = np.random.randint(margin_height, maxt)
            l = np.random.randint(margin_width, maxl)
            bbox_list.append((t, l, h, w))

        return torch.tensor(bbox_list, dtype=torch.int64)

    def bbox2mask(self, bboxes):
        img_height, img_width = self.config['img_shape']  # image shape
        mask = torch.zeros((1, 1, img_height, img_width), dtype=torch.float32)

        max_delta_h, max_delta_w = np.random.randint(1, self.config['max_delta_hw'][0]), np.random.randint(1, self.config['max_delta_hw'][1])

        num_bboxes = bboxes.size(0)
        for i in range(num_bboxes):
            bbox = bboxes[i]
            delta_h = np.random.randint(max_delta_h // 2 + 1)
            delta_w = np.random.randint(max_delta_w // 2 + 1)
            mask[0, 0, bbox[0] + delta_h:bbox[0] + bbox[2] - delta_h, bbox[1] + delta_w:bbox[1] + bbox[3] - delta_w] = 1.

        # mask size -> (1, 1, 256, 256)
        return mask

    def generate(self, batch_size, mask_type='FF'):
        """
        Generate mask
        :param batch_size:  batch_size
        :param mask_type:  'FF' or 'bbox'
        :return: mask Tensor
        """
        H, W = self.config['img_shape']

        expand_matrix = torch.ones(size=(batch_size, 1))

        random_select = np.random.randint(100)
        if random_select < 50:
            mask = self.transform(self.random_ff_mask()).view((1, -1))   # (1, 256*256)
        else:
            bboxes = self.random_bbox()
            mask = self.bbox2mask(bboxes).view((1, -1))  # (1, 256*256)

        mask = torch.matmul(expand_matrix, mask) # (batch_size, 256*256)
        mask = mask.view((batch_size, 1, H, W))  # (batch_size, 1, 256, 256)
        return mask.contiguous()

