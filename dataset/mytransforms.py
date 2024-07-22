import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import pdb
import cv2
from torchvision import transforms


class Compose2(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask
        for t in self.transforms:
            img, mask, bbx = t(img, mask, bbx)
        return img, mask, bbx


class FreeScale(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, img, mask):
        return img.resize((self.size[1], self.size[0]), Image.BILINEAR), mask.resize((self.size[1], self.size[0]),
                                                                                     Image.NEAREST)


class FreeScaleMask(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, mask):
        return mask.resize((self.size[1], self.size[0]), Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        if img.size != mask.size:
            print(img.size)
            print(mask.size)
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomRotate(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image, label):
        assert label is None or image.size == label.size

        angle = random.randint(0, self.angle * 2) - self.angle

        label = label.rotate(angle, resample=Image.NEAREST)
        image = image.rotate(angle, resample=Image.BILINEAR)

        return image, label


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class RandomLROffsetLABEL(object):
    def __init__(self, max_offset):
        self.max_offset = max_offset

    def __call__(self, img, label):
        offset = np.random.randint(-self.max_offset, self.max_offset)
        w, h = img.size

        img = np.array(img)
        if offset > 0:
            img[:, offset:, :] = img[:, 0:w - offset, :]
            img[:, :offset, :] = 0
        if offset < 0:
            real_offset = -offset
            img[:, 0:w - real_offset, :] = img[:, real_offset:, :]
            img[:, w - real_offset:, :] = 0

        label = np.array(label)
        if offset > 0:
            label[:, offset:] = label[:, 0:w - offset]
            label[:, :offset] = 0
        if offset < 0:
            offset = -offset
            label[:, 0:w - offset] = label[:, offset:]
            label[:, w - offset:] = 0
        return Image.fromarray(img), Image.fromarray(label)


class RandomUDoffsetLABEL(object):
    def __init__(self, max_offset):
        self.max_offset = max_offset

    def __call__(self, img, label):
        offset = np.random.randint(-self.max_offset, self.max_offset)
        w, h = img.size

        img = np.array(img)
        if offset > 0:
            img[offset:, :, :] = img[0:h - offset, :, :]
            img[:offset, :, :] = 0
        if offset < 0:
            real_offset = -offset
            img[0:h - real_offset, :, :] = img[real_offset:, :, :]
            img[h - real_offset:, :, :] = 0

        label = np.array(label)
        if offset > 0:
            label[offset:, :] = label[0:h - offset, :]
            label[:offset, :] = 0
        if offset < 0:
            offset = -offset
            label[0:h - offset, :] = label[offset:, :]
            label[h - offset:, :] = 0
        return Image.fromarray(img), Image.fromarray(label)


class RandomScaleAndPad(object):
    def __init__(self, target_size=(1640, 590), scale_range=(0.5, 1.0), max_offset=10):
        self.target_size = target_size
        self.scale_range = scale_range
        self.max_offset = max_offset
        self.original_size = None  # 保存原始图像大小

    def __call__(self, img, label):
        width, height = img.size

        # 保存原始图像大小
        self.original_size = (width, height)

        # 随机生成缩放比例
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])

        # 计算缩放后的图像大小
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # 对图像进行缩放
        img = img.resize((new_width, new_height), Image.BILINEAR)

        # 创建新的填充图像和标签
        padded_img = Image.new('RGB', self.target_size, (0, 0, 0))
        padded_label = Image.new('L', self.target_size, 0)  # 创建单通道灰度图像作为标签

        # 将调整后的图像放置在填充图像中心
        x_offset = (self.target_size[0] - new_width) // 2
        y_offset = (self.target_size[1] - new_height) // 2 + random.randint(-self.max_offset, self.max_offset)
        padded_img.paste(img, (x_offset, y_offset))

        # 对标签进行相同的缩放操作
        label = label.resize((new_width, new_height), Image.NEAREST)

        # 将调整后的标签填充在填充标签中心
        padded_label.paste(label, (x_offset, y_offset))

        # 将标签调整回原始尺寸
        padded_label = padded_label.resize(self.original_size, Image.NEAREST)

        return padded_img, padded_label



def hflip(img, mask, p=0.5):
    if random.random() < p:
        img_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
        ])
        img = img_transform(img)
        mask = img_transform(mask)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        img_transform = transforms.Compose([
            transforms.GaussianBlur(kernel_size=3),
        ])
        img = img_transform(img)
    return img


def color_jitter(img, p=0.5):
    if random.random() < p:
        color_jitter_transform = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)
        img = color_jitter_transform(img)
    return img


def random_grayscale(img, p=0.5):
    if random.random() < p:
        grayscale_transform = transforms.RandomGrayscale(p=1.0)
        img = grayscale_transform(img)
    return img
