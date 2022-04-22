import numpy as np
import random
import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
from PIL import Image


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, kpts=None, mask=None, amodal_mask=None):
        for t in self.transforms:
            img, kpts, mask, amodal_mask = t(img, kpts, mask, amodal_mask)
        return img, kpts, mask, amodal_mask

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):

    def __call__(self, img, kpts, mask, amodal_mask):
        return np.asarray(img).astype(np.float32) / 255., kpts, mask, amodal_mask


class Normalize(object):

    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, img, kpts, mask, amodal_mask):
        img -= self.mean
        img /= self.std
        if self.to_bgr:
            img = img.transpose(2, 0, 1).astype(np.float32)
        return img, kpts, mask, amodal_mask


class ColorJitter(object):

    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, kpts, mask, amodal_mask):
        image = np.asarray(self.color_jitter(Image.fromarray(np.ascontiguousarray(image, np.uint8))))
        return image, kpts, mask, amodal_mask


class RandomBlur(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, kpts, mask, amodal_mask):
        if random.random() < self.prob:
            sigma = np.random.choice([3, 5, 7, 9])
            image = cv2.GaussianBlur(image, (sigma, sigma), 0)
        return image, kpts, mask, amodal_mask


def make_transforms(cfg, is_train):
    if is_train is True:
        transform = Compose(
            [
                RandomBlur(0.5),
                ColorJitter(0.1, 0.1, 0.05, 0.05),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transform
