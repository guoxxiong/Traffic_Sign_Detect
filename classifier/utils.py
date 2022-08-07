import random

import cv2
import numpy as np
import torch.nn as nn


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'.
    """
    with open(path, "r") as f:
        return [i.strip() for i in f.readlines()]


def weights_init_normal(m):
    """
    Init weights.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def resize(image, size, mode="nearest"):
    """
    Resize input image.
    """
    if mode == "bilinear":
        return nn.functional.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=True).squeeze(0)
    else:
        return nn.functional.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)


def random_reinforcement(img):
    """
    Random input reinforcement. Rotate, noise and brightness.
    """
    img = img.astype(np.float32)

    # rotate
    angle = random.uniform(-0.35, 0.35)
    M = cv2.getRotationMatrix2D((img.shape[0]//2, img.shape[1]//2), angle, 1.)
    img = cv2.warpAffine(img, M, img.shape[1:: -1], cv2.INTER_NEAREST)

    # noise -- not work
    # noise = np.random.randn(*(img.shape)) * 10
    # img += noise

    # rgb channel brightness
    for k in range(3):
        factor = random.uniform(-0.7, 0.7) + 1.0
        img[:, :, k] *= factor

    img[img > 255] = 255
    img[img < 0] = 0
    return img.astype(np.uint8)