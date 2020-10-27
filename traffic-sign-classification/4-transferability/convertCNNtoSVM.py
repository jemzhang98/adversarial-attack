import os
import numpy as np
import cv2
import torch
import sys
import art
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from torch import nn
from art.estimators.classification import PyTorchClassifier


# import StnCnn class
CNNFOLDER = os.path.dirname(os.path.split(os.getcwd())[0]) + r"\traffic-sign-classification\2-cnn"
sys.path.insert(1, CNNFOLDER)
from stn_cnn import StnCnn



def svm2Normal(input_image):
    return np.array([unflattenImage(img) for img in input_image])


def unflattenImage(img):
    image3D = img.reshape((50, 50, 3)).transpose()
    fixChannelImg = np.moveaxis(image3D, 0, -1)
    fixMirrorImg = np.moveaxis(fixChannelImg, 0, 1)
    return fixMirrorImg

def loopNpArray(input_image):
    return np.array([resizeImg(img) for img in input_image])

def resizeImg(img):
    res = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    return res

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

IMG_SIZE = 48
x_test_adv = np.load(r'../3-attack/Generated Adversarial Data/cw_cnn_adv_colour.npy')
# To make it channel first
x_test_adv = np.array(x_test_adv).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# Transformation to tensor
transform = transforms.Compose(
    [transforms.ToTensor()])

# show or save image
count = 0
saveLocation = r'./cwCNN/'
unorm = UnNormalize(mean=(0.440985, 0.390349, 0.438721), std=(0.248148, 0.230837, 0.237781))

for i in range(x_test_adv.shape[0]):
    currentImage = x_test_adv[count, :]
    img_tensor = unorm(transform(currentImage))
    img_rgb = img_tensor.numpy()
    img_rgb = np.moveaxis(img_rgb, 0, -1)
    # print(count, img_rgb.min(), img_rgb.max())
    # img_rgb = img_rgb * 255
    # im = Image.fromarray(img_rgb.astype(np.uint8))
    # im.save(saveLocation + str(count) + ".png")
    plt.imshow(img_rgb)
    plt.show()
    count += 1
    print(count)

