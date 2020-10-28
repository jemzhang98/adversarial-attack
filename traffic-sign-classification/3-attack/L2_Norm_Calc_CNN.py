import logging
import random
import numpy as np
import cv2
import copy
import os
import statistics
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def svm2Normal(input_image):
    return np.array([unflattenImage(img) for img in input_image])


def unflattenImage(img):
    image3D = img.reshape((50, 50, 3)).transpose()
    fixChannelImg = np.moveaxis(image3D, 0, -1)
    fixMirrorImg = np.moveaxis(fixChannelImg, 0, 1)
    return fixMirrorImg



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


x_clean = np.load(r'../3-attack/Generated Adversarial Data/clean_cnn_colour.npy')
x_test_n = np.load(r'../3-attack/Generated Adversarial Data/cw_cnn_adv_colour.npy')
# x_test_n = np.load(r'../3-attack/Generated Adversarial Data/BIM_cnn_adv.npy')
# x_test_n = np.load(r'../3-attack/Generated Adversarial Data/dpf_cnn_adv_colour.npy')
x_clean_normal = x_clean
x_test_normal = np.moveaxis(x_test_n, 1, -1)

# Transformation to tensor
transform = transforms.Compose(
    [transforms.ToTensor()])

allDist = []
count = 0
tmp = 0
unorm = UnNormalize(mean=(0.440985, 0.390349, 0.438721), std=(0.248148, 0.230837, 0.237781))
for cleanImage in x_clean_normal:
    # channelLast = np.array(x_test_normal[count]).reshape(48, 48, 3)
    img_tensor = unorm(transform(x_test_normal[count]))
    img_rgb = img_tensor.numpy()
    channelLast = np.moveaxis(img_rgb, 0, -1)
    singleImageDis = []
    for c in range(3):
        singleChannelAdv = channelLast[..., c]
        singleChannelClean = cleanImage[..., c]
        dist = np.linalg.norm(singleChannelAdv - singleChannelClean)
        singleImageDis.append(dist)
    dist = statistics.mean(singleImageDis)
    if dist == 0.0:
        tmp += 1
    # if dist > 21:
    #     plt.imshow(channelLast)
    #     plt.show()
    #     plt.imshow(cleanImage)
    #     plt.show()
    #     print('lol')
    print('Dist is', dist)
    allDist.append(dist)
    count += 1
print('Avg Dist:', statistics.mean(allDist))
print(tmp, 'did not change much')
