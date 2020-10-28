import logging
import random
import numpy as np
import cv2
import copy
import os
import statistics
import matplotlib.pyplot as plt

def svm2Normal(input_image):
    return np.array([unflattenImage(img) for img in input_image])


def unflattenImage(img):
    image3D = img.reshape((50, 50, 3)).transpose()
    fixChannelImg = np.moveaxis(image3D, 0, -1)
    fixMirrorImg = np.moveaxis(fixChannelImg, 0, 1)
    return fixMirrorImg


x_clean = np.load(r'../3-attack/Generated Adversarial Data/clean_cnn_colour.npy')
x_test_n = np.load(r'../3-attack/Generated Adversarial Data/cw_cnn_adv_colour.npy')
x_clean_normal = x_clean
x_test_normal = x_test_n

# x_clean_normal = np.clip(x_clean, 0, 1)
# x_test_normal = np.clip(x_test_n, 0, 1)

allDist = []
count = 0
for cleanImage in x_clean_normal:
    channelLast = np.array(x_test_normal[count]).reshape(48, 48, 3)
    # channelLast = np.moveaxis(x_test_normal[count], 0, -1)
    plt.imshow(channelLast)
    plt.show()
    plt.imshow(cleanImage)
    plt.show()

    dist = np.linalg.norm(channelLast - cleanImage)
    print('Dist is', dist)
    allDist.append(dist)
    count += 1
print('Avg Dist:', statistics.mean(allDist))