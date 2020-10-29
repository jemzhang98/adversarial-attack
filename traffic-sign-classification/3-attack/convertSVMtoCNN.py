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
# CNNFOLDER = os.path.dirname(os.path.split(os.getcwd())[0]) + r"\traffic-sign-classification\2-cnn"
# sys.path.insert(1, CNNFOLDER)
# from stn_cnn import StnCnn



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

#read in images and class labels for test data



x_test_adv = np.load(r'../3-attack/Generated Adversarial Data/boundary_svm_adv.npy')
IMG_SIZE = 48
normalImgs = svm2Normal(x_test_adv)
resizedImgs = loopNpArray(normalImgs)
resizedImgs = resizedImgs * 255

# show or save image
count = 0
saveLocation = r'./boundarySVM/'
for img in resizedImgs:
    im = Image.fromarray(img.astype(np.uint8))
    im.save(saveLocation + str(count) + ".png")
    count += 1