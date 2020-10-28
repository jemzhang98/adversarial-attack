from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import random
import numpy as np
import copy
import os
import statistics
import cv2
import tensorflow as tf

from art.attacks.evasion import CarliniL2Method, UniversalPerturbation, BoundaryAttack
from art.estimators.classification import KerasClassifier, SklearnClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from numpy import save
from PIL import Image


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
def create_testing_data():
    path = DATADIR_TEST
    for img in os.listdir(path):
        try:
            class_name = img.split('_')[0]
            class_num = CATEGORIES.index(class_name)
            # img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_array = np.array(Image.open(os.path.join(path, img)).convert('RGB')) # remove the a channel
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            testing_data.append([new_array, class_num])
        except Exception as e:
            pass


def transform2Grey(input_image):
    """perform the transformation and return an array"""
    return np.array([create_features(img) for img in input_image])


def create_features(img):
    color_features = img.flatten()
    flat_features = np.hstack(color_features)
    return flat_features


def convertLabel(labels):
    outputLabels = []
    zeroListTemplate = [0] * (max(labels) + 1)
    for label in labels:
        newList = zeroListTemplate.copy()
        newList[label] = 1
        outputLabels.append(newList)

    return outputLabels

# Set training dataset directory and limiting the numbers to 2 category
DATADIR_TRAIN = os.path.dirname(os.path.split(os.getcwd())[0]) + r"/Final_Data/Train"
DATADIR_TEST = os.path.dirname(os.path.split(os.getcwd())[0]) + r"/Final_Data/Test"
CATEGORIES = ['i2', 'i4', 'i5', 'io', 'p11', 'p26', 'pl5', 'pl30', 'pl40', 'pl50']
IMG_SIZE = 50

tf.compat.v1.disable_eager_execution()

testing_data = []
create_testing_data()


X_testing = []
y_testing = []

for features, label in testing_data:
    X_testing.append(features)
    y_testing.append(label)


X_testing = np.array(X_testing).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
# X_testing = X_testing / 255

x_test = X_testing
y_test = y_testing

x_test_n = np.load(r'../3-attack/Generated Adversarial Data/cw_svm_adv_colour.npy')
normalImgs = svm2Normal(x_test_n)
resizedImgs = loopNpArray(normalImgs)
resizedImgs = resizedImgs * 255

allDist = []
count = 0
for cleanImage in x_test:
    dist = np.linalg.norm(resizedImgs[count] - cleanImage)
    print('Dist is', dist)
    allDist.append(dist)
    count += 1
print('Avg Dist:', statistics.mean(allDist))