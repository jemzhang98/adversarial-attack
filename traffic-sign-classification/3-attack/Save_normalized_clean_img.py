import numpy as np
import cv2
import torch
import os
import random
import sys
import art
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from art.estimators.classification import PyTorchClassifier
from torch import nn
from PIL import Image
from numpy import save
# import StnCnn class
CNNFOLDER = os.path.dirname(os.path.split(os.getcwd())[0]) + r"\traffic-sign-classification\2-cnn"
sys.path.insert(1, CNNFOLDER)
from stn_cnn import StnCnn


def create_training_data():
    for categories in CATEGORIES:
        path = os.path.join(DATADIR, categories)
        class_num = CATEGORIES.index(categories)
        for img in os.listdir(path):
            try:
                img_bgr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                # cv2 reads in image in BGR format, need to convert to RGB
                img_array = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
    # This is to transform and normalize the image data points, which would be in tensor format.
    # And we converted it back to ndarray because this ndarray <-> tensor is done internally in PyT classifer
    for i in range(len(training_data)):
        toBeTransformed = Image.fromarray(np.uint8(training_data[i][0])).convert('RGB')
        img_tensor = transform(toBeTransformed)
        img_rgb = img_tensor.numpy()
        training_data[i][0] = img_rgb

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

    for i in range(len(testing_data)):
        toBeTransformed = Image.fromarray(np.uint8(testing_data[i][0]))
        img_tensor = transform(toBeTransformed)
        img_rgb = img_tensor.numpy()
        testing_data[i][0] = img_rgb

def calc_accuracy(predictions, y_test):
    # Convert prediction to just the 10 interested class
    convertedPred = []
    for singlePrediction in predictions:
        maxCompare = []
        for location in labelLoc:
            maxCompare.append(singlePrediction[location])
        maxConfidence = max(maxCompare)
        maxIndex = maxCompare.index(maxConfidence)
        convertedPred.append(maxIndex)

    # Calc accuracy of prediction
    correctCount = 0
    for i in range(len(convertedPred)):
        if convertedPred[i] == y_test[i]:
            correctCount += 1
    acc = correctCount / len(convertedPred)
    return acc


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


# Set training dataset directory and limiting the numbers to 2 category
# DATADIR = os.path.dirname(os.path.split(os.getcwd())[0]) + r"\Data\Train"
DATADIR = os.path.dirname(os.path.split(os.getcwd())[0]) + r"/Final_Data/Train"
DATADIR_TEST = os.path.dirname(os.path.split(os.getcwd())[0]) + r"/Final_Data/Test"
CATEGORIES = ['i2', 'i4', 'i5', 'io', 'p11', 'p26', 'pl30', 'pl40', 'pl5', 'pl50']
# Need to update labelLoc to the index of the labels you're interested in
labelLoc = [0, 1, 2, 3, 5, 7, 9, 10, 11, 12]
# CATEGORIES = ['i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
#               'pl40', 'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57']

IMG_SIZE = 48

# The original model was trained with these normalization values, need to use these to ensure to get a similar result
transform = transforms.Compose(
    [transforms.Resize((48, 48)),
     transforms.ToTensor(),
     transforms.Normalize((0.440985, 0.390349, 0.438721), (0.248148, 0.230837, 0.237781))])

# Note: the script randomly shuffles the data, so the accuracy may fluctuate a bit
training_data = []
testing_data = []
create_training_data()
create_testing_data()
random.shuffle(training_data)

X = []
y = []
X_testing = []
y_testing = []

for features, label in training_data:
    X.append(features)
    y.append(label)

for features, label in testing_data:
    X_testing.append(features)
    y_testing.append(label)


# Since we're using pretrained mode, we really don't need to do the test/train split lol
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X_testing = np.array(X_testing).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

x_train = X
y_train = y
x_test = X_testing
y_test = y_testing

# Transformation to tensor
toTensor = transforms.Compose(
    [transforms.ToTensor()])

# show or save image
count = 0
# saveLocation = r'./cwCNN/'
unorm = UnNormalize(mean=(0.440985, 0.390349, 0.438721), std=(0.248148, 0.230837, 0.237781))
for i in range(x_test.shape[0]):
    currentImage = x_test[count, :]
    img_tensor = unorm(toTensor(currentImage))
    img_rgb = img_tensor.numpy()
    img_rgb = np.moveaxis(img_rgb, 0, -1)
    # print(count, img_rgb.min(), img_rgb.max())
    # img_rgb = img_rgb * 255
    # im = Image.fromarray(img_rgb.astype(np.uint8))
    # im.save(saveLocation + str(count) + ".png")
    if count == 490:
        plt.imshow(img_rgb)
        plt.show()
    x_test[count] = img_rgb
    count += 1
    print(count)


# save to npy file
print('Saving normalized clean img for cnn')
save(r'./Generated Adversarial Data/clean_cnn_colour.npy', x_test)