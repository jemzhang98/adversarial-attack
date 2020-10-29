import numpy as np
import cv2
import torch
import os
import random
import sys
import art
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from art.estimators.classification import PyTorchClassifier
from sklearn.model_selection import train_test_split
from art.attacks.evasion import CarliniL2Method, DeepFool
from torch.autograd import Variable
from torch import nn
from numpy import save
from PIL import Image

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
    # Convert prediction to just the 4 interested class
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

# def reverseSVM():


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

# Read the model value
# checkpointLoc = os.path.dirname(os.path.split(os.getcwd())[0]) + \
#                 r'\traffic-sign-classification\2-cnn\lightning_logs\version_18\checkpoints\epoch=104.ckpt'
checkpointLoc = os.path.dirname(os.path.split(os.getcwd())[0]) + r'\traffic-sign-classification\2-cnn\Train Model\full_dataset_epoch=47.ckpt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: StnCnn = StnCnn.load_from_checkpoint(checkpointLoc)
model.to(device)
model.eval()

# Wrap our PyTorch model with ART's classifier
PyTorchClassifier = art.estimators.classification.PyTorchClassifier
loss = nn.NLLLoss()
classifier = PyTorchClassifier(model=model,
                               loss=loss,
                               input_shape=(3, 48, 48),
                               nb_classes=19)

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

# PyTorch takes in input in shape of [N, Channel, H, W], while the input had channel in the last index
x_test_n = x_test.reshape((x_test.shape[0], x_test.shape[-1], IMG_SIZE, IMG_SIZE))
predictions = classifier.predict(x_test_n)

acc = calc_accuracy(predictions, y_test)
print("Accuracy on benign test examples: {}%".format(acc * 100))

# attack = CarliniL2Method(classifier=classifier, targeted=False)
attack = DeepFool(classifier=classifier)
x_test_adv = attack.generate(x=x_test_n)

# save to npy file
print('Saving generated adv data')
save(r'./Generated Adversarial Data/dpf_cnn_adv_colour.npy', x_test_adv)
# x_test_adv = np.load(r'./Generated Adversarial Data/cw_svm_adv_colour.npy')
# for x_tmp in x_test_adv:
#     tmp = x_tmp.reshape(50, 50, 3).T
#     print('lol')

advPredictions = classifier.predict(x_test_adv)
advAcc = calc_accuracy(advPredictions, y_test)
print("Accuracy for adversarial images: {}%".format(advAcc * 100))

# TODO: figure out how to display the adv image?
