import logging
import random
import numpy as np
import os
import cv2
import copy
import tensorflow as tf

from art.attacks.evasion import CarliniL2Method, UniversalPerturbation, BoundaryAttack
from art.estimators.classification import KerasClassifier, SklearnClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from numpy import save
from PIL import Image


# read in images and class labels for training data
def create_training_data():
    for categories in CATEGORIES:
        path = os.path.join(DATADIR_TRAIN, categories)
        class_num = CATEGORIES.index(categories)
        for img in os.listdir(path):
            try:
                # img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = np.array(Image.open(os.path.join(path, img)).convert('RGB')) # remove the a channel
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

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


def create_features(img):
    color_features = img.flatten()
    flat_features = np.hstack(color_features)
    return flat_features


def transform2Grey(input_image):
    """perform the transformation and return an array"""
    return np.array([create_features(img) for img in input_image])


def convertLabel(labels):
    outputLabels = []
    zeroListTemplate = [0] * (max(labels) + 1)
    for label in labels:
        newList = zeroListTemplate.copy()
        newList[label] = 1
        outputLabels.append(newList)

    return outputLabels


def transformPred(predictions):
    # Convert prediction to just the 10 interested class
    convertedPred = []
    for singlePrediction in predictions:
        for i in range(len(singlePrediction)):
            if singlePrediction[i] == 1:
                convertedPred.append(i)
    return convertedPred

# Set training dataset directory and limiting the numbers to 2 category
DATADIR_TRAIN = os.path.dirname(os.path.split(os.getcwd())[0]) + r"/Final_Data/Train"
DATADIR_TEST = os.path.dirname(os.path.split(os.getcwd())[0]) + r"/Final_Data/Test"
CATEGORIES = ['i2', 'i4', 'i5', 'io', 'p11', 'p26', 'pl5', 'pl30', 'pl40', 'pl50']
IMG_SIZE = 50

tf.compat.v1.disable_eager_execution()

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


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X = X / 255
X_testing = np.array(X_testing).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X_testing = X_testing / 255

x_train = X
y_train = y
x_test = X_testing
y_test = y_testing

min_ = 0
max_ = 1
im_shape = x_train[0].shape

# Step 2: Create the model
model = SVC(C=1.0, kernel="rbf")

# Step 3: Create the ART classifier
classifier = SklearnClassifier(model=model, clip_values=(0, 1))
x_train_n = transform2Grey(x_train)
x_test_n = transform2Grey(x_test)


y_train_n = convertLabel(y_train)
y_test_n = convertLabel(y_test)

model = SVC(C=1.0, kernel="rbf")

# Step 3: Create the ART classifier
print('Creating model')
classifier = SklearnClassifier(model=model, clip_values=(0, 1))
classifier.fit(x_train_n, y_train_n)
print('Creating model done')
x_test_n = np.load(r'../3-attack/Generated Adversarial Data/BIM_svm_adv.npy')
predictions = classifier.predict(x_test_n)
transformedPred = transformPred(predictions)
totalPerClass = {}
for truth in y_test:
    if truth not in totalPerClass:
        totalPerClass[truth] = 1
    else:
        totalPerClass[truth] += 1
correctPerClass = copy.deepcopy(totalPerClass)
detailedError = {}
for i in range(len(y_test)):
    if transformedPred[i] != y_test[i]:
        correctPerClass[y_test[i]] -= 1
        # Record the error details for later computation
        if y_test[i] not in detailedError:
            detailedError[y_test[i]] = {}
            detailedError[y_test[i]][transformedPred[i]] = 1
        else:
            if transformedPred[i] not in detailedError[y_test[i]]:
                detailedError[y_test[i]][transformedPred[i]] = 1
            else:
                detailedError[y_test[i]][transformedPred[i]] += 1
accuracyPerClass = {}
print("SVM's accuracy with CW input:")
for trafficClass in totalPerClass:
    accuracyPerClass[CATEGORIES[trafficClass]] = float(correctPerClass[trafficClass] / totalPerClass[trafficClass])
    print("    ", CATEGORIES[trafficClass], str(accuracyPerClass[CATEGORIES[trafficClass]]* 100) + '%,'
          , correctPerClass[trafficClass], 'out of', totalPerClass[trafficClass])
    totalIncorrect = totalPerClass[trafficClass] - correctPerClass[trafficClass]
    if totalIncorrect != 0:
        print("            Out of", totalIncorrect, "incorrect images:")
        for details in detailedError[trafficClass]:
            incorrectPercentage = str(round(float(detailedError[trafficClass][details] / totalIncorrect), 2) * 100) + '%'
            print("            ", incorrectPercentage, ',', detailedError[trafficClass][details], 'images misclassified as', CATEGORIES[details])
