from __future__ import absolute_import, division, print_function, unicode_literals


import random
import numpy as np
import copy
import os
import cv2
import tensorflow as tf

from art.attacks.evasion import CarliniL2Method
from art.estimators.classification import KerasClassifier, SklearnClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from numpy import save
from PIL import Image
from sklearn.externals import joblib
from pathlib import Path


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


svmModelLoc = r'./Model/trained_svm.model'
advDataLoc = r'./Generated Adversarial Data/cw_svm_adv_colour.npy'

# Only train the model if the model cannot be found

# Set training dataset directory and limiting the numbers to 2 category
print('Loading images...')
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

x_train_n = transform2Grey(x_train)
x_test_n = transform2Grey(x_test)

# save to npy file
# print('Saving testing data')
# save(r'./Generated Adversarial Data/svm_testing_data.npy', x_test_n)

y_train_n = convertLabel(y_train)
y_test_n = convertLabel(y_test)

if not (Path(svmModelLoc).is_file()):
    classifier = SklearnClassifier(model=model, clip_values=(0, 1))
    classifier.fit(x_train_n, y_train_n)

    # save to trained SVM model
    print('Saving SVM model...')
    joblib.dump(classifier, svmModelLoc)
else:
    print('Loading trained SVM model...')
    classifier = joblib.load(svmModelLoc)

print('Predicting image classes...')
predictions = classifier.predict(x_test_n)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_n, axis=1)) / len(y_test_n)

incorrectIndex = []
for i in range(len(y_test_n)):
    if not (predictions[i] == y_test_n[i]).all():
        incorrectIndex.append(i)

print("Original Accuracy: {}%".format(accuracy * 100))

regenerateAtk = False
if (regenerateAtk and Path(advDataLoc).is_file()):
    attack = CarliniL2Method(classifier=classifier, targeted=False)

    x_test_adv = attack.generate(x=x_test_n, targeted=False)

    # save to npy file
    print('Saving generated adv data')
    save(r'./Generated Adversarial Data/cw_svm_adv_colour.npy', x_test_adv)
else:
    print('Loading adversarial output...')
    x_test_adv = np.load(advDataLoc)

x_test_adv_flat = transform2Grey(x_test_adv)
predictions = classifier.predict(x_test_adv_flat)

advAccuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_n, axis=1)) / len(y_test_n)
print("Raw Adv Accuracy: {}%".format(advAccuracy * 100))

# Adjust the adv accuracy calc so that it excludes ones that were misclassified to start with
correctCount = 0
for i in range(len(predictions)):
    if i not in incorrectIndex and (predictions[i] == y_test_n[i]).all():
        correctCount += 1
adjAccuracy = correctCount / (len(y_test_n) - len(incorrectIndex))
print("Adjusted Adv Accuracy is: {}%".format(adjAccuracy * 100))
print("The adv images has decreased the accuracy by: {}%".format((1 - adjAccuracy) * 100))

