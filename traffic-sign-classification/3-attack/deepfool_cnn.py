from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np
import os
import cv2
from art.attacks.evasion import DeepFool
from art.estimators.classification import KerasClassifier
from art.utils import get_file
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import joblib

from argparse import ArgumentParser
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score



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

print(x_train)
im_shape = x_train[0].shape

# model_svc = SVC(kernel='rbf')
# model_svc.fit(x_train,y_train)
# model_predict = model_svc.predict(x_test)
# acc = accuracy_score(y_test,model_predict)


model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Create classifier wrapper
classifier = KerasClassifier(model=model, clip_values=(min_, max_))
classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)

# Craft adversarial samples with DeepFool
logger.info("Create DeepFool attack")
adv_crafter = DeepFool(classifier)
logger.info("Craft attack on training examples")
x_train_adv = adv_crafter.generate(x_train)
logger.info("Craft attack test examples")
x_test_adv = adv_crafter.generate(x_test)

# Evaluate the classifier on the adversarial samples
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == y_test) / len(y_test)
logger.info("Classifier before adversarial training")
logger.info("Accuracy on adversarial samples: %.2f%%", (acc * 100))

# Data augmentation: expand the training set with the adversarial samples
x_train = np.append(x_train, x_train_adv, axis=0)
y_train = np.append(y_train, y_train, axis=0)

# Retrain the CNN on the extended dataset
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)

# Evaluate the adversarially trained classifier on the test set
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == y_test) / len(y_test)
logger.info("Classifier with adversarial training")
logger.info("Accuracy on adversarial samples: %.2f%%", (acc * 100))

