from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np
import os
import cv2
import art
from art.attacks.evasion import DeepFool
from art.estimators.classification import KerasClassifier
from art.utils import get_file
import tensorflow as tf
import matplotlib.pyplot as plt

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



tf.compat.v1.disable_eager_execution()

# Configure a logger to capture ART outputs; these are printed in console and the level of detail is set to INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Read CIFAR10 dataset

DATADIR = "Testimage"
CATEGORIES = ["i2","i4"]

for categories in CATEGORIES:
    path = os.path.join(DATADIR,categories)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap = "gray")
        plt.show()
        break
    break

training_data=[]
IMG_SIZE = 50

def create_training_data():
    for categories in CATEGORIES:
        path = os.path.join(DATADIR, categories)
        class_num = CATEGORIES.index(categories)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass

create_training_data()

import random

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE, 1)
X = X/255


x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.10)
print('x_test',x_test)
print('y_test',y_test)
min_ = 0
max_ = 1
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

classifier = KerasClassifier(model=model, clip_values=(min_, max_))
classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)

logger.info("Create FFSM attack")
FGSM = art.attacks.evasion.FastGradientMethod
adv_crafter = FGSM(estimator=classifier,eps=0.2)
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























