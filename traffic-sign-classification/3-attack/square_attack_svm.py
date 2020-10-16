from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import random
import numpy as np
import os
import cv2
import joblib

from art.attacks.evasion import DeepFool, UniversalPerturbation, SquareAttack
from art.estimators.classification import KerasClassifier, SklearnClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from numpy import save


def create_training_data():
    for categories in CATEGORIES:
        path = os.path.join(DATADIR, categories)
        class_num = CATEGORIES.index(categories)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


def transform2Grey(input_image):
    """perform the transformation and return an array"""
    return np.array([create_features(img) for img in input_image])


def create_features(img):
    # flatten three channel color image
    color_features = img.flatten()
    # combine color and hog features into a single array
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
DATADIR = os.path.dirname(os.path.split(os.getcwd())[0]) + r"\Data\Train"
CATEGORIES = ['i4', 'pl30', 'pl80', 'w57']
IMG_SIZE = 50

training_data = []
create_training_data()
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X = X.astype('float64')
# X = X / 255

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
min_ = 0
max_ = 255
im_shape = x_train[0].shape

# Step 2: Create the model
model = SVC(C=1.0, kernel="rbf")

# Step 3: Create the ART classifier
classifier = SklearnClassifier(model=model, clip_values=(min_, max_))
x_train_n = transform2Grey(x_train)
x_test_n = transform2Grey(x_test)

y_train_n = convertLabel(y_train)
y_test_n = convertLabel(y_test)

# save to npy file
print('Saving testing and training data')
save(r'./Data/Test/svm_reduced_testing_data.npy', x_test_n)
save(r'./Data/Test/svm_reduced_testing_label_data.npy', y_test_n)
save(r'./Data/Train/svm_reduced_training_data.npy', x_train_n)
save(r'./Data/Train/svm_reduced_training_label_data.npy', y_train_n)

print('Training model...')
classifier.fit(x_train_n, y_train_n)
predictions = classifier.predict(x_test_n)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_n, axis=1)) / len(y_test_n)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))
print('Saving model...')
joblib.dump(classifier, os.path.join(r'./Model', 'svm_color.model'))

print('Generating attack...')
attack = SquareAttack(estimator=classifier, eps=0.2, max_iter=1000)
x_test_adv = attack.generate(x=x_test)

# save to npy file
print('Saving generated adv data')
save(r'./Generated Adversarial Data/square_svm_color_adv.npy', x_test_adv)

# Step 7: Evaluate the ART classifier on adversarial test examples
x_test_adv_flat = transform2Grey(x_test_adv)
predictions = classifier.predict(x_test_adv_flat)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_n, axis=1)) / len(y_test_n)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

