from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import random
import numpy as np
import os
import cv2
import tensorflow as tf

from art.attacks.evasion import DeepFool, UniversalPerturbation
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
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
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

tf.compat.v1.disable_eager_execution()

# Configure a logger to capture ART outputs; these are printed in console and the level of detail is set to INFO
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# formatter = logging.Formatter("[%(levelname)s] %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)

# for categories in CATEGORIES:
#     path = os.path.join(DATADIR, categories)
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#         # plt.imshow(img_array, cmap="gray")
#         # plt.show()
#         break
#     break

training_data = []
create_training_data()
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X / 255

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
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
# Step 4: Train the ART classifier
classifier.fit(x_train_n, y_train_n)
predictions = classifier.predict(x_test_n)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_n, axis=1)) / len(y_test)

print("Accuracy on benign test examples: {}%".format(accuracy * 100))
# Step 6: Generate adversarial test examples
# attack = FastGradientMethod(estimator=classi  ssssfier, eps=0.2)
attack = UniversalPerturbation(classifier=classifier, attacker='fgsm')
x_test_adv = attack.generate(x=x_test_n)

# save to npy file
print('Saving generated adv data')
save(r'./Generated Adversarial Data/uni_fgsm_svm_adv.npy', x_test_adv)

# Step 7: Evaluate the ART classifier on adversarial test examples
x_test_adv_flat = transform2Grey(x_test_adv)
predictions = classifier.predict(x_test_adv_flat)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_n, axis=1)) / len(y_test_n)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

