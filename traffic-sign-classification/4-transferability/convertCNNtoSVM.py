import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import random
import numpy as np
import os
import cv2
import tensorflow as tf

from art.estimators.classification import KerasClassifier, SklearnClassifier
from sklearn.svm import SVC
from PIL import Image

# import StnCnn class
CNNFOLDER = os.path.dirname(os.path.split(os.getcwd())[0]) + r"\traffic-sign-classification\2-cnn"
sys.path.insert(1, CNNFOLDER)
from stn_cnn import StnCnn



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
    count = 0
    for img in os.listdir(path):
        try:
            class_name = img.split('_')[0]
            class_num = CATEGORIES.index(class_name)
            testing_data.append([x_test_adv_new[count], class_num])
            count += 1
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

IMG_SIZE = 48
# x_test_adv = np.load(r'../3-attack/Generated Adversarial Data/cw_cnn_adv_colour.npy')
x_test_adv = np.load(r'../3-attack/Generated Adversarial Data/boundary_cnn_adv.npy')
# To make it channel first
# x_test_adv = np.array(x_test_adv).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
x_test_adv = np.moveaxis(x_test_adv, 1, -1)

# Transformation to tensor
transform = transforms.Compose(
    [transforms.ToTensor()])

# show or save image
count = 0
saveLocation = r'./cwCNN/'
unorm = UnNormalize(mean=(0.440985, 0.390349, 0.438721), std=(0.248148, 0.230837, 0.237781))

x_test_adv_new = np.zeros((500, 50, 50, 3))
for i in range(x_test_adv.shape[0]):
    currentImage = x_test_adv[count, :]
    img_tensor = unorm(transform(currentImage))
    img_rgb = img_tensor.numpy()
    img_rgb = np.moveaxis(img_rgb, 0, -1)
    img_rgb = cv2.resize(img_rgb, dsize=(50, 50), interpolation=cv2.INTER_CUBIC)
    x_test_adv_new[count] = img_rgb * 255
    # print(count, img_rgb.min(), img_rgb.max())
    # img_rgb = img_rgb * 255
    # im = Image.fromarray(img_rgb.astype(np.uint8))
    # im.save(saveLocation + str(count) + ".png")
    # plt.imshow(img_rgb)
    # plt.show()
    count += 1
    # print(count)

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

# save to npy file
# print('Saving testing data')
# save(r'./Generated Adversarial Data/svm_testing_data.npy', x_test_n)

y_train_n = convertLabel(y_train)
y_test_n = convertLabel(y_test)
# Step 4: Train the ART classifier
print('Training model')
classifier.fit(x_train_n, y_train_n)
predictions = classifier.predict(x_test_n)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_n, axis=1)) / len(y_test_n)

print("Accuracy on benign test examples: {}%".format(accuracy * 100))