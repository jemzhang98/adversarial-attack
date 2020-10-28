import os
import numpy as np
import cv2
import torch
import sys
import art
import copy
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from torch import nn
from art.estimators.classification import PyTorchClassifier


# import StnCnn class
CNNFOLDER = os.path.dirname(os.path.split(os.getcwd())[0]) + r"\traffic-sign-classification\2-cnn"
sys.path.insert(1, CNNFOLDER)
from stn_cnn import StnCnn



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
    count = 0
    for img in os.listdir(path):
        try:
            class_name = img.split('_')[0]
            class_num = CATEGORIES.index(class_name)
            currentImg = resizedImgs[count, :]
            testing_data.append([currentImg, class_num])
            count += 1
        except Exception as e:
            pass

    for i in range(len(testing_data)):
        toBeTransformed = Image.fromarray(np.uint8(testing_data[i][0]))
        img_tensor = transform(toBeTransformed)
        img_rgb = img_tensor.numpy()
        testing_data[i][0] = img_rgb

def calc_accuracy(predictions, y_test):
    # Convert prediction to just the 10 interested class
    convertedPred = transformPred(predictions)

    # Calc accuracy of prediction
    correctCount = 0
    for i in range(len(convertedPred)):
        if convertedPred[i] == y_test[i]:
            correctCount += 1
    acc = correctCount / len(convertedPred)
    return acc


def transformPred(predictions):
    # Convert prediction to just the 10 interested class
    convertedPred = []
    for singlePrediction in predictions:
        maxCompare = []
        for location in labelLoc:
            maxCompare.append(singlePrediction[location])
        maxConfidence = max(maxCompare)
        maxIndex = maxCompare.index(maxConfidence)
        convertedPred.append(maxIndex)
    return convertedPred

# Ignore this, this is only needed ot get the y_test labels.
x_test_adv = np.load(r'../3-attack/Generated Adversarial Data/cw_svm_adv_colour.npy')
IMG_SIZE = 48
normalImgs = svm2Normal(x_test_adv)
resizedImgs = loopNpArray(normalImgs)
resizedImgs = resizedImgs * 255

# Convert to CNN input
# The original model was trained with these normalization values, need to use these to ensure to get a similar result
transform = transforms.Compose(
    [transforms.Resize((48, 48)),
     transforms.ToTensor(),
     transforms.Normalize((0.440985, 0.390349, 0.438721), (0.248148, 0.230837, 0.237781))])

testing_data = []
DATADIR_TEST = os.path.dirname(os.path.split(os.getcwd())[0]) + r"/Final_Data/Test"
CATEGORIES = ['i2', 'i4', 'i5', 'io', 'p11', 'p26', 'pl30', 'pl40', 'pl50', 'pl5']
labelLoc = [0, 1, 2, 3, 5, 7, 9, 10, 12, 11]
originalCat = ['i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
              'pl40', 'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57']
create_testing_data()

y_testing = []

for features, label in testing_data:
    y_testing.append(label)

y_test = y_testing

# Load CNN model
checkpointLoc = os.path.dirname(os.path.split(os.getcwd())[0]) + r'\traffic-sign-classification\2-cnn\Train Model\full_dataset_epoch=47.ckpt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
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

x_test_n = np.load(r'../3-attack/Generated Adversarial Data/cw_cnn_adv_colour.npy')
# If the batch size is not set (predict all at once), sometimes it throws an error for CUDA out of memory. Which is why
# i've set a limit
predictions = classifier.predict(x_test_n, batch_size=100)
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
print("CNN's accuracy with CW input:")
for trafficClass in totalPerClass:
    accuracyPerClass[CATEGORIES[trafficClass]] = float(correctPerClass[trafficClass] / totalPerClass[trafficClass])
    print("", CATEGORIES[trafficClass], str(accuracyPerClass[CATEGORIES[trafficClass]]* 100) + '%,'
          , correctPerClass[trafficClass], 'out of', totalPerClass[trafficClass])
    totalIncorrect = totalPerClass[trafficClass] - correctPerClass[trafficClass]
    print("  Out of", totalIncorrect, "incorrect images:")
    for details in detailedError[trafficClass]:
        incorrectPercentage = str(round(float(detailedError[trafficClass][details] / totalIncorrect), 2) * 100) + '%'
        print(" ", incorrectPercentage, ',', detailedError[trafficClass][details], 'images misclassified as', CATEGORIES[details])
