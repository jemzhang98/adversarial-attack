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
from torch.autograd import Variable
from torch import nn
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



# Set training dataset directory and limiting the numbers to 2 category
# DATADIR = os.path.dirname(os.path.split(os.getcwd())[0]) + r"\Data\Train"
DATADIR = os.path.dirname(os.path.split(os.getcwd())[0]) + r"\Augmented_Data\Train"
CATEGORIES = ['i4', 'pl30', 'pl80', 'w57']
# Need to update labelLoc to the index of the labels you're interested in
labelLoc = [1, 9, 14, 18]
# CATEGORIES = ['i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30',
#               'pl40', 'pl5', 'pl50', 'pl60', 'pl80', 'pn', 'pne', 'po', 'w57']

IMG_SIZE = 48

# Read the model value
checkpointLoc = r'C:\Users\jungc\Desktop\epoch=47.ckpt'
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
create_training_data()
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

# Since we're using pretrained mode, we really don't need to do the test/train split lol
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
x_test = x_test.astype('float32')
# PyTorch takes in input in shape of [N, Channel, H, W], while the input had channel in the last index
x_test_n = x_test.reshape((x_test.shape[0], x_test.shape[-1], IMG_SIZE, IMG_SIZE))
predictions = classifier.predict(x_test_n)

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
print("Accuracy on benign test examples: {}%".format(acc * 100))

# TODO: figure out how to display the adv image?

print('lol')


