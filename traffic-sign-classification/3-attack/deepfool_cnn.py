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
from art.attacks.evasion import BoundaryAttack
from torch.autograd import Variable
from torch import nn
from PIL import Image
from numpy import save


# import StnCnn class
CNNFOLDER = os.path.dirname(os.path.split(os.getcwd())[0]) + r"/traffic-sign-classification/2-cnn"
sys.path.insert(1, CNNFOLDER)
from stn_cnn import StnCnn



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
    for i in range(len(training_data)):
        toBeTransformed = Image.fromarray(np.uint8(training_data[i][0])).convert('RGB')
        img_tensor = transform(toBeTransformed)
        img_rgb = img_tensor.numpy()
        training_data[i][0] = img_rgb



#read in images and class labels for test data
def create_testing_data():
    path = DATADIR_TEST
    # i = 0
    for img in os.listdir(path):
    # while i < 10:
        try:
            class_name = img.split('_')[0]
            class_num = CATEGORIES.index(class_name)
            # img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_array = np.array(Image.open(os.path.join(path, img)).convert('RGB')) # remove the a channel
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            testing_data.append([new_array, class_num])
            # i += 1
        except Exception as e:
            pass

    for i in range(len(testing_data)):
        toBeTransformed = Image.fromarray(np.uint8(testing_data[i][0])).convert('RGB')
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

# Set training dataset directory and limiting the numbers to 2 category
DATADIR_TRAIN = os.path.dirname(os.path.split(os.getcwd())[0]) + r"/Final_Data/Train"
DATADIR_TEST = os.path.dirname(os.path.split(os.getcwd())[0]) + r"/Final_Data/Test"
CATEGORIES = ['i2', 'i4', 'i5', 'io', 'p11', 'p26', 'pl30', 'pl40', 'pl5', 'pl50']
labelLoc = [0, 1, 2, 3, 5, 7, 9, 10, 11, 12]

IMG_SIZE = 48

# Read the model value
checkpointLoc = os.path.dirname(os.path.split(os.getcwd())[0]) + r'/traffic-sign-classification/2-cnn/lightning_logs/version_0/checkpoints/epoch=47.ckpt'
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


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X_testing = np.array(X_testing).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

x_train = X
y_train = y
x_test = X_testing
y_test = y_testing



# Since we're using pretrained mode, we really don't need to do the test/train split lol
x_test = x_test.astype('float32')
# PyTorch takes in input in shape of [N, Channel, H, W], while the input had channel in the last index
x_test_n = x_test.reshape((x_test.shape[0], x_test.shape[-1], IMG_SIZE, IMG_SIZE))
predictions = classifier.predict(x_test_n)

acc = calc_accuracy(predictions, y_test)
print("Accuracy on benign test examples: {}%".format(acc * 100))

# attack = CarliniL2Method(classifier=classifier, targeted=False)
attack = DeepFool(classifier)
x_test_adv = attack.generate(x=x_test_n)

save(r'./Generated Adversarial Data/deepfool_cnn_adv.npy', x_test_adv)


advPredictions = classifier.predict(x_test_adv)
advAcc = calc_accuracy(advPredictions, y_test)
print("Accuracy for adversarial images: {}%".format(advAcc * 100))

