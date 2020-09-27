import cv2
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn import preprocessing


img: np.ndarray = cv2.imread('../../Data/Train/i2/i2_0000.png')
img = cv2.resize(img, (28, 28))
npArr = np.array(img)
print(npArr.shape)
print(len(npArr))
reshaped = npArr.reshape(-1, 3) #combines the width/height dimensions to 1 (1D array of all the pixels) (114*70, 3)
# print(reshaped)
data = pd.DataFrame(data=reshaped, columns = ['b', 'g', 'r'])
# print(data.head())

#scale the data (to 0?)
scaled_data = preprocessing.scale(data)

# apply pca - works for rgb or bgr
pca = PCA(1)
principalComponents = pca.fit_transform(scaled_data)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1'])
# print(principalDf)
print(np.array(principalDf).reshape(len(npArr), len(npArr)))

img_gray1: np.ndarray = cv2.cvtColor(cv2.resize(img, (28, 28)), cv2.COLOR_BGR2GRAY)
print(img_gray1.shape)
img_gray2: np.ndarray = cv2.cvtColor(cv2.resize(img, (40, 40)), cv2.COLOR_BGR2GRAY)



