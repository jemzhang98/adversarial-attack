import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd


# As your pre-trained CNN uses colour images, consider training SVM with colored images to make the comparison fair. Suggested ideas, but not limit to: 
# 1. Vectorise the colour images, and then apply PCA. 
# Using cross-validation to learn optimal k components. 
# Train the SVM with k components. 
# 2. Apply a convolutional encoder.

#n_components=0.80 means it will return the Eigenvectors that have the 80% of the variation in the dataset
# faces_pca = PCA(n_components=0.8)
# faces_pca.fit(faces)
# fig, axes = plt.subplots(2,10,figsize=(9,3),
#  subplot_kw={‘xticks’:[], ‘yticks’:[]},
#  gridspec_kw=dict(hspace=0.01, wspace=0.01))
# for i, ax in enumerate(axes.flat):
#  ax.imshow(faces_pca.components_[i].reshape(112,92),cmap=”gray”)

imageArr = []
path = '../../Data/Train/i2/'
for filename in listdir(path):
  filepath = join(path, filename)
  if isfile(filepath):
    print('filepath is', filepath)
    #change all rgba images to rgb
    img = Image.open(filepath).convert('RGB')
    npArr = np.array(img)
    npArr = npArr.reshape(-1, 3) #combines the width/height dimensions to 1 (1D array of all the pixels)
    print(npArr.shape) #height*width, channel (114*70, 3)
    pca = PCA(1)
    pca2 = PCA(2)
    pca3 = PCA(3)
    print(npArr)
    principalComponents = pca.fit_transform(npArr)
    principalComponents2 = pca2.fit_transform(npArr)
    principalComponents3 = pca3.fit_transform(npArr)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1'])
    principalDf2 = pd.DataFrame(data = principalComponents2, columns = ['principal component 1', 'principal component 2'])
    principalDf3 = pd.DataFrame(data = principalComponents3, columns = ['principal component 1', 'principal component 2', 'principal component 3'])
    print(principalDf)
    print(principalDf2)
    print(principalDf3)

    break #test with only 1 pic atm
  
    # print(img.getdata())
    # print(type(im.getdata()))
    # pixel_values = list(img.getdata())
    # print(len(pixel_values))
    # print(pixel_values)
    # imageArr.append(img)
imageArr = np.array(imageArr)
# print(imageArr)
# print(imageArr.data.shape)
# arr = np.array(img) #array only have data
# print(arr.data) #memory@
# print(arr.data.shape) #(29, 28, 4)
# pca = PCA(0.95)
# # pca.fit(arr.data)

# np.min(arr), np,max(arr)


