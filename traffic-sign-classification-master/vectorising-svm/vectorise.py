import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import cv2
import statistics

""" 
# As your pre-trained CNN uses colour images, consider training SVM with colored images to make the comparison fair. Suggested ideas, but not limit to: 
# 1. Vectorise the colour images, and then apply PCA. 
# Using cross-validation to learn optimal k components. 
# Train the SVM with k components. 
# 2. Apply a convolutional encoder.

sources:
https://www.youtube.com/watch?v=QdBy02ExhGI
https://www.youtube.com/watch?v=Lsue2gEM9D0

"""
def components_in_folder():
  per_var_list = []
  path = '../../Data/Train/i2/'
  for filename in listdir(path):
    filepath = join(path, filename)
    if isfile(filepath):
      # print('filepath is', filepath)
      img = Image.open(filepath).convert('RGB') #change all rgba images to rgb
      # img = cv2.imread(filepath) #opencv read in BGR order
      npArr = np.array(img)
      npArr = npArr.reshape(-1, 3) #combines the width/height dimensions to 1 (1D array of all the pixels) (114*70, 3)
      data = pd.DataFrame(data=npArr, columns = ['r', 'g', 'b'])
      # print(data.head())
      
      #scale the data (to 0?)
      scaled_data = preprocessing.scale(data)

      # apply pca - works for rgb or bgr
      pca3 = PCA()
      principalComponents3 = pca3.fit_transform(scaled_data)
      principalDf3 = pd.DataFrame(data = principalComponents3, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

      per_var = np.round(pca3.explained_variance_ratio_* 100, decimals=1)
      per_var_list.append(per_var)
      # labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

      # plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
      # plt.ylabel('Percentage of Explained Variance')
      # plt.xlabel('Principal Component')
      # plt.title=('Scree Plot')
      # plt.show() 
  # print(per_var_list)

  # get all the PC contribution percentages in a training folder
  pc1_list = []
  pc2_list = []
  pc3_list = []
  for i in range(len(per_var_list)):
    pc1_list.append(per_var_list[i][0])
    pc2_list.append(per_var_list[i][1])
    pc3_list.append(per_var_list[i][2])
  print(len(pc1_list))

  plt.scatter(range(len(pc1_list)), pc1_list, color='red', s=0.7, label='PC1')
  plt.scatter(range(len(pc2_list)), pc2_list, color='orange', s=0.7, label='PC2')
  plt.scatter(range(len(pc3_list)), pc3_list, color='yellow', s=0.7, label='PC3')
  plt.ylabel('Percentage of Explained Variance')
  plt.xlabel('Pics')
  plt.legend(loc='lower right')
  plt.show()

def get_pc1(folder_directory, folder_name):
  pc1_list =[]
  per_var_list = []
  path = join(folder_directory, folder_name)
  for filename in listdir(path):
    filepath = join(path, filename)
    if isfile(filepath):
      img = Image.open(filepath).convert('RGB')
      npArr = np.array(img)
      npArr = npArr.reshape(-1, 3) #combines the width/height dimensions to 1 (1D array of all the pixels) (114*70, 3)
      data = pd.DataFrame(data=npArr, columns = ['r', 'g', 'b'])
      
      #scale the data (to 0?)
      scaled_data = preprocessing.scale(data)

      # apply pca
      pca = PCA(1)
      principalComponents = pca.fit_transform(scaled_data)
      principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1'])
      # print(principalDf)
      per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
      per_var_list.append(per_var)
  for i in range(len(per_var_list)):
    pc1_list.append(per_var_list[i][0])
  return statistics.mean(pc1_list)

def main():
  path_to_get = '../../Data/Train/'
  folders = ['w57', 'pl5', 'pl40', 'p5', 'p26', 'p11', 'io', 'pl30', 'pl80', 'pn', 'po', 'p23', 'i4', 'i2', 'i5', 'ip', 'pne', 'pl50', 'pl60']
  result = []
  for f in folders:
    result.append(get_pc1(path_to_get, f))
  y_pos = np.arange(len(folders))
  plt.bar(y_pos, result)
  plt.xticks(y_pos, folders)
  plt.show()


main()