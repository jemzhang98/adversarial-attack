from os import walk
import os
from os.path import join
import matplotlib.pyplot as plt


folders = ['w57', 'pl5', 'pl40', 'p5', 'p26', 'p11', 'io', 'pl30', 'pl80', 'pn', 'po', 'p23', 'i4', 'i2', 'i5', 'ip', 'pne', 'pl50', 'pl60']

path = '../../Augmented_Data/Train'

count = []
for folder in folders:
  count.append(len(os.listdir(join(path, folder))))

  plt.bar(folders, count)
  plt.xlabel('Classes')
  plt.ylabel('Number of Images')
  plt.title('Labelled Images in New Train Data')
  plt.show()