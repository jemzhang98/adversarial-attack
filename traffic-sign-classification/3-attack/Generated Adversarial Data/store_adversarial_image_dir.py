import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import cm

"""
This file will take in a .npy file (image array) and create a directory of images.
"""

path = './square_svm_adv.npy'
# path2 = './square_svm_color_adv.npy'
path2 = './boundary_cnn_adv.npy'

advImgArr = np.load(path2)

output_path = './Test_Images_Cnn_Color'
count = 4000
for img in advImgArr:
  im = Image.fromarray(np.uint8((img*255)).reshape(48, 48, 3))
  # print(img.shape)
  # print(np.array(im).shape)
  im.save(output_path + '/test_' + str(count) + '.png')
  count += 1


