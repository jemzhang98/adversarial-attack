import numpy as np
from matplotlib import pyplot as plt


filePath = r'.\square_svm_color_adv.npy'
advData = np.load(filePath)
if advData.dtype != np.unit8:
    advData = advData.astype(np.unit8)

for image in advData:
    # showImage = np.reshape(image, (50, 50))
    # plt.imshow(showImage)
    plt.imshow(image)
    plt.show()

