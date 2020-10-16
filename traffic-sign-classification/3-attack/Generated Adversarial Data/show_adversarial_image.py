import numpy as np
from matplotlib import pyplot as plt


filePath = r'.\square_svm_adv.npy'
advData = np.load(filePath)
if advData.dtype != np.uint8 and advData.shape[3] != 1:
    advData = advData.astype(np.uint8)

for image in advData:
    # showImage = np.reshape(image, (50, 50))
    # plt.imshow(showImage)
    if image.shape[2] == 1:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.show()

