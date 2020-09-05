import cv2
import numpy as np


hog1_descriptor = cv2.HOGDescriptor(_winSize=(40, 40), _blockSize=(10, 10), _blockStride=(5, 5),
                                    _cellSize=(5, 5), _nbins=8, _signedGradient=False)
hog2_descriptor = cv2.HOGDescriptor(_winSize=(40, 40), _blockSize=(10, 10), _blockStride=(5, 5),
                                    _cellSize=(5, 5), _nbins=8, _signedGradient=True)
hog3_descriptor = cv2.HOGDescriptor(_winSize=(40, 40), _blockSize=(8, 8), _blockStride=(4, 4),
                                    _cellSize=(4, 4), _nbins=9, _signedGradient=False)


def calc_hog1(img: np.ndarray) -> np.ndarray:
    return hog1_descriptor.compute(img).flatten()


def calc_hog2(img: np.ndarray) -> np.ndarray:
    return hog2_descriptor.compute(img).flatten()


def calc_hog3(img: np.ndarray) -> np.ndarray:
    return hog3_descriptor.compute(img).flatten()
