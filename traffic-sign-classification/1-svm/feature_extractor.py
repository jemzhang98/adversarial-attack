#%%
import cv2
import numpy as np
import os
from phog import get_sampling_grid, calc_phog
from hog_x import calc_hog1, calc_hog2, calc_hog3
from tqdm import trange


def extract_features(filenames) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    n_sample = len(filenames)
    grids_y, grids_x, weights = get_sampling_grid((28, 28), [(14, 14), (7, 7), (4, 4)])

    feat_intensity = np.empty((n_sample, 784))
    feat_phog = np.empty((n_sample, 2172))
    feat_hog1 = np.empty((n_sample, 1568))
    feat_hog2 = np.empty((n_sample, 1568))
    feat_hog3 = np.empty((n_sample, 2916))

    for index in trange(n_sample):
        img: np.ndarray = cv2.imread(filenames[index])
        img_gray1: np.ndarray = cv2.cvtColor(cv2.resize(img, (28, 28)), cv2.COLOR_BGR2GRAY)
        img_gray2: np.ndarray = cv2.cvtColor(cv2.resize(img, (40, 40)), cv2.COLOR_BGR2GRAY)

        phog = calc_phog(img_gray1, 12, grids_y, grids_x, weights)
        hog1 = calc_hog1(img_gray2)
        hog2 = calc_hog2(img_gray2)
        hog3 = calc_hog3(img_gray2)

        feat_intensity[index, :] = img_gray1.flatten()
        feat_phog[index, :] = phog
        feat_hog1[index, :] = hog1
        feat_hog2[index, :] = hog2
        feat_hog3[index, :] = hog3

    return feat_intensity, feat_phog, feat_hog1, feat_hog2, feat_hog3


def extract_features_and_save(filenames, save_path='./features'):
    feat_intensity, feat_phog, feat_hog1, feat_hog2, feat_hog3 = extract_features(filenames)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, 'feat_intensity'), feat_intensity)
    np.save(os.path.join(save_path, 'feat_phog'), feat_phog)
    np.save(os.path.join(save_path, 'feat_hog1'), feat_hog1)
    np.save(os.path.join(save_path, 'feat_hog2'), feat_hog2)
    np.save(os.path.join(save_path, 'feat_hog3'), feat_hog3)


def load_all_features(file_path='./features', combined=True):
    feat_intensity: np.ndarray = np.load(os.path.join(file_path, 'feat_intensity.npy'))
    feat_phog: np.ndarray = np.load(os.path.join(file_path, 'feat_phog.npy'))
    feat_hog1: np.ndarray = np.load(os.path.join(file_path, 'feat_hog1.npy'))
    feat_hog2: np.ndarray = np.load(os.path.join(file_path, 'feat_hog2.npy'))
    feat_hog3: np.ndarray = np.load(os.path.join(file_path, 'feat_hog3.npy'))
    if combined:
        return np.hstack((feat_intensity, feat_phog, feat_hog1, feat_hog2, feat_hog3))
    else:
        return feat_intensity, feat_phog, feat_hog1, feat_hog2, feat_hog3


def load_phog_feature(file_path='./features'):
    feat_phog: np.ndarray = np.load(os.path.join(file_path, 'feat_phog.npy'))
    return feat_phog
