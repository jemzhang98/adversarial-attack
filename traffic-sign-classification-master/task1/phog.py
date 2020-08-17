import numpy as np
import scipy.ndimage
from numpy import pi
from typing import List

dog_mask = np.array(
    [(0.0069, 0.0101, 0.0115, 0.0101, 0.0069),
     (0.0684, 0.1012, 0.1152, 0.1012, 0.0684),
     (0.0000, 0.0000, 0.0000, 0.0000, 0.0000),
     (-0.0684, -0.1012, -0.1152, -0.1012, -0.0684),
     (-0.0069, -0.0101, -0.0115, -0.0101, -0.0069)]
)


def calc_phog(img: np.ndarray, n_bin: int, grids_y: List[np.ndarray], grids_x: List[np.ndarray],
              weights: List) -> np.ndarray:
    gradient = calc_gradient(img, n_bin)
    feature = calc_gradient_feature(gradient, grids_y, grids_x, weights)
    return feature


def get_sampling_grid(img_size, block_size_list: List) -> (List[np.ndarray], List[np.ndarray], List[np.ndarray]):
    overlap_factor = 0.5
    grids_y = []
    grids_x = []
    weights = []
    for s in range(len(block_size_list)):
        overlap_y = block_size_list[s][0] * overlap_factor
        overlap_x = block_size_list[s][1] * overlap_factor
        pos_y = np.unique(np.round(np.arange(0, block_size_list[s][0] - overlap_y + 1, overlap_y)).astype(np.int))
        pos_x = np.unique(np.round(np.arange(0, block_size_list[s][1] - overlap_x + 1, overlap_x)).astype(np.int))
        offset_y = np.floor(np.mod(img_size[0], block_size_list[s][0]) / 2).astype(np.int)
        offset_x = np.floor(np.mod(img_size[1], block_size_list[s][1]) / 2).astype(np.int)

        for i in range(len(pos_y)):
            for j in range(len(pos_x)):
                if i == len(pos_y) - 1 and j == len(pos_x) - 1:
                    continue
                y, x = np.meshgrid(np.arange(offset_y + pos_y[i], img_size[0] + 1, block_size_list[s][0]),
                                   np.arange(offset_x + pos_x[j], img_size[1] + 1, block_size_list[s][1]),
                                   indexing='ij')
                grids_y.append(y)
                grids_x.append(x)
                weights.append(2 ** s)

    return grids_y, grids_x, weights


def calc_gradient(img: np.ndarray, n_bin: int, smooth=False) -> np.ndarray:
    img = img.astype(np.float)
    g_y = scipy.ndimage.correlate(img, dog_mask, mode='nearest')
    g_x = scipy.ndimage.correlate(img, dog_mask.T, mode='nearest')
    y, x = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
    angle = np.arctan2(g_y, g_x)
    magnitude = np.power(np.power(g_y, 2) + np.power(g_x, 2), 0.5)
    result = np.zeros((*img.shape, n_bin))

    # make bins and compute histogram
    min_bin = -pi
    max_bin = pi
    bin_size = (max_bin - min_bin) / n_bin
    center_offset = bin_size / 2
    bin_centers = np.arange(min_bin + center_offset, max_bin - center_offset + bin_size, bin_size)

    # compute bin index
    g_bin = np.fmin(n_bin - 1, np.floor(n_bin / (max_bin - min_bin) * (angle - min_bin))).astype(np.int)

    if smooth:
        # no smoothing (performs better on test)
        result.put(np.ravel_multi_index((y, x, g_bin), dims=result.shape), magnitude)
    else:
        # linearly interpolate the membership between adjacent bins
        frac = (angle - bin_centers[g_bin]) / bin_size
        g_bin_neighbor = np.mod(g_bin + np.sign(frac), n_bin).astype(np.int)
        result.put(np.ravel_multi_index((y, x, g_bin), dims=result.shape), magnitude * (1 - np.abs(frac)))
        result.put(np.ravel_multi_index((y, x, g_bin_neighbor), dims=result.shape), magnitude * np.abs(frac))

    return result


def calc_gradient_feature(gradient: np.ndarray, grids_y: List[np.ndarray], grids_x: List[np.ndarray],
                          weights: List) -> np.ndarray:
    n_bin = gradient.shape[2]
    dim = calc_feature_dim(grids_y, n_bin)
    n_level = len(dim)
    feature_list = []
    g_cum = cumsum_2d(gradient)
    weight_sum = g_cum.sum() + 1e-10

    for k in range(n_level):
        xs = grids_x[k]
        ys = grids_y[k]
        feature_list.append(np.zeros(dim[k]))
        dim_index = 0
        for i in range(1, xs.shape[0]):
            for j in range(1, xs.shape[1]):
                x = xs[i, j]
                x_ = xs[i - 1, j - 1]
                y = ys[i, j]
                y_ = ys[i - 1, j - 1]
                cell_sum = g_cum[y, x, :] - g_cum[y_, x, :] - g_cum[y, x_, :] + g_cum[x_, y_, :]
                feature_list[k][dim_index:dim_index + n_bin] = cell_sum / weight_sum * weights[k]
                dim_index += n_bin

    return np.hstack(feature_list)


def calc_feature_dim(grid: List[np.ndarray], n_bin):
    dim = np.zeros(len(grid))
    for i in range(len(grid)):
        dim[i] = n_bin * (grid[i].shape[0] - 1) * (grid[i].shape[1] - 1)
    return dim.astype(np.int)


def cumsum_2d(x: np.ndarray):
    y = np.zeros((x.shape[0] + 1, x.shape[1] + 1, x.shape[2]))
    y[1:, 1:, :] = np.cumsum(np.cumsum(x, 0), 1)
    return y
