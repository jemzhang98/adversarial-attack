import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LCN(nn.Module):
    def __init__(self, kernel_shape, sigma=1):
        super(LCN, self).__init__()
        self.padding = (kernel_shape[2] // 2, kernel_shape[3] // 2)
        self.kernel = gaussian_kernel(kernel_shape, sigma)

    def forward(self, x):
        filtered_out = F.conv2d(x, self.kernel, padding=self.padding)

        # Subtractive Normalization
        centered_image = x - filtered_out

        # Variance Calc
        sum_sqr_image = F.conv2d(centered_image.pow(2), self.kernel, padding=self.padding)
        s_deviation = sum_sqr_image.sqrt()
        per_img_mean = s_deviation.mean()

        # Divisive Normalization
        divisor = torch.max(per_img_mean, s_deviation)
        divisor = torch.max(divisor, torch.Tensor([1e-4]).to(device))
        # divisor = torch.max(torch.Tensor([1e-4].to(device)), s_deviation)

        new_image = centered_image / divisor

        return new_image


def gaussian_kernel(kernel_shape, sigma):
    x = np.zeros(kernel_shape, dtype='float32')

    def gauss(x, y):
        return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    mid_i = kernel_shape[2] // 2
    mid_j = kernel_shape[3] // 2
    for kernel_idx in range(0, kernel_shape[1]):
        for i in range(0, kernel_shape[2]):
            for j in range(0, kernel_shape[3]):
                x[0, kernel_idx, i, j] = gauss(i - mid_i, j - mid_j)

    return torch.Tensor(x / np.sum(x)).to(device)
