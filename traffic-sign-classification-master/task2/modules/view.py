import torch.nn as nn


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(-1, self.size)
