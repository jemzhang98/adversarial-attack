import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.view import View


class ST(nn.Module):
    def __init__(self, input_sizes, loc_net_sizes):
        super(ST, self).__init__()

        fc1_input_size = loc_net_sizes[1] * (input_sizes[1] // 2 // 2 // 2) ** 2
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(input_sizes[0], loc_net_sizes[0], kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(loc_net_sizes[0], loc_net_sizes[1], kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            View(fc1_input_size),
            nn.Linear(fc1_input_size, loc_net_sizes[2]),
            nn.ReLU(True),
            nn.Linear(loc_net_sizes[2], 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def forward(self, x):
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x
