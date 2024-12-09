import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


def polar_to_cartesian(lidar_data_batches, grid_size=100):
    """
    Convert 1D lidar polar data into a 2D Cartesian grid.

    Args:
        lidar_data_batches (torch.Tensor): Tensor of shape [batch_size, 360]
        grid_size (int): Size of the 2D grid (grid_size x grid_size).

    Returns:
        torch.Tensor: 2D binary image of shape [batch_size, 1, grid_size, grid_size]
    """
    # Create device and dtype consistent with input
    device = lidar_data_batches.device
    dtype = lidar_data_batches.dtype

    # Prepare grid and angles
    grid = torch.zeros(
        lidar_data_batches.size(0), 1, grid_size, grid_size, device=device, dtype=dtype
    )
    center = grid_size // 2

    # Angles in radians
    angles = torch.linspace(0, 2 * torch.pi, 360, device=device)

    for b in range(lidar_data_batches.size(0)):
        lidar_data = lidar_data_batches[b]
        for r, theta in zip(lidar_data, angles):
            if r > 0 and r < 1:  # Ignore zero-magnitude points
                # Convert polar to Cartesian coordinates
                for mag in torch.linspace(r, 2, steps=int((2 - r) * 500) + 1):
                    x = mag * torch.cos(theta)
                    y = mag * torch.sin(theta)

                    # Map to grid coordinates
                    x_grid = int(center + x * 100)
                    y_grid = int(center + y * 100)

                    # Ensure the coordinates are within bounds
                    if 0 <= x_grid < grid_size and 0 <= y_grid < grid_size:
                        grid[b, 0, y_grid, x_grid] = 1
        # plt.imshow(grid[b, 0, :, :].cpu().numpy())
        # plt.show()
    return grid


class CNN_2D(nn.Module):
    def __init__(
        self,
        input_size,
        kernel_size,
        hidden_size_lidar,
        hidden_size_wind,
        prediction_window_size,
    ):
        super().__init__()
        self.pred_size = prediction_window_size

        # Wind processing layers
        self.wind_layer = nn.Sequential(
            nn.Linear(2, hidden_size_wind),
            nn.ReLU(),
            nn.Linear(hidden_size_wind, hidden_size_wind),
            nn.ReLU(),
            nn.Linear(hidden_size_wind, hidden_size_wind),
            nn.ReLU(),
        )

        # Convolutional layers for lidar input
        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

        # Calculate conv output size
        conv_output_size = 32 * 250 * 250

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size + hidden_size_wind, hidden_size_lidar)
        self.fc2 = nn.Linear(hidden_size_lidar, hidden_size_lidar // 2)
        self.fc3 = nn.Linear(
            hidden_size_lidar // 2, 2 * self.pred_size * self.pred_size
        )

    def forward(self, lidar_input, wind_input):
        # Convert lidar input to Cartesian grid
        lidar_input = polar_to_cartesian(lidar_input, grid_size=250)

        # Process wind input
        wind_input = self.wind_layer(wind_input)

        # Apply convolutional layers to lidar input
        lidar_input = torch.relu(self.conv1(lidar_input))
        lidar_input = torch.relu(self.conv2(lidar_input))
        lidar_input = lidar_input.view(lidar_input.size(0), -1)  # Flatten

        # Concatenate lidar and wind features
        x = torch.cat((lidar_input, wind_input), dim=1)

        # Apply fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        # Reshape to prediction window size
        return x.view(x.size(0), self.pred_size, self.pred_size, 2)
