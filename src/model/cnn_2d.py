import torch
from torch import nn


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
        conv_output_size = 32 * 75 * 75

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size + hidden_size_wind, hidden_size_lidar)
        self.fc2 = nn.Linear(hidden_size_lidar, hidden_size_lidar // 2)
        self.fc3 = nn.Linear(
            hidden_size_lidar // 2, 2 * self.pred_size * self.pred_size
        )

    def forward(self, lidar_input, wind_input):
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
