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
        self.pred_size = prediction_window_size
        super().__init__()
        self.wind_layer = nn.Sequential(
            nn.Linear(2, hidden_size_wind),
            nn.ReLU(),
            nn.Linear(hidden_size_wind, hidden_size_wind),
            nn.ReLU(),
            nn.Linear(hidden_size_wind, hidden_size_wind),
            nn.ReLU(),
        )

        self.conv1 = nn.Conv2d(1, 32, kernel_size, 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size, 1)
        conv_output_size = input_size - 2 * (kernel_size - 1)

        self.fc1 = nn.Linear(
            (64 * conv_output_size * conv_output_size) + hidden_size_wind,
            hidden_size_lidar,
        )
        self.fc2 = nn.Linear(hidden_size_lidar, hidden_size_lidar // 2)
        self.fc3 = nn.Linear(
            hidden_size_lidar // 2, 2 * self.pred_size * self.pred_size
        )

    def forward(self, x):
        lidar_input, wind_input = x[:, :-2], x[:, -2:]
        lidar_input = lidar_input.view(lidar_input.size(0), 1, -1)  # reshape to 2D

        # do linear on the wind data
        wind_input = torch.relu(self.wind_layer(wind_input))

        # do conv on the lidar data
        lidar_input = torch.relu(self.conv1(lidar_input))
        lidar_input = torch.relu(self.conv2(lidar_input))
        lidar_input = lidar_input.view(lidar_input.size(0), -1)

        #  concatenate the two inputs
        x = torch.cat((lidar_input, wind_input), 1)

        # fully connected layers on the concatenated data
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        # reshape to prediction window size
        return x.view(x.size(0), self.pred_size, self.pred_size, 2)
