import torch
from torch import nn


class MLPDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, prediction_window_size):
        self.pred_size = prediction_window_size
        super().__init__()
        # a separte one layer that takes (2,) as the input
        self.sep_layer = nn.Linear(2, 64)
        self.fc1 = nn.Linear(input_size - 2, hidden_size)

        # now merge output of sep_layer and fc1

        self.fc2 = nn.Linear(hidden_size + 64, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size, hidden_size * 3)
        self.fc4 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc5 = nn.Linear(hidden_size * 4, (self.pred_size**2) * 2)

    def forward(self, x):
        # split x into 360, and 2,
        x, x_wind = x[:, :-2], x[:, -2:]
        x_wind = torch.relu(self.sep_layer(x_wind))
        x = torch.relu(self.fc1(x))
        x = torch.cat([x, x_wind], dim=1)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        # reshape to prediction window size
        return x.view(x.size(0), self.pred_size, self.pred_size, 2)
