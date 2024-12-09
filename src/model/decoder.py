import torch
from torch import nn


class MLPDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, prediction_window_size):
        self.pred_size = prediction_window_size
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Sequential(
            *[
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size * 3),
                nn.ReLU(),
                nn.Linear(hidden_size * 3, hidden_size * 4),
                nn.ReLU(),
            ]
        )
        self.fc3 = nn.Linear(hidden_size * 4, (self.pred_size**2) * 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        # reshape to prediction window size
        return x.view(x.size(0), self.pred_size, self.pred_size, 2)
