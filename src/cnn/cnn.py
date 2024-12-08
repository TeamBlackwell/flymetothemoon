import torch
from torch import nn


class ConvEncoder(torch.nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()

        # in: (N, 1, 362) ; 360 Lidar, 2 x,y
        self.input_layer = nn.Conv1d(
            in_channels=1, out_channels=4, kernel_size=3
        )  # (N, 4, 360)
        self.hidden_layer_1 = nn.Conv1d(
            in_channels=4, out_channels=16, kernel_size=3, padding=1
        )  # (N, 16, 360)
        self.pool_conv_1 = nn.Conv1d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=2
        )  # (N, 16, 180)

        self.hidden_layer_2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )  # (N, 32, 180)
        self.pool_conv_2 = nn.Conv1d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2
        )  # (N, 32, 90)

        self.hidden_layer_3 = nn.Conv1d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )  # (N, 64, 90)
        self.pool_conv_3 = nn.Conv1d(
            in_channels=64, out_channels=64, kernel_size=5, padding=1, stride=3
        )  # (N, 64, 30)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer_1(x)
        x = self.pool_conv_1(x)

        x = self.hidden_layer_2(x)
        x = self.pool_conv_2(x)

        x = self.hidden_layer_3(x)
        x = self.pool_conv_3(x)

        return x


class MLPDecoder(torch.nn.Module):
    def __init__(self, pred_size):
        super(MLPDecoder, self).__init__()
        self.pred_size = pred_size

        # in: (N, 64, 30)
        self.flatten = nn.Flatten()  # (N, 1920)
        self.input_layer = nn.Linear(in_features=1920, out_features=512)  # (N, 512)
        self.hidden_layer_1 = nn.Linear(512, 256)  # (N, 256)
        self.hidden_layer_2 = nn.Linear(256, (pred_size**2) * 2)  # (N, 2*S^2)

    def forward(self, x):
        x = self.input_layer(self.flatten(x))
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)

        return x.view(1, self.pred_size, self.pred_size, 2)


class ConvModel(torch.nn.Module):
    def __init__(self, pred_size=11):
        super(ConvModel, self).__init__()
        self.encoder = ConvEncoder()
        self.decoder = MLPDecoder(pred_size=pred_size)

    def forward(self, x):
        x = self.encoder(x)
        y= self.decoder(x)
        return y
