from lightning.pytorch import LightningModule
from model.decoder import MLPDecoder
from model.cnn_2d import CNN_2D
import torch
from torch import nn
from torchmetrics import MeanSquaredError
from utils.metrics import (
    compute_and_save_my_metrics,
    compute_velocity_error,
    compute_direction_error,
)


class WindFlowDecoder(LightningModule):

    def __init__(
        self,
        prediction_size=10,
        learning_rate=1e-3,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.decoder = MLPDecoder(362, 256, prediction_size)
        self.mse_criterion = nn.MSELoss()
        # self.metric = MeanSquaredError()

    def training_step(self, batch, batch_idx):
        prediction_gt, wind_vector, lidar_scan = batch
        input_data = torch.cat([wind_vector, lidar_scan], dim=1)
        prediction = self.decoder(input_data)

        loss = self.mse_criterion(prediction, prediction_gt)

        compute_and_save_my_metrics(self, loss, prediction, prediction_gt, val=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Calculates metrics for validation step
        """
        prediction_gt, wind_vector, lidar_scan = batch
        input_data = torch.cat([wind_vector, lidar_scan], dim=1)
        prediction = self.decoder(input_data)

        loss = self.mse_criterion(prediction, prediction_gt)

        compute_and_save_my_metrics(self, loss, prediction, prediction_gt, val=True)

    def configure_optimizers(self):
        lr = self.hparams["learning_rate"]
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        return optimizer


class BinarizedCNN2D(LightningModule):

    def __init__(
        self,
        prediction_size=10,
        learning_rate=0.0002,
    ):

        super().__init__()
        self.save_hyperparameters()
        self.model = CNN_2D(
            input_size=362,
            kernel_size=3,
            hidden_size_lidar=256,
            hidden_size_wind=64,
            prediction_window_size=prediction_size,
        )
        self.mse_criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        prediction_gt, wind_vector, lidar_scan = batch
        input_data = torch.cat([wind_vector, lidar_scan], dim=1)
        prediction = self.model(input_data)
        loss = self.mse_criterion(prediction, prediction_gt)
        compute_and_save_my_metrics(self, loss, prediction, prediction_gt, val=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Calculates metrics for validation step
        """
        prediction_gt, wind_vector, lidar_scan = batch
        input_data = torch.cat([wind_vector, lidar_scan], dim=1)
        prediction = self.model(input_data)
        loss = self.mse_criterion(prediction, prediction_gt)
        compute_and_save_my_metrics(self, loss, prediction, prediction_gt, val=True)

    def configure_optimizers(self):
        lr = self.hparams["learning_rate"]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return optimizer
