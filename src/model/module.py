from lightning.pytorch import LightningModule
from model.decoder import MLPDecoder
from model.cnn_2d import CNN_2D
import torch
from torch import nn
from torchmetrics import MeanSquaredError
from utils.metrics import compute_and_save_my_metrics, compute_velocity_error, compute_direction_error


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


class BinartizedCNN2D(LightningModule):

    def __init__(
        self,
        prediction_size=10,
        learning_rate=0.0002,
    ):

        super().__init__()
        self.save_hyperparameters()
        self.model = CNN_2D(362, 256, prediction_size)
        self.mse_criterion = nn.L1Loss()
        self.metric = MeanSquaredError()

    def training_step(self, batch, batch_idx):
        prediction_gt, wind_vector, lidar_scan = batch
        input_data = torch.cat([wind_vector, lidar_scan], dim=1)
        prediction = self.model(input_data)
        loss = self.mse_criterion(prediction, prediction_gt)
        velocity_diff = compute_velocity_error(prediction, prediction_gt)
        direction_diff = compute_direction_error(prediction, prediction_gt)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log(
            "train_velocity_diff",
            velocity_diff,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        self.log(
            "train_direction_diff",
            direction_diff,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Calculates metrics for validation step
        """
        prediction_gt, wind_vector, lidar_scan = batch
        input_data = torch.cat([wind_vector, lidar_scan], dim=1)
        prediction = self.model(input_data)
        velocity_diff = compute_velocity_error(prediction, prediction_gt)
        direction_diff = compute_direction_error(prediction, prediction_gt)
        self.log(
            "val_velocity_diff",
            velocity_diff,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log(
            "val_direction_diff",
            direction_diff,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

    def configure_optimizers(self):
        lr = self.hparams["learning_rate"]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return optimizer
