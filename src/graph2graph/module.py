from lightning.pytorch import LightningModule
from graph2graph.graph2graph import Graph2GraphModel

import torch
from torch import nn
from torchmetrics import MeanSquaredError
from pathlib import Path


class WindflowCNN(LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = ConvModel()
        self.mse_criterion = nn.L1Loss()
        self.metric = MeanSquaredError()

    def training_step(self, batch):
        prediction_gt, wind_vector, lidar_scan = batch
        input_data = torch.cat([wind_vector, lidar_scan], dim=1)
        prediction = self.model(input_data)
        loss = self.mse_criterion(prediction, prediction_gt)
        acc = self.metric(prediction, prediction_gt)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_mse", acc, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch):
        prediction_gt, wind_vector, lidar_scan = batch
        input_data = torch.cat([wind_vector, lidar_scan], dim=1)
        prediction = self.model(input_data)
        loss = self.mse_criterion(prediction, prediction_gt)
        acc = self.metric(prediction, prediction_gt)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_mse", acc, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = self.hparams["learning_rate"]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return optimizer
