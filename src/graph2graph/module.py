from lightning.pytorch import LightningModule
from graph2graph.graph2graph import Graph2GraphModel

import torch
from torch import nn

from utils.metrics import compute_and_save_my_metrics


class GraphNN(LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = Graph2GraphModel()
        self.mse_criterion = nn.MSELoss()

    def training_step(self, batch):
        prediction_gt, wind_vector, lidar_scan = batch
        input_data = torch.cat([wind_vector, lidar_scan], dim=1)
        prediction = self.model(input_data)
        loss = self.mse_criterion(prediction, prediction_gt)

        compute_and_save_my_metrics(self, loss, prediction, prediction_gt, val=False)

        return loss

    def validation_step(self, batch):
        prediction_gt, wind_vector, lidar_scan = batch
        input_data = torch.cat([wind_vector, lidar_scan], dim=1)
        prediction = self.model(input_data)
        loss = self.mse_criterion(prediction, prediction_gt)
        compute_and_save_my_metrics(self, loss, prediction, prediction_gt, val=True)

        return loss

    def configure_optimizers(self):
        lr = self.hparams["learning_rate"]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        return optimizer
