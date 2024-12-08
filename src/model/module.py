from lightning.pytorch import LightningModule
from model.decoder import MLPDecoder
import torch
from torch import nn
from torchmetrics import MeanSquaredError
from pathlib import Path


class WindFlowDecoder(LightningModule):

    def __init__(
        self,
        prediction_size=10,
        learning_rate=0.0002,
    ):

        super().__init__()
        self.save_hyperparameters()

        self.decoder = MLPDecoder(362, 256, prediction_size)
        self.mse_criterion = nn.L1Loss()
        self.metric = MeanSquaredError()

    def training_step(self, batch, batch_idx):
        prediction_gt,wind_vector,lidar_scan = batch
        input_data = torch.cat([wind_vector,lidar_scan], dim=1)
        prediction = self.decoder(input_data)
        loss = self.mse_criterion(prediction, prediction_gt)
        acc = self.metric(prediction, prediction_gt)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log('train_mse', acc, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = self.hparams["learning_rate"]
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        return optimizer