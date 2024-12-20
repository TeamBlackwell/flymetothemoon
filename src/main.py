import torch
from lightning.pytorch.cli import LightningCLI
from datasets.urban_2d_datamodule import (
    UrbanWinds2DDataModule,
    UrbanWinds2DGraphModule,
    UrbanWinds2DLidarDataModule,
)

from model.module import WindFlowDecoder, WindFlowDecoderAdvanced, BinarizedCNN2D
from model_in_sep.module import WindFlowDecoderButWithSep
from cnn.module import WindflowCNN
from graph2graph.module import GraphNN

torch.set_float32_matmul_precision("medium")


def cli_main():
    cli = LightningCLI()


if __name__ == "__main__":
    cli_main()
