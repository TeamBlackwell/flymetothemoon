import torch
from lightning.pytorch.cli import LightningCLI
from datasets.urban_2d_datamodule import UrbanWinds2DDataModule, UrbanWinds2DGraphModule
from model.module import WindFlowDecoder
from model_in_sep.module import WindFlowDecoderButWithSep
from cnn.module import WindflowCNN

torch.set_float32_matmul_precision("medium")


def cli_main():
    cli = LightningCLI()


if __name__ == "__main__":
    cli_main()
