import lightning as L
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import pandas as pd
import numpy as np


class UrbanWinds2DDataModule(L.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int = 4, prediction_window_size: int = 10):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.prediction_window_size = prediction_window_size

    def setup(self,stage):

        self.urbanflows_train = UrbanWindFlows2D(
            self.data_dir / "train", prediction_window_size=self.prediction_window_size
        )

        self.urbanflows_val = UrbanWindFlows2D(
            self.data_dir / "val", prediction_window_size=self.prediction_window_size
        )
        

    def train_dataloader(self):
        return DataLoader(
            self.urbanflows_train,
            batch_size=1,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.urbanflows_val,
            batch_size=1,
            shuffle=True,
            num_workers=0,
        )

class UrbanWindFlows2D(Dataset):
    def __init__(self, path, prediction_window_size:int):
        self.windflow_dir = path / "windflow"
        self.lidar_dir = path / "lidar"
        self.position_df = pd.read_csv(path / "lidar_positions.csv")
        self.window_offset_from_center = prediction_window_size // 2


    def __len__(self):
        return len(self.position_df)

    def __getitem__(self, idx):


        position = self.position_df.iloc[idx]
        city_id = int(position["city_id"])
        lidar_scan_id = int(position["position_id"])
        x,y = int(position["xr"]), int(position["yr"])

        # load the required data
        windflow_data = np.load(self.windflow_dir / f"city_{city_id}.npy")
        lidar_data = np.load(self.lidar_dir /f"city_{city_id}_pos{lidar_scan_id}.npy")

        # extract the ground truth
        l = x - self.window_offset_from_center
        r = x + self.window_offset_from_center
        t = y - self.window_offset_from_center
        b = y + self.window_offset_from_center

        prediction_window_gt = windflow_data[l:r,t:b,:]

        # extract the input data
        w_x,w_y = windflow_data[x, y, :]

        # convert to tensor
        prediction_window_gt = torch.tensor(prediction_window_gt).float()
        w_x = torch.tensor(w_x).float()
        w_y = torch.tensor(w_y).float()
        lidar_data = torch.tensor(lidar_data).float()
        wind_vector = torch.tensor([w_x,w_y])

        return prediction_window_gt, wind_vector, lidar_data