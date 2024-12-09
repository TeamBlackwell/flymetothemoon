import lightning as L
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def polar_to_cartesian(lidar_data, grid_size=100):
    """
    Convert 1D lidar polar data into a 2D Cartesian grid using vectorized operations.

    Args:
        lidar_data (torch.Tensor): Tensor of shape [360]
        grid_size (int): Size of the 2D grid (grid_size x grid_size).

    Returns:
        torch.Tensor: 2D binary image of shape [1, grid_size, grid_size]
    """
    device = lidar_data.device
    dtype = lidar_data.dtype

    # Create grid and prepare coordinates
    grid = torch.zeros(1, grid_size, grid_size, device=device, dtype=dtype)
    center = grid_size // 2

    # Generate angles and radial coordinates
    angles = torch.linspace(0, 2 * torch.pi, 360, device=device)
    max_range = 2.0  # Maximum range

    # Filter valid points
    valid_mask = (lidar_data > 0) & (lidar_data < 1)
    valid_lidar = lidar_data[valid_mask]
    valid_angles = angles[valid_mask]

    for r, theta in zip(valid_lidar, valid_angles):
        # Generate interpolated magnitudes
        mags = torch.linspace(
            r, max_range, steps=int((max_range - r) * 20) + 1, device=device
        )

        # Compute Cartesian coordinates
        x = mags * torch.cos(theta)
        y = mags * torch.sin(theta)

        # Map to grid coordinates
        x_grid = (center + x * 40).long()
        y_grid = (center + y * 40).long()

        # Filter valid grid coordinates
        valid_x = (x_grid >= 0) & (x_grid < grid_size)
        valid_y = (y_grid >= 0) & (y_grid < grid_size)
        valid_coords = valid_x & valid_y

        # Set grid points
        grid[0, y_grid[valid_coords], x_grid[valid_coords]] = 1

    return grid


class UrbanWinds2DDataModule(L.LightningDataModule):
    def __init__(
        self, data_dir: Path, batch_size: int = 4, prediction_window_size: int = 11
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.prediction_window_size = prediction_window_size

    def setup(self, stage):

        self.urbanflows_train = UrbanWindFlows2D(
            self.data_dir / "train", prediction_window_size=self.prediction_window_size
        )

        self.urbanflows_val = UrbanWindFlows2D(
            self.data_dir / "val", prediction_window_size=self.prediction_window_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.urbanflows_train,
            batch_size=1024,
            shuffle=True,
            num_workers=4,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.urbanflows_val,
            batch_size=512,
            shuffle=False,
            num_workers=4,
            persistent_workers=True
        )


class UrbanWinds2DLidarDataModule(L.LightningDataModule):
    def __init__(
        self, data_dir: Path, batch_size: int = 4, prediction_window_size: int = 11
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.prediction_window_size = prediction_window_size

    def setup(self, stage):

        self.urbanflows_train = UrbanWind2DLidar(
            self.data_dir / "train", prediction_window_size=self.prediction_window_size
        )

        self.urbanflows_val = UrbanWind2DLidar(
            self.data_dir / "val", prediction_window_size=self.prediction_window_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.urbanflows_train,
            batch_size=1024,
            shuffle=True,
            num_workers=6,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.urbanflows_val,
            batch_size=512,
            shuffle=False,
            num_workers=6,
            persistent_workers=True,
        )


class UrbanWinds2DGraphModule(L.LightningDataModule):
    def __init__(
        self, data_dir: Path, batch_size: int = 4, prediction_window_size: int = 11
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.prediction_window_size = prediction_window_size

    def setup(self, stage):

        self.urbanflows_train = UrbanWindFlows2D(
            self.data_dir / "train", prediction_window_size=self.prediction_window_size
        )

        self.urbanflows_val = UrbanWindFlows2D(
            self.data_dir / "val", prediction_window_size=self.prediction_window_size
        )

    def train_dataloader(self):
        return DataLoader(
            self.urbanflows_train,
            batch_size=60,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.urbanflows_val,
            batch_size=6,
            shuffle=False,
            num_workers=0,
        )


class UrbanWind2DLidar(Dataset):
    def __init__(self, path, prediction_window_size: int):
        self.windflow_dir = path / "windflow"
        self.lidar_dir = path / "lidar"
        self.position_df = pd.read_csv(path / "lidar_positions.csv")
        self.window_offset_from_center = prediction_window_size // 2
        self.lidar_memo = {}

    def __len__(self):
        return len(self.position_df)

    def __getitem__(self, idx):

        position = self.position_df.iloc[idx]
        city_id = int(position["city_id"])
        lidar_scan_id = int(position["position_id"])
        x, y = int(position["xr"]), int(position["yr"])

        # load the required data
        windflow_data = np.load(self.windflow_dir / f"city_{city_id}.npy")
        lidar_data = np.load(self.lidar_dir / f"city_{city_id}_pos{lidar_scan_id}.npy")

        # extract the ground truth
        l = x - self.window_offset_from_center - 1
        r = x + self.window_offset_from_center
        t = y - self.window_offset_from_center - 1
        b = y + self.window_offset_from_center

        # if somehow the window is out of bounds, pad with zeros
        if l < 0 or r >= windflow_data.shape[0] or t < 0 or b >= windflow_data.shape[1]:
            prediction_window_gt = np.zeros((self.window_offset_from_center * 2 + 1, self.window_offset_from_center * 2 + 1, 2))
        else:
            prediction_window_gt = windflow_data[l:r, t:b, :]

    
        # extract the input data
        w_x, w_y = windflow_data[x, y, :]

        # convert to tensor
        prediction_window_gt = torch.tensor(prediction_window_gt).float()
        w_x = torch.tensor(w_x).float()
        w_y = torch.tensor(w_y).float()
        lidar_data = torch.tensor(lidar_data).float()
        if lidar_scan_id in self.lidar_memo:
            lidar_data = self.lidar_memo[lidar_scan_id]
        else:
            lidar_data = polar_to_cartesian(lidar_data, grid_size=75)
            self.lidar_memo[lidar_scan_id] = lidar_data
        wind_vector = torch.tensor([w_x, w_y])
        
        # print(prediction_window_gt.shape, wind_vector.shape, lidar_data.shape)
        return prediction_window_gt, wind_vector, lidar_data


class UrbanWindFlows2D(Dataset):
    def __init__(self, path, prediction_window_size: int):
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
        x, y = int(position["xr"]), int(position["yr"])

        # load the required data
        windflow_data = np.load(self.windflow_dir / f"city_{city_id}.npy")
        lidar_data = np.load(self.lidar_dir / f"city_{city_id}_pos{lidar_scan_id}.npy")

        # extract the ground truth
        l = x - self.window_offset_from_center - 1
        r = x + self.window_offset_from_center
        t = y - self.window_offset_from_center - 1
        b = y + self.window_offset_from_center


        # if somehow the window is out of bounds, pad with zeros
        if l < 0 or r >= windflow_data.shape[0] or t < 0 or b >= windflow_data.shape[1]:
            prediction_window_gt = np.zeros((self.window_offset_from_center * 2 + 1, self.window_offset_from_center * 2 + 1, 2))
        else:
            prediction_window_gt = windflow_data[l:r, t:b, :]

        # extract the input data
        w_x, w_y = windflow_data[x, y, :]

        # convert to tensor
        prediction_window_gt = torch.tensor(prediction_window_gt).float()
        w_x = torch.tensor(w_x).float()
        w_y = torch.tensor(w_y).float()
        lidar_data = torch.tensor(lidar_data).float()
        wind_vector = torch.tensor([w_x, w_y])

        return prediction_window_gt, wind_vector, lidar_data
