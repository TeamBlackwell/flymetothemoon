# Learning Local Wind Fields LT :zap: :helicopter:

Deep Learning project to predict local wind fields in urban landscapes through range sensing.

This project aims to predict the local windflow around a drone in an urban environment through range sensing.
The model does not use the entire map knowledge but rather LiDAR data to understand the structures present around it.

The lightweight nature of the model allows it to efficiently run on mobile and power constrained environments such as a drone.

## Data

The 2D data for the urban cityscapes, windflows and LiDAR data have been generated synthetically.
The code for data generation can be found [here](https://github.com/TeamBlackwell/SyntheticDataGen).

The data folder file structure is mentioned below.

```
data/
├─ cityscapes/
│  ├─ city_0.csv
│  ├─ city_1.csv
│  ├─ city_(...).csv
│  ├─ city_60.csv
├─ drone_positions/
│  ├─ city_0.csv
│  ├─ city_1.csv
│  ├─ city_(...).csv
│  ├─ city_60.csv
├─ exportviz/
│  ├─ city_0.png
│  ├─ city_1.png
│  ├─ city_(...).png
│  ├─ city_60.png
├─ matlab_meshes/
│  ├─ city_0.mat
│  ├─ city_1.mat
│  ├─ city_(...).mat
│  ├─ city_60.mat
├─ lidar/
│  ├─ city_0_pos0.npy
│  ├─ city_0_pos1.npy
│  ├─ city_0_pos(...).npy
│  ├─ city_1_pos0.npy
│  ├─ city_1_pos1.npy
│  ├─ city_1_pos(...).npy
│  ├─ city_(...)_pos(...).npy
├─ windflow/
│  ├─ city_0.npy
│  ├─ city_1.npy
│  ├─ city_(...).npy
│  ├─ city_60.npy
├─ pointclouds/
│  ├─ city_0/
│  │  ├─ pointcloud_1.csv
│  │  ├─ pointcloud_(...).csv
│  │  ├─ pointcloud_10.csv
│  ├─ city_1/
│  │  ├─ pointcloud_1.csv
│  │  ├─ pointcloud_(...).csv
│  │  ├─ pointcloud_10.csv
│  ├─ city_60/
│  │  ├─ pointcloud_1.csv
│  │  ├─ pointcloud_(...).csv
│  │  ├─ pointcloud_10.csv
```