# Learning Local Wind Fields through Range Sensing LT :zap: :helicopter:

Deep Learning project to predict local wind fields in urban landscapes.

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
├─ lidar_scans/
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
├─ windflows/
```
