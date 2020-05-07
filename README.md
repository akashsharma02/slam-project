# Offline Object SLAM Pipeline based on Open3D

This project implements a simple reconstruction system based on Open3D, and leverages semantic segmentation of images 
in a preprocessing step to improve reconstruction efficiency and accuracy. 

## Usage
Before you run the system you are required to run the preprocess step, which expects the following: 
```
python main.py -i <path to dataset> -o <path to output folder>
```
Note that this system requires detectron2 in a conda environment.

To run the system use:
```
python run_system <config_file> --make --register --refine --integrate
```
You may also run the separate tasks individually

To generate the ground truth reconstruction from the dataset ground truth poses use the below.
```
python run_system <config_file> --get_ground_truth 
```
System has been tested with *RGBD Scenes* and *BundleFusion* dataset

## Dataset structural requirement
```
.
├── camera-intrinsics.json
├── color [723 entries]
├── depth [723 entries]
├── fragments [created by pipeline]
├── ground_truth.ply [created by pipeline]
├── gt_poses [723 entries]
├── preprocessed [1446 entries] [created by preprocess step and moved to folder before run_system]
└── scene [created by pipeline]
```
