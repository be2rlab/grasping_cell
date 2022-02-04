# Computer vision
This repo combines: 
1. class-agnostic segmentation with wrappers for Detectron2 and MMDetection
2. classification based on transformer feature extractor and kNN classifier

# System requirements

This project was tested with:
- Ubuntu 20.04
- ROS noetic
- torch 1.10
- CUDA 11.3
- NVIDIA GTX 1050ti / RTX 3090

## Preparations:
1. clone this folder to your workspace/src directory
2. download model checkpoint and config from **[GDrive](https://drive.google.com/file/d/1GHeLyvsXV3rrEWwBA5H-omxduFUOOlH7/view?usp=sharing)** and extract it in scripts/checkpoints folder

## Environment setup with Anaconda
1. Create anaconda environment: ```conda env create -n conda_environment.yml```
2. ```conda activate segmentation_ros```

## Environment setup with Docker

1. build docker image ```sudo sh build_docker.sh```
2. In line 7 in ```run_docker.sh``` change first path to your workspace folder
3. run docker container ```sudo sh run_docker.sh```
4. ```cd ws; catkin_make; source devel/setup.bash```

## Using
Run node:
```roslaunch segmentation segmentation_node.launch```

By default, it runs publisher. Optionally you can pass an argument mode:=service to run in service mode.
Along with inference mode, this node has training mode to save new objects in classifier.

An algorithm for training:
1. place a new object in a field of view of camera so that it is the nearest detected object in s screen.
2. Call \segmentation_train_service to mask this object, get featues from feature extractor and save them
3. Repeat previous step with different angle of view
4. Call \segmentation_end_train_service to add all saved features to kNN.

