# Computer vision
This repo combines: 
1. class-agnostic segmentation with wrappers for Detectron2 and MMDetection
2. classification based on transformer feature extractor and kNN classifier


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
```roslaunch segmentation segmentation_node.launch```
