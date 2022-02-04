# Computer vision
This repo combines: 
1. class-agnostic segmentation with wrappers for Detectron2 and MMDetection
2. classification based on transformer feature extractor and kNN classifier


## Preparations:
1. clone this folder to your workspace/src directory

## Environment setup with Anaconda
1. Create anaconda environment: ```conda env create -n conda_environment.yml```
2. ```conda activate segmentation_ros```
3. download model checkpoint from **[GDrive](https://drive.google.com/file/d/1mrNft0aeIqAggnsW2WRUrhQexIHl0shU/view?usp=sharing)** and put it in scripts/checkpoints folder

## Environment setup with Docker

1. build docker image ```sudo sh build_docker.sh```
2. download model checkpoint from **[GDrive](https://drive.google.com/file/d/1mrNft0aeIqAggnsW2WRUrhQexIHl0shU/view?usp=sharing)** and put it in scripts/checkpoints folder
3. In line 7 in ```run_docker.sh``` change first path to your workspace folder
4. run docker container ```sudo sh run_docker.sh```
5. ```cd ws; catkin_make; source devel/setup.bash```

## Using
```roslaunch segmentation segmentation_node.launch```
