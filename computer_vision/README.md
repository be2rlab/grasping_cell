# Computer vision
This repo combines: 
1. class-agnostic segmentation with wrappers for Detectron2 and MMDetection
2. classification based on transformer feature extractor and kNN classifier

## Environment setup with Anaconda

1. Create anaconda environment: ```conda env create -n conda_environment.yml```
2. ```conda activate segmentation_ros```
3. download model checkpoint from **[GDrive](https://drive.google.com/file/d/1mrNft0aeIqAggnsW2WRUrhQexIHl0shU/view?usp=sharing)** and put it in scripts/checkpoints folder


## Using
```roslaunch segmentation segmentation_node.launch```
