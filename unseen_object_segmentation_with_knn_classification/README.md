# unseen_object_segmentation_with_knn_classification
This repo combines class-agnostic segmentation with classification based on transformer feature extractor and kNN classifier

# Installation

## Docker setup (preferred)
1. Create workspace: ```mkdir cv_ws/src -p; cd cv_ws/src```
2. clone this repo ```git clone -b ros_wrapper https://github.com/IvDmNe/unseen_object_segmentation_with_knn_classification.git```
3. build docker image: ```sh build_docker.sh```
4. download model checkpoint from **[GDrive](https://drive.google.com/file/d/1mrNft0aeIqAggnsW2WRUrhQexIHl0shU/view?usp=sharing)** and put it in scripts/models folder



## Docker Running
1. Launch outside docker ```roslaunch realsense2_camera rs_aligned_depth.launch```
2. go to docker ```sh run_docker.sh```
3. Run segmentation and filtering node ```roslaunch segmentation launch_node.py```


## Environment setup with Anaconda

1. Create anaconda environment: ```conda env create -n environment.yml```
2. download model checkpoint from **[GDrive](https://drive.google.com/file/d/1mrNft0aeIqAggnsW2WRUrhQexIHl0shU/view?usp=sharing)** and put it in scripts/models folder
3. ```conda activate segmentation_ros```
