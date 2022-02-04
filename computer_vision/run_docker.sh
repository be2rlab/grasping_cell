#! /bin/bash

sudo docker run \
    --net host \
    --gpus all \
    --rm \
    # -v ~/Nenakhov/segm_ros_ws:/ws \
    -v ~/Nenakhov/segm_ros_ws:/ws \
    -v /dev:/dev \
    -it \
    --privileged \
    ivan/iiwa_cv 