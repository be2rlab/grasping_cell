#! /bin/bash

sudo docker run \
    --net host \
    --gpus all \
    --rm \
    -v ~/Nenakhov/sandbox_ws:/ws \
    -v /dev:/dev \
    -it \
    --privileged \
    ivan/iiwa_cv 