FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Timezone Configuration
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    apt-utils \
    net-tools \
    mesa-utils \
    gnupg2 \
    wget \
    curl \
    git \
    mc \
    nano \
    cmake \
    gcc \
    cmake-curses-gui \
    build-essential \
    python3.8 \
    && rm -rf /var/lib/apt/lists/*


ENV DEBIAN_FRONTEND=noninteractive

# ROS install
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -
RUN apt-get update && apt-get install -y \
    ros-noetic-robot \
    ros-noetic-rosconsole \
    ros-noetic-realsense2-camera\
    ros-noetic-pcl-ros \
    ros-noetic-image-pipeline
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc


# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
# ENV PATH="/usr/bin/python3.8:${PATH}"
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

RUN apt-get update && apt-get install -y \
    python3-opencv ca-certificates python3-dev git wget sudo ninja-build

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py


RUN pip3 install tensorboard cmake   # cmake from apt-get is too old
# RUN pip3 install Pillow==8.2.0 numpy==1.18.5 torch==1.8.1 torchvision==0.9.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
RUN pip3 install Pillow numpy torch torchvision -f https://download.pytorch.org/whl/cu113/torch_stable.html --upgrade
# RUN pip3 install Pillow numpy --upgrade

RUN pip3 install 'git+https://github.com/facebookresearch/fvcore' opencv-python==4.5.2.54

# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip3 install -e detectron2_repo

RUN pip install pandas rospkg scipy pytimedinput

# install cv_bridge for python3
RUN apt-get install libpcl-dev -y

RUN apt-get update && apt-get install -y python3-catkin-tools python3-dev libopencv-dev
EXPOSE 11311

RUN pip3 install netifaces shapely torchfile opencv-python pyfastnoisesimd rapidfuzz && \
    export ROS_HOSTNAME=localhost 

RUN mkdir -p /cv_bridge_ws/src && \
    cd /cv_bridge_ws/src && \
    git clone https://github.com/IvDmNe/vision_opencv.git && \
    cd /cv_bridge_ws && \
    catkin config \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DPYTHON_INCLUDE_DIR=/usr/include/python3.8 \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so \
    -DCMAKE_BUILD_TYPE=Release \
    -DSETUPTOOLS_DEB_LAYOUT=OFF \
    -Drosconsole_DIR=/opt/ros/noetic/share/rosconsole/cmake \
    -Drostime_DIR=/opt/ros/noetic/share/rostime/cmake \
    -Droscpp_traits_DIR=/opt/ros/noetic/share/roscpp_traits/cmake \
    -Dstd_msgs_DIR=/opt/ros/noetic/share/std_msgs/cmake \
    -Droscpp_serialization_DIR=/opt/ros/noetic/share/roscpp_serialization/cmake \
    -Dmessage_runtime_DIR=/opt/ros/noetic/share/message_runtime/cmake \
    -Dgeometry_msgs_DIR=/opt/ros/noetic/share/geometry_msgs/cmake \
    -Dsensor_msgs_DIR=/opt/ros/noetic/share/sensor_msgs/cmake \
    -Dcpp_common_DIR=/opt/ros/noetic/share/cpp_common/cmake && \
    cd src && git clone https://github.com/ros/catkin.git &&  cd .. && \
    catkin config --install && \
    catkin build cv_bridge && \
    echo "source /cv_bridge_ws/devel/setup.bash --extend" >> ~/.bashrc





# RUN git clone https://github.com/IvDmNe/unseen_object_segmentation_with_knn_classification.git&&\
# cd unseen_object_segmentation_with_knn_classification &&\
# git checkout -b ros_wrapper


# Install MMCV
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# Install MMDetection
# RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
# WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r mmdetection/requirements/build.txt &&\
    pip install --no-cache-dir -e mmdetection

RUN pip install faiss-cpu

RUN echo "source /ws/devel/setup.bash --extend" >> ~/.bashrc

# RUN git clone https://github.com/IvDmNe/uoais.git && \
#     cd uoais && \
#     mkdir output && \
#     python setup.py build develop 

