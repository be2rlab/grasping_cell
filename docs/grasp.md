# Pose estimation

[Contact-GraspNet](https://github.com/NVlabs/contact_graspnet) is used as the main grabber configuration algorithm. 

The goal of the Contact-GraspNet method is to generate various configurations for grabbing an object without collision from a scene's point cloud using segmentation. The main advantages of this method are the absence of the need to know the data about the capture object and the absence of the assumption that the captures are always perpendicular to the surface. It is assumed that at least one of the possible surface contacts is visible before capture. 

The task of learning to define a grip with 6 degrees of freedom is reduced to estimating the rotation of a grip with 3 degrees of freedom. The capture view is shown in the figure: 

![grasp1](images/grasp1.png)

Unlike axis-angle representations, there are no ambiguities or breaks in this rotation representation. The reduced dimensionality greatly speeds up the learning process compared to methods that estimate capture poses in an unbounded space.
**IMPORTANT!** For the algorithm to work correctly and apply to all objects on the scene, object segmentation is required. 

[Official repository with this module](https://github.com/deyakovleva/contact_graspnet/tree/dev_ros)