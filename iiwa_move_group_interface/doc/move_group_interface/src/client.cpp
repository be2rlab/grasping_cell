#include <ros/ros.h>
#include "iiwa_move_group_interface/GoalPoses.h"
#include "geometry_msgs/Pose.h"

void SetTarget(geometry_msgs::Pose &target_pose, double Ort_w, double Ort_x, double Ort_y,
               double Ort_z, double Pos_x, double Pos_y, double Pos_z) {
    target_pose.orientation.w = Ort_w;
    target_pose.orientation.x = Ort_x;
    target_pose.orientation.y = Ort_y;
    target_pose.orientation.z = Ort_z;
    target_pose.position.x = Pos_x;
    target_pose.position.y = Pos_y;
    target_pose.position.z = Pos_z;
}

int main(int argc, char **argv){

    ros::init(argc, argv, "move_group_interface_iiwa");
    ros::NodeHandle nh;
    ros::ServiceClient ClientAction
    = nh.serviceClient<iiwa_move_group_interface::GoalPoses>("/GoalPoses");

    ROS_INFO_STREAM("Service Created");

    //while (ros::ok())
    //{
    ROS_INFO_STREAM("*************");
    ROS_INFO_STREAM("I am looping");
    iiwa_move_group_interface::GoalPoses srv;

    std::vector<geometry_msgs::Pose> Poses;
    geometry_msgs::Pose pose;
    // srv.request.poses = poses;
    //int Goal;
    SetTarget(pose, 1.0, 0.0, 0.0, 0.0, -0.3, -0.3, -0.5);
    Poses.push_back(pose);
    SetTarget(pose, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.6);
    Poses.push_back(pose);
    SetTarget(pose, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.6);
    Poses.push_back(pose);
    SetTarget(pose, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.6);
    Poses.push_back(pose);
    SetTarget(pose, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.6);
    Poses.push_back(pose);
    srv.request.Poses = Poses;
    srv.request.GoalBoxIndex = 2;
    srv.request.WorkingMode = 2;
    ROS_INFO_STREAM("Oh yeah, it is a new pose");

    ClientAction.waitForExistence();
    if (ClientAction.call(srv))
    {
       ROS_INFO_STREAM("Let's check");
       if(srv.response.AllMotionWorked)
           ROS_INFO_STREAM("We are awesome, it is success");
       else
           ROS_INFO_STREAM("It is Fail, but Do not worry, we will fix it");
    }

    // }
    return 0;

}
