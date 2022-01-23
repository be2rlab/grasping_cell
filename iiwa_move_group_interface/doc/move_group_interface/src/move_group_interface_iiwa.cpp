#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/CollisionObject.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <map>
#include <string>
#include <sstream>
#include <cmath>
#include <vector>
#include <unistd.h>
#include "iiwa_move_group_interface/GoalPoses.h"
#include "uhvat_ros_driver/SetGripperState.h"
#include <chrono>
#include <thread>

#include <tf/transform_listener.h>
#include <ros/ros.h>



static const std::string PLANNING_GROUP = "manipulator";

std::vector<std::pair<std::string, std::vector<double>>> read_csv(std::string filename)
{
    std::vector<std::pair<std::string, std::vector<double>>> result;
    std::ifstream myFile(filename);
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");
    std::string line, colname;
    double val;

    if(myFile.good())
    {
        std::getline(myFile, line);
        std::stringstream ss(line);
        while(std::getline(ss, colname, ',')){
            result.push_back({colname, std::vector<double> {}});
        }
    }

    while(std::getline(myFile, line))
    {
        std::stringstream ss(line);
        int colIdx = 0;

        while(ss >> val){
            result.at(colIdx).second.push_back(val);
            if(ss.peek() == ',') ss.ignore();
            colIdx++;
        }
    }
    myFile.close();

    return result;
}

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

class demo
{
private:
    ros::NodeHandle node_handle;
    int state;
    moveit::planning_interface::MoveGroupInterface *move_group;
    moveit_visual_tools::MoveItVisualTools *visual_tools;
    moveit::planning_interface::PlanningSceneInterface *planning_scene_interface;
    std::string PlannerId;
    std::string PlannerType;
    std::map <std::string, std::string> MapParam;
    ros::ServiceServer service;
    ros::ServiceClient GripperClient = node_handle.serviceClient<uhvat_ros_driver::SetGripperState>("/gripper_state");

    std::vector <moveit_msgs::CollisionObject> collision_objects;
    std::vector<moveit_msgs::ObjectColor> object_colors;


public:
    demo() : node_handle("~")
    {
        move_group = new moveit::planning_interface::MoveGroupInterface(PLANNING_GROUP);
        planning_scene_interface = new moveit::planning_interface::PlanningSceneInterface;
        visual_tools = new moveit_visual_tools::MoveItVisualTools("iiwa_link_0");

        PlannerId = "SBL";
        PlannerType = "geometric::SBL";
        move_group->setPlannerId(PlannerId);
        move_group->setPlanningTime(10.0);
        MapParam = move_group->getPlannerParams(PlannerId, PLANNING_GROUP);
        MapParam["range"] = "1.24";
        MapParam["type"] = PlannerType;
        MapParam["projection_evaluator"] = "joints(iiwa_joint_1,iiwa_joint_2)";
        move_group->setPlannerParams(PlannerId, PLANNING_GROUP, MapParam);
        move_group->setMaxAccelerationScalingFactor(0.5);
        move_group->setMaxVelocityScalingFactor(0.5);
        move_group->setEndEffectorLink("iiwa_link_ee_grasp");
        state = 0;
    }

    void Start()
    {
        // attach tool
        visual_tools->prompt("Press 'next' to move to correct start pose ...");
        geometry_msgs::Pose target_pose;
        /* from current to pregrasp */
        PosIndexToCoordinates(target_pose, 0);
        bool start_success = PlanAndExec(target_pose);
        while (!start_success)
            start_success = PlanAndExec(target_pose);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        service = node_handle.advertiseService("/GoalPoses", &demo::HandleCallback, this);
    }

    bool HandleCallback(iiwa_move_group_interface::GoalPoses::Request &req,
                          iiwa_move_group_interface::GoalPoses::Response &res)
    {
        if (req.WorkingMode == 0)
        {
            HandleGrasping(req, res);
            return res.AllMotionWorked;
        }
        else if (req.WorkingMode == 1)
        {
            HandleLearningProcess(req, res);
            return res.AllMotionWorked;
        }
        else if (req.WorkingMode == 2)
        {
            HandleChangingView(req, res);
            return res.AllMotionWorked;
        }
        else if (req.WorkingMode == 3)
        {
            HandleChangingPose(req, res);
            return res.AllMotionWorked;
        }
        else if (req.WorkingMode == 4)
        {
            HandleReturningToInitialPose(req, res);
            return res.AllMotionWorked;
        }
        else
            return false;
    }

    bool HandleLearningProcess(iiwa_move_group_interface::GoalPoses::Request &req,
                              iiwa_move_group_interface::GoalPoses::Response &res)
    {
        /* the first part of the task */
        int k = 0;
        geometry_msgs::Pose target_pose;
        // geometry_msgs::Pose pre_target_pose;
        visual_tools->prompt("Press 'next' to go to grasp pose ...");
        target_pose = req.Poses[k];
        // pre_target_pose = target_pose;
        // pre_target_pose.position.z -= 0.2;
        // Transfrom(target_pose, "iiwa_link_0", "iiwa_link_camera", "iiwa_link_ee");
        std::cout<<"attempt: "<<k<<std::endl;
        bool grasp_success =  ExecuteTraj(target_pose);
        while(!grasp_success)
        {
            k++;
            target_pose = req.Poses[k];
            // Transfrom(target_pose, "iiwa_link_0", "iiwa_link_camera", "iiwa_link_ee");
            std::cout<<"attempt: "<<k<<std::endl;
            grasp_success = ExecuteTraj(target_pose);
            if (k >= req.Poses.size() - 1) break;
        }
        if (!grasp_success)
        {
            res.GraspWorked = grasp_success;
            res.AllMotionWorked = false;
            ROS_INFO_STREAM("Grasp failed, I am at start");
            return grasp_success;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        ROS_INFO_STREAM("Attaching Object");
        // attach object
        ROS_INFO_STREAM("Closing the Gripper");
        // close the gripper

        /* the second part of the task */
        visual_tools->prompt("Press 'next' to go to plane ...");
        // correct the coordinates, this must be looped, consider putting the fixed poses in the constructor or in that function
        PosIndexToCoordinates(target_pose, 6);
        bool plane_success = PlanAndExec(target_pose);
        if (!plane_success)
            ROS_INFO_STREAM("Plane failed, I am waiting for an action");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        ROS_INFO_STREAM("Opening the Gripper");
        // open the gripper
        // close gripper


        res.GraspWorked = grasp_success;
        res.AllMotionWorked = plane_success;


        return plane_success;
    }

    bool HandleChangingView(iiwa_move_group_interface::GoalPoses::Request &req,
                            iiwa_move_group_interface::GoalPoses::Response &res)
    {
        geometry_msgs::Pose target_pose;
        visual_tools->prompt("Press 'next' to go to next view point ...");
        target_pose = req.Poses[0];
//        target_pose.position.x = 0.1;
//        target_pose.position.y = 0.1;
//        target_pose.position.z = 0.1;
        std::cout<<target_pose.position.x<<std::endl;
        std::cout<<target_pose.position.y<<std::endl;
        std::cout<<target_pose.position.z<<std::endl;
        std::cout<<"******************************************"<<std::endl;
        Transfrom(target_pose, "iiwa_link_0", "iiwa_link_ee_camera", "iiwa_link_ee");
        AddObject("iiwa_link_0", 0.0, 1.0, 0.0, "temp", 0.05, 0.05, 0.05,
                  target_pose.position.x, target_pose.position.y, target_pose.position.z, 0.0, 0.0, 0.0, 1.0, false);
        std::cout<<target_pose.position.x<<std::endl;
        std::cout<<target_pose.position.y<<std::endl;
        std::cout<<target_pose.position.z<<std::endl;
        visual_tools->prompt("Press 'next' to go to next view point ...");
        RemoveObj("temp");
        // move_group->setEndEffectorLink("iiwa_link_ee");
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        bool view_success =  ExecuteTraj(target_pose);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        OpenGripper(3);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        visual_tools->prompt("Press 'next' to go ...");
        PosIndexToCoordinates(target_pose, 0);
        bool start_success = PlanAndExec(target_pose);
        res.AllMotionWorked = view_success;
        return view_success;
//        res.AllMotionWorked = false;
//        res.GraspWorked = false;
//        return false;
    }

    bool HandleChangingPose(iiwa_move_group_interface::GoalPoses::Request &req,
                            iiwa_move_group_interface::GoalPoses::Response &res)
    {
        geometry_msgs::Pose target_pose;
        visual_tools->prompt("Press 'next' to go to required pose ...");
        target_pose = req.Poses[0];
        Transfrom(target_pose, "iiwa_link_0", "iiwa_link_ee_camera", "iiwa_link_ee");
        bool pose_success =  PlanAndExec(target_pose);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        res.AllMotionWorked = pose_success;
        return pose_success;
    }

    bool HandleReturningToInitialPose(iiwa_move_group_interface::GoalPoses::Request &req,
                            iiwa_move_group_interface::GoalPoses::Response &res)
    {
        geometry_msgs::Pose target_pose;
        visual_tools->prompt("Press 'next' to go to initial point ...");
        // needs to be corrected consider in constructor or in function
        PosIndexToCoordinates(target_pose, 0);
        bool initial_success =  PlanAndExec(target_pose);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // make previous pregrasp and following with a line
        res.AllMotionWorked = initial_success;
        return initial_success;
    }

    bool HandleGrasping(iiwa_move_group_interface::GoalPoses::Request &req,
                        iiwa_move_group_interface::GoalPoses::Response &res)
    {
        /* the first part of the task */
        int k = 0;
        geometry_msgs::Pose target_pose;
        visual_tools->prompt("Press 'next' to go to grasp pose ...");
        target_pose = req.Poses[k];
        Transfrom(target_pose, "iiwa_link_0", "iiwa_link_ee_camera", "iiwa_link_ee");
        std::cout<<"attempt: "<<k<<std::endl;
        bool grasp_success =  ExecuteTraj(target_pose);
        while(!grasp_success)
        {
            k++;
            target_pose = req.Poses[k];
            Transfrom(target_pose, "iiwa_link_0", "iiwa_link_ee_camera", "iiwa_link_ee");
            std::cout<<"attempt: "<<k<<std::endl;
            grasp_success = ExecuteTraj(target_pose);
            if (k >= req.Poses.size() - 1) break;
        }
        if (!grasp_success)
        {
            res.GraspWorked = grasp_success;
            res.AllMotionWorked = false;
            ROS_INFO_STREAM("Grasp failed, I am at start");
            return grasp_success;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        ROS_INFO_STREAM("Closing the Gripper");
        // close the gripper

        ROS_INFO_STREAM("Attaching Object");
        // attach object

        /* the second part of the task */
        visual_tools->prompt("Press 'next' to go to goal box ...");
        // can be in the constructor
        PosIndexToCoordinates(target_pose, req.GoalBoxIndex);
        bool box_goal_successs = PlanAndExec(target_pose);
        if (!box_goal_successs)
            ROS_INFO_STREAM("Goal box failed, I am waiting for an action");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // open Gripper
        // close Gripper

        res.GraspWorked = grasp_success;
        res.AllMotionWorked = box_goal_successs;


        return box_goal_successs;
    }

    void Transfrom(geometry_msgs::Pose &target_pose, std::string BaseFrame, std::string CurrentFrame, std::string EE_Frame)
    {
        tf::TransformListener listener_;
        tf::StampedTransform FromCurrentToEE;
        /* transform from CurrentFrame to ee */
        listener_.waitForTransform(CurrentFrame, EE_Frame, ros::Time(0), ros::Duration(1.0));
        listener_.lookupTransform(CurrentFrame, EE_Frame, ros::Time(0), FromCurrentToEE);

        /* transform from BaseFrame to ee */
        tf::StampedTransform FromBaseToEE;
        listener_.waitForTransform(BaseFrame, CurrentFrame, ros::Time(0), ros::Duration(1.0));
        listener_.lookupTransform(BaseFrame, CurrentFrame, ros::Time(0), FromBaseToEE);

        tf::Transform RecievedPosition;
        tf::poseMsgToTF(target_pose, RecievedPosition);
        tf::Transform Result;
        Result = FromBaseToEE * RecievedPosition;

        tf::poseTFToMsg(Result, target_pose);
    }

    bool ExecuteTraj(geometry_msgs::Pose target_pose)
    {
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        move_group->setPoseTarget(target_pose);
        bool plan_success = (move_group->plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
        if (!plan_success)
            return plan_success;
        else
        {
            std::vector<geometry_msgs::Pose> waypoints;
            waypoints.push_back(target_pose);
            moveit_msgs::RobotTrajectory trajectory;
            const double jump_threshold = 0.0;
            const double eef_step = 0.01;
            move_group->computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);

            my_plan.trajectory_ = trajectory;
            bool exec_success = (move_group->execute(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
            return exec_success;
        }
    }

    void OpenGripper(int state)
    {
        visual_tools->prompt("Press 'next' to move uhvat ...");
        uhvat_ros_driver::SetGripperState srv;
        // GripperClient.waitForExistence();
        srv.request.state = state;
        GripperClient.call(srv);
    }

    void Test()
    {
        int k = 1;
        while (k < 100)
        {
            k++;
            OpenGripper(0);
            // move_group->setEndEffectorLink("iiwa_link_ee_camera");
            geometry_msgs::Pose RecievedPose;
            visual_tools->prompt("Press 'next' to go to initial ");
            PosIndexToCoordinates(RecievedPose, 0);
            PlanAndExec(RecievedPose);
            visual_tools->prompt("Press 'next' to go to next pos ");
            PosIndexToCoordinates(RecievedPose, 7);
            PlanAndExec(RecievedPose);
//            visual_tools->prompt("Press 'next' to go to next pos ");
//            PosIndexToCoordinates(RecievedPose, 8);
//            PlanAndExec(RecievedPose);
        }
    }

    bool PlanAndExec(geometry_msgs::Pose target_pose)
    {
        // visual_tools->prompt("Press 'next' to plan");
        move_group->setPoseTarget(target_pose);
        moveit::planning_interface::MoveGroupInterface::Plan my_plan;
        bool success = (move_group->plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

        visual_tools->prompt("Press 'next' to execute");
        bool exec_success = false;
        if (success)
            exec_success = (move_group->execute(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
        return exec_success;
    }

    void PosIndexToCoordinates(geometry_msgs::Pose &target_pose, int index)
    {
        if (index == 0)
            SetTarget(target_pose, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.6);
        if (index == 1)
            SetTarget(target_pose, 0.0, 0.0, 1.0, 0.0, 0.0, -0.5, 0.3);
        if (index == 2)
            SetTarget(target_pose, 0.0, 0.0, 1.0, 0.0, 0.0, -0.5, 0.3);
        if (index == 3)
            SetTarget(target_pose, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.3);
        if (index == 4)
            SetTarget(target_pose, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.3);
        if (index == 5)
            SetTarget(target_pose, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.3);
        if (index == 6)
            SetTarget(target_pose, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.2);
        if (index == 7)
            SetTarget(target_pose, 0.0, 0.0, -0.707, 0.707, 0.5, 0.0, 0.5);
        if (index == 8)
            SetTarget(target_pose, 0.0, 1.0, 0.0, 0.0, 0.5, 0.3, 0.4);
        if (index == 9)
            SetTarget(target_pose, 0.0, 0.0, 1.0, 0.0, 0.4, -0.3, 0.24);
    }

    void AddObject(std::string ObjFrame, double r, double g, double b, std::string ObjId, double x, double y, double z,
                   double X, double Y, double Z, double Ort_x, double Ort_y, double Ort_z, double Ort_w, bool attach){

        moveit_msgs::CollisionObject collision_object;
        moveit_msgs::ObjectColor object_color;

        collision_object.header.frame_id = ObjFrame;

        collision_object.id = ObjId;

        shape_msgs::SolidPrimitive primitive;
        primitive.type = primitive.BOX;
        primitive.dimensions.resize(3);
        primitive.dimensions[0] = x;
        primitive.dimensions[1] = y;
        primitive.dimensions[2] = z;

        geometry_msgs::Pose box_pose;
        box_pose.orientation.w = Ort_w;
        box_pose.orientation.x = Ort_x;
        box_pose.orientation.y = Ort_y;
        box_pose.orientation.z = Ort_z;
        box_pose.position.x = X;
        box_pose.position.y = Y;
        box_pose.position.z = Z;


        collision_object.primitives.push_back(primitive);
        collision_object.primitive_poses.push_back(box_pose);
        collision_object.operation = collision_object.ADD;

        object_color.id = ObjId;
        object_color.color.a = 1;
        object_color.color.r = r;
        object_color.color.g = g;
        object_color.color.b = b;

        collision_objects.push_back(collision_object);
        object_colors.push_back(object_color);
        planning_scene_interface->addCollisionObjects(collision_objects, object_colors);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (attach)
            move_group->attachObject(collision_object.id);
    }

    void AddEnv()
    {
        visual_tools->prompt("Press 'next' to add objects");
        std::vector<std::pair<std::string, std::vector<double>>> env = read_csv("/home/abdul/Moveit_Grasping/src/iiwa_stack/iiwa_move_group_interface/doc/move_group_interface/src/environments/env4.txt");
        std::vector<std::string> bar_id{"wall1", "base", "wall2", "table", "box1", "box2", "box3", "box4", "box5",
                                        "obs1", "obs2", "obs3", "obs4", "obs5",
                                        "box-goal1", "box-goal2", "box-goal3", "box-goal4", "box-goal5"};

//        std::vector<std::string> bar_id{"wall1", "base", "wall2", "table", "box1", "box2", "box3", "box4", "box5",
//                                        "obs1", "obs2", "obs3", "obs4", "obs5",
//                                        "box-goal1", "box-goal2", "box-goal3", "box-goal4", "box-goal5"};

//        std::vector<std::string> bar_id{"wall1", "base", "wall2", "table", "box1", "box2", "box3", "box4", "box5",
//                                        "obs1", "obs2", "obs3", "obs4", "obs5"};

        for (int i=0; i<bar_id.size(); i++)
        {
            AddObject("iiwa_link_0", env.at(0).second.at(i), env.at(1).second.at(i),
                      env.at(2).second.at(i), bar_id.at(i), env.at(3).second.at(i),
                      env.at(4).second.at(i), env.at(5).second.at(i), env.at(6).second.at(i),
                      env.at(7).second.at(i), env.at(8).second.at(i), 0, 0, -0.7071068, 0.7071068, false);
        }
    }

    void AttachToolAndCamera()
    {
        visual_tools->prompt("Press 'next' to add Obj");
        AddObject("iiwa_link_ee", 0.5, 0.0, 0.5,
                  "tool",
                  0.12, 0.05, 0.08,
                  0.0, 0.0, -0.129,
                  0.0, 0.0, 0.0, 1.0,
                  true);
        AddObject("iiwa_link_ee", 0.5, 0.0, 0.5,
                  "tool-extension",
                  0.12, 0.05, 0.1,
                  0.0, 0.0,-0.04,
                  0.0, 0.0, 0.0, 1.0,
                  true);
        AddObject("iiwa_link_camera", 0.5, 0.0, 0.5,
                  "camera",
                  0.12, 0.065, 0.025,
                  0.0, 0.025,0.0,
                  0.0, 0.0, 0.0, 1.0,
                  true);
    }

    void AttachObj()
    {
        visual_tools->prompt("Press 'next' to add Obj");
        AddObject("iiwa_link_0", 0.5, 0.0, 0.5,
                  "object",
                  0.1, 0.1, 0.1,
                  0.0, 0.0, 2.0,
                  0.0, 0.0, 0.0, 1.0,
                  true);
        AddObject("iiwa_link_ee", 0.5, 0.0, 0.5,
                  "object",
                  0.06, 0.22, 0.06,
                  0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 1.0,
                  true);
    }

    void RemoveObj(std::string ObjId)
    {
        move_group->detachObject(ObjId);
        std::vector<std::string> object_ids;
        object_ids.push_back(ObjId);
        planning_scene_interface->removeCollisionObjects(object_ids);
    }

};


int main(int argc, char **argv) {

    ros::init(argc, argv, "move_group_interface_iiwa");
    ros::AsyncSpinner spinner(8);
    spinner.start();

    demo go;
    // std::cout<<"w 5raaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"<<std::endl;
    go.AddEnv();
    // go.Start();
    // go.AttachToolAndCamera();
    // go.AttachObj();
    // go.RemoveObj("tool-extension");
    go.Test();

    ros::waitForShutdown();
    ros::shutdown();
    return 0;
}
