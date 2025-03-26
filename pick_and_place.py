#!/usr/bin/env python

import rospy
import time
import tf2_ros
import math

from geometry_msgs.msg import Pose, TransformStamped
from std_msgs.msg import Float64MultiArray
from rmp2_ros.srv import goal,goalResponse
from ur_ros_driver.srv import SetGripper, SetGripperRequest
from rmp2_ros.srv import SetCollRadii, SetCollRadiiRequest
from rmp2_ros.msg import Radii


class PickPlace:

    def __init__(self):
        
        rospy.init_node("PickPlace")
        grasp_pose = rospy.Subscriber ("/result",Float64MultiArray,self.get_grasp_pose,queue_size=1)

    
    def get_grasp_pose (self,msg) : 
        
        pos_x = msg [0]
        pos_y = msg [1]
        pos_z = msg [2]
        rot_z = msg [3]
        
        q_x = 0.0 
        q_y = 0.0
        q_z = math.sin(rot_z / 2)
        q_w = math.cos(rot_z / 2)
        

    def compare_pose_with_transform(pose):
    
        global tf_buffer
        # Wait for the transform between "base_link" and "tcp_link"
        while not rospy.is_shutdown():
            try:
                transform_stamped = tf_buffer.lookup_transform("base_link", "tcp_link", rospy.Time(), rospy.Duration(1.0))
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rospy.logwarn("Failed to lookup transform between 'base_link' and 'tcp_link'. Retrying...")

        # Calculate the distance between the original pose and the transformed pose
        distance = pose_distance(pose, transform_stamped)
        print(distance)
        return distance

    def array_to_pose(array):
        if len(array) != 7:
            print("Error: Input array must have 7 elements")
            return None

        pose = Pose()
        pose.position.x = array[0]
        pose.position.y = array[1]
        pose.position.z = array[2]
        pose.orientation.x = array[3]
        pose.orientation.y = array[4]
        pose.orientation.z = array[5]
        pose.orientation.w = array[6]

        return pose

    def send_pose_to_rmp_goal(pose):
        rospy.wait_for_service('set_rmp_goal')
        try:
            set_rmp_goal = rospy.ServiceProxy('set_rmp_goal', goal)
            resp = set_rmp_goal(pose)
            return resp
        except rospy.ServiceException as e:
            print("Service call failed:", e)

    def send_gripper_request(position, speed = 100, force = 100):
        rospy.wait_for_service('/ur_hardware_interface/robotiq/set_gripper')
        try:
            set_gripper = rospy.ServiceProxy('/ur_hardware_interface/robotiq/set_gripper', SetGripper)
            req = SetGripperRequest()
            req.position_unit = 0
            req.position = position
            req.speed = speed
            req.force = force
            req.asynchronous = 0
            resp = set_gripper(req)
            return resp
        except rospy.ServiceException as e:
            print("Service call failed:", e)

    def move_to_pose(pose):
        resp_rmp = send_pose_to_rmp_goal(pose)
        while(compare_pose_with_transform(pose)>0.02):
            time.sleep(0.2)

    def add_radius(name,radius,srv):
        msg = Radii()
        msg.joint = name
        msg.radius = radius
        # msg.interpolation_pts = int_points
        srv.radii.append(msg)

    def send_srv(srv):
        rospy.wait_for_service('/set_coll_radii')
        try:
            print(srv)
            set_rmp_goal = rospy.ServiceProxy('/set_coll_radii', SetCollRadii)
            resp = set_rmp_goal(srv)
            return resp
        except rospy.ServiceException as e:
        print("Service call failed:", e)

if __name__ == '__main__':
    rospy.init_node('pick_and_place_example')

     # Initialize TF2 listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    srv_1 = SetCollRadiiRequest()
    add_radius("tcp_joint",-1,srv_1)
    add_radius("hande_joint",-1,srv_1)
    add_radius("hande_base_joint",-1,srv_1)
    add_radius("wrist_3_joint",-1,srv_1)
    add_radius("wrist_2_joint",-1,srv_1)
    add_radius("shoulder_lift_joint",-1,srv_1)
    add_radius("elbow_joint",-1,srv_1)

    srv_2 = SetCollRadiiRequest()
    add_radius("tcp_joint",.05,srv_2)
    add_radius("hande_joint",.05,srv_2)
    add_radius("hande_base_joint",.07,srv_2)
    add_radius("wrist_3_joint",.1,srv_2)
    add_radius("wrist_2_joint",.1,srv_2)
    add_radius("shoulder_lift_joint",.12,srv_2)
    add_radius("elbow_joint",.12,srv_2)

    srv_3 = SetCollRadiiRequest()
    add_radius("hande_joint",.15,srv_3)
    add_radius("hande_base_joint",.15,srv_3)
    add_radius("wrist_3_joint",.15,srv_3)
    add_radius("wrist_2_joint",.15,srv_3)
    add_radius("shoulder_lift_joint",.15,srv_3)
    add_radius("elbow_joint",.15,srv_3)

    while(not rospy.is_shutdown()):
        ## Movement for Pick and Place
        # send_srv(srv_2)
        # move_to_pose(pose_1A)
        # send_srv(srv_1)
        # move_to_pose(pose_1CA)
        # rospy.sleep(1)
        # move_to_pose(pose_1)
        # rospy.sleep(1)
        # resp_gripper = send_gripper_request(100)
        # move_to_pose(pose_1A)
        # send_srv(srv_2)
        # move_to_pose(pose_2A)
        # send_srv(srv_1)
        # move_to_pose(pose_2CA)
        # rospy.sleep(1)
        # move_to_pose(pose_2)
        # rospy.sleep(3)
        # resp_gripper = send_gripper_request(0)
        # move_to_pose(pose_2A)

        ## Movement for Pick and Place with coll avoidance
        # move_to_pose(pose_5A)
        # send_srv(srv_2)
        # move_to_pose(pose_2A)
        # send_srv(srv_1)
        # move_to_pose(pose_2CA)
        # rospy.sleep(1)
        # move_to_pose(pose_2)
        # rospy.sleep(1)
        # resp_gripper = send_gripper_request(0)
        # move_to_pose(pose_2A)
        # send_srv(srv_2)
        # move_to_pose(pose_5A)
        # move_to_pose(pose_2A)
        # send_srv(srv_1)
        # move_to_pose(pose_2CA)
        # rospy.sleep(1)
        # move_to_pose(pose_2)
        # rospy.sleep(3)
        # resp_gripper = send_gripper_request(100)
        # move_to_pose(pose_2A)
        # send_srv(srv_2)
        # move_to_pose(pose_5A)

        ## Movement for reaktive control Showcase
        send_srv(srv_2)
        move_to_pose(pose_4A)
        move_to_pose(pose_5A)
