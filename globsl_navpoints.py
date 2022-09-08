#!/usr/bin/env python3
from numpy.lib.polynomial import poly
import rospy
import actionlib
import csv
import time
#from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import numpy as np
import math
import json
from ast import literal_eval
from geometry_msgs.msg import PoseWithCovariance, PointStamped , Twist ,PoseArray, Pose
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import time
import cv2
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib


  
    
def add_nav_targets(x,y,w,s):
    navigation_targets.append({"x": x, "y": y, "w":w, "charge_flag":s})



def go_to_waypoint(goal,i):
    server_up = move_base_client.wait_for_server()
    if server_up:
        move_base_client.send_goal(goal)
        result = move_base_client.wait_for_result()
        server_check = move_base_client.wait_for_server(timeout=rospy.Duration(10.0))
        if move_base_client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            print("Goal reached")
            i+=1
            print("Next waypoint :", i)
            load_next_wp(i)
        
    else:
        print("Cannot send goal to move_base. Server down")
            

def load_next_wp(i):
    next_wp = Pose()
    next_wp.position.x = navigation_targets[i]["x"]
    next_wp.position.y = navigation_targets[i]["y"]
    next_wp.orientation.w = 1.0
    waypoint_goal.target_pose.pose = next_wp
    waypoint_goal.target_pose.header.stamp = rospy.Time.now()
    waypoint_goal.target_pose.header.frame_id = 'map'
    go_to_waypoint(waypoint_goal,i)






if __name__ == '__main__':
    rospy.init_node('waypoint_follower')
    waypoint_goal = MoveBaseGoal()
    move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    move_base_client.wait_for_server()
    result = 0

    #add nav targets
    navigation_targets = []

    with open('waypoints.txt', mode='r') as waypoints_file:
        waypoints = csv.reader(waypoints_file, delimiter=',')
        for row in waypoints:
            add_nav_targets(float(row[0]),float(row[1]),float(row[2]),float(row[3]))
            print('Loading trajectory waypoint - > {}'.format(row))

    iterations = len(navigation_targets)
    print("Total waypoints : {} . Loading Complete!".format(iterations))
    print("Starting Navigation ...")

    load_next_wp(0)
    rospy.spin()
