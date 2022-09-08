#!/usr/bin/env python3
import gym 
import threading
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import random
import os
from stable_baselines3 import PPO, DQN , A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import csv
import time
from datetime import datetime
from SensorEnv import NeedySensor

import rospy
from geometry_msgs.msg import Pose
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from std_msgs.msg import String
score = 0

#ros init
rospy.init_node('Reinforcement_Agent_Invoker', anonymous=True)
target = Pose()

target_goal = rospy.Publisher("/waypoint_pose", Pose , queue_size=1)



#RL Stuff

env=NeedySensor()
# csv_log = open(NeedySensor.model_name+".csv", "w")
# logger = csv.writer(csv_log, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
# logger.writerow(['Episode','Action','Distance_Travelled','Total_Distance','Dead_Sensors','Reward'])
# log_dict = csv.DictWriter(csv_log, fieldnames=['Episode','Action','Distance_Travelled','Total_Distance','Dead_Sensors','Reward'])
env.reset()
model_name=NeedySensor.model_name
model_path = NeedySensor.model_path
model = A2C.load(path=model_path, env=env)


obs = env.reset()
#env.display_map()
#time.sleep(10)
start_time = datetime.now()
csv_log = open("critical_sensors_EDF.csv", "w")
logger = csv.writer(csv_log, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
logger.writerow(['Time','#Critical_Sensors','Distance_Travelled'])

def get_min_index(array):
    min = array[0]
    min_index = 0
    for i in range(len(array)):
        if array[i] < min:
            min = array[i]
            min_index = i
    return min_index

action = -1
#ros Stuff
def listener_callback(msg):
    global action
    logger.writerow([datetime.now()-start_time,env.get_critical_sensor_count(),env.get_distance_traveled()])
    if msg:
        if msg.data == "0":
            pass
        elif msg.data == "1":
            print("Robot reached the goal")
            if action != -1:
                env.charge(action)
            print("Predicting next goal")
            #action, _states = model.predict(obs)
            action = get_min_index(obs[:15])
            print("Next Goal: ", action)
            target.position.x = env.sensor_positions[action][0]
            target.position.y = env.sensor_positions[action][1]
            target.orientation.w = 1.0
            target_goal.publish(target)
            env.render()
            n_state, reward, done, info = env.step(action)
            time.sleep(2)
            #print('Score:{}'.format(score))
            #score+=reward








#charge_flag = 0
req_listener_sub = rospy.Subscriber("/request_next", String ,listener_callback, queue_size=1)



while not rospy.is_shutdown():
    env.decay()
    env.render()
    print("Finishing Objectives")
    time.sleep(0.5)


rospy.spin()



