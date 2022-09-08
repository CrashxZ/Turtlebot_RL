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
from datetime import datetime
import math
from scipy.spatial import distance
import time
import csv

class NeedySensor(Env):
    model_name = "prototype_scale_15" #prototype_scale_15
    model_path = "Training/Models/"+model_name
    log_path = "Training/Logs/"+model_name
    #logger = csv.writer(csv_log, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #logger.writerow(['Episode','Action','Distance Travelled','Reward'])
    sens_pos = []
    def __init__(self):
        self.sensor_count = 15
        self.critical_level = 10 #percent charge
        self.warning_level = 20  #percent charge
        self.pose_x = 0
        self.pose_y = 0
        self.map_bound = 30
        self.current_position=(self.pose_x,self.pose_y)
        self.sensor_positions = []
        self.total_distance = 0
        self.dead_sensors = 0
        self.previous_action = -1
        self.min = -1
        self.minbatt = -1
        self.navigation_targets = []

        with open('waypoints2.txt', mode='r') as waypoints_file:
            waypoints = csv.reader(waypoints_file, delimiter=',')
            for row in waypoints:
                if(row[3]=="1"):
                    self.sensor_positions.append([float(row[0]),float(row[1])])
                    print('Loading trajectory waypoint - > {}'.format(row))
        
       
        #self.sensor_positions = sensor_positions
        now = datetime.now()
        
        self.action_space = Discrete(self.sensor_count)
        self.observation_space = Box(0,100,shape=(self.sensor_count*2,))
        self.state = np.empty([self.sensor_count,2]).flatten('F')
        self.P_list = np.zeros(self.sensor_count)
        self.episodes = 100
        

    def get_sensor_count(self):
        return self.sensor_count

    def add_nav_targets(self,x,y,w,s):
        self.navigation_targets.append([x,y,w,s])
    
    def get_state(self):
        return self.state
    
    def decay(self):
        for i in range(self.sensor_count):
            self.state[i] = self.state[i] - random.random()*0.2 #*0.1 for gazebo #8
            

    def decay_actual(self):
        for i in range(self.sensor_count):
            self.state[i] = self.state[i] - random.random()*0.83
    
    def distance(self,a,b):
        return distance.euclidean(a, b)
    
    def update_distance(self):
        for i in range(self.sensor_count):
            self.state[self.sensor_count+i] = self.distance(self.current_position,self.sensor_positions[i])
        
    def get_critical_sensor_count(self):
        count = 0
        for i in range(self.sensor_count):
            if self.state[i] < self.critical_level:
                count += 1
        return count
        
    def get_distance_traveled(self):
        return self.total_distance
        

    def step(self, action): 
        self.episodes -= 1
        self.decay()
        r = 0
        min_dist = self.state[self.sensor_count+action]
        min_pos = action
        lowest_batt = self.state[action]
        lowest_batt_pos = action
        for i in range(self.sensor_count):
            #get minimum distance
            if self.state[self.sensor_count+i] < min_dist and self.previous_action != i:
                min_dist = self.state[self.sensor_count+i]
                min_pos = i
            #get lowest battery
            if self.state[i] < lowest_batt:
                lowest_batt = self.state[i]
                lowest_batt_pos = i
            if self.state[i] < self.warning_level:
                r -= abs(((100 - self.state[i])/100))
            if self.state[i] < self.critical_level:
                self.dead_sensors += 1
                done = True
        self.minbatt = lowest_batt_pos
        self.min = min_pos
        #reward calculation
        if self.state[action] > 80:
            r -= 2 



            # if action != i:
            #     if self.state[i] > 60:
            #         r += 0
            #     elif self.state[i] > self.warning_level:
            #         r += self.state[i]/100
            #     elif self.state[i] < self.warning_level and self.state[i] > self.critical_level:
            #         r -= ((100 - self.state[i])/100)*2
            #     elif self.state[i] < self.critical_level:
            #         r -= 2

        # if action == self.previous_action+1 or action == self.previous_action-1:
        #     r += 10

        #no repetation
        if self.previous_action == action:
            r -= 1
            
        if self.state[action] >= self.warning_level:
            r += 1

        if self.state[action] <= 0:
            self.state[action] = 0

        #reward for lowest battery
        if action == lowest_batt_pos:
            r += 2
        else:
            r -= 1.5
        
        #reward for lowest distance
        if self.previous_action == min_pos and self.state[self.sensor_count+action] != 0:
            r += 2.5
        else:
            r -= 1.5
        
        self.previous_action = action
        


        
        #Uncoment for training and testing(without gazebo)
        #self.charge(action)

        reward = r

        self.P_list = np.insert(self.P_list,0,action)
        self.P_list = self.P_list[:-1]
        

        #NeedySensor.logger.writerow([self.episodes,action,distance_travelled,reward])
        


        if self.episodes <= 0: 
            done = True
        else:
            done = False

        distance_travelled = self.state[action + self.sensor_count]
        self.total_distance += distance_travelled
        info = {'Episode':self.episodes,'Action':action,'Distance_Travelled':distance_travelled,'Total_Distance':self.total_distance,'Dead_Sensors':self.dead_sensors,'Reward':reward}
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        self.display_battery()
        pass
    
    def reset(self):
        self.state.fill(60)
        self.episodes = 100
        self.update_distance()
        return self.state
    
    def display_battery(self):
        for i in range(self.sensor_count):
            print("Sensor - ",i,end=" -> |")
            for n in range(50):
                if(n <= int(self.state[i]/2)):
                    print("-",end="")
                else:
                    print(" ",end="")
            print("| - ",self.state[i],"%" , "Distance:" , self.state[self.sensor_count+i])
        
        print("")
        print("")
        print("Path -> ",self.P_list)
        print("min - " , self.min)
        print("min batt- " , self.minbatt)
        
    
    def check_cell(self,i,j):
        not_clear = False
        for k in range(10):
            if i == int(self.sensor_positions[k][0]) and j == int(self.sensor_positions[k][1]):
                not_clear = True
        return not_clear
        
    def charge(self,sensor_index):
        self.state[sensor_index] = 100
        self.current_position = self.sensor_positions[sensor_index]
        self.pose_x=self.sensor_positions[sensor_index][0]
        self.pose_y=self.sensor_positions[sensor_index][1]
        self.update_distance()
        
    def display_map(self):
        os.system('clear')
        #print(self.pose_x, self.pose_y)
        for i in range(self.map_bound):
            print("|",end="")
            for j in range(self.map_bound):
                #update robot location
                if self.pose_x == i and self.pose_y == j:
                    print(" \u0394 ",end="")
                elif self.check_cell(i,j):
                    print(" S ",end="")
                else:
                    print("   ",end="")
            print("|")
        print("")
        print("")
        self.display_battery()
        print("")
        print("")
        print("Path -> ",self.P_list)
                    
                
                
