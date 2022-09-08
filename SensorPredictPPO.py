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
from SensorEnv2 import NeedySensor


env=NeedySensor()

csv_log = open(NeedySensor.model_name+".csv", "w")
logger = csv.writer(csv_log, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
logger.writerow(['Episode','Action','Distance_Travelled','Total_Distance','Dead_Sensors','Reward'])
log_dict = csv.DictWriter(csv_log, fieldnames=['Episode','Action','Distance_Travelled','Total_Distance','Dead_Sensors','Reward'])
env.reset()
model_name=NeedySensor.model_name
model_path = NeedySensor.model_path
model = PPO.load(path=model_path, env=env)

episodes = 100
obs = env.reset()
env.display_map()
time.sleep(10)
for episode in range(1, episodes+1):
    #state = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action, _states = model.predict(obs)
        n_state, reward, done, info = env.step(action)
        time.sleep(0.5)
        log_dict.writerow(info)
        score+=reward
        print('Episode:{} Score:{}'.format(episode, score))
env.close()
