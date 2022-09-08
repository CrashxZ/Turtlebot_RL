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
import time
from SensorEnv import NeedySensor


env=NeedySensor()
env.reset()

episodes = 50
state = env.reset()
for episode in range(1, episodes+1):
    #state = env.reset()
    done = False
    score = 0 
    
    while not done:
        env.render()
        
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()