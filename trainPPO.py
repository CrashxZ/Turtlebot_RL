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

from SensorEnv import NeedySensor


env=NeedySensor()
env.reset()
model_name=NeedySensor.model_name
model_path = NeedySensor.model_path
model = PPO("MlpPolicy", env, verbose=1,tensorboard_log=NeedySensor.log_path)
model.learn(total_timesteps=450000)
model.save(model_path)

# episodes = 5
# state = env.reset()
# for episode in range(1, episodes+1):
#     #state = env.reset()
#     done = False
#     score = 0 
    
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score+=reward
#     print('Episode:{} Score:{}'.format(episode, score))
# env.close()
