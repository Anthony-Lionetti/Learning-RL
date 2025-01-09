import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


class Actions(Enum):
    left = 0
    right = 1
    up = 2
    down = 3


class GridEnv(gym.Env):
    """
    Custom Grid environment
    """

    def __init__(self, size:int = 5):
        self.size = size
        self.state = (0,0)
        self.total_reward = 0

    def reset(self): 
        self.state = (0,0)
        self.total_reward = 0
    
    def step(self, action:Actions):
        max_size = self.size - 1 


        # Agent takes an action, update state and reward accordingly
        if action == 0 and self.state[0] > 0:
            self.state = (self.state[0] - 1, self.state[1])
        elif action == 1 and self.state[0] < self.size-1:
            self.state = (self.state[0] + 1, self.state[1])
        elif action == 2 and self.state[1] > 0:
            self.state = (self.state[0], self.state[1] - 1)
        elif action == 3 and self.state[1] < self.size-1:
            self.state = (self.state[0], self.state[1] + 1)
        
        # reward or punish based on the state
        if self.state[0] == max_size and self.state[1] == max_size:
            reward = 50
        elif self.state[0] == max_size and self.state[1] == 0:
            reward = -50
        else:
            reward = -1
        
        # return false if done
        
        done = (self.state == (max_size, 0) or self.state == (max_size, max_size))

        return self.state, reward, done



