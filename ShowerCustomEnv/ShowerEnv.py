import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import math

class ShowerEnv(gym.Env):
    """
    Custom Shower Water Temperature Environment
    """
    def __init__(self, T_hot, T_cold, max_angle_change, T_desired):
        # set from variables
        self.T_hot = T_hot
        self.T_cold = T_cold
        self.max_angle_change = max_angle_change
        self.T_desired = T_desired

        # fix the cold flow state
        self.cold_flow = 50
    
    def reset(self):
        self.angle_hot = 20
        self.state = self.__calculate_temp()
        return np.array([self.state])

    def step(self, action:int):
        # action can be 0 - 20
        angle_change = (action - 10) * 10
        self.angle_hot = np.clip(self.angle_hot + angle_change, 0, 100) 
        # remember angle_hot is 1:1 flow_hot so flow hot is not defined
        self.state = self.__calculate_temp()

        # determine the reward
        error = 3 
        reward = self.__calculate_reward(error)

        if (self.state - self.T_desired)>= 1.5*error:
            done = True
        else:
            done = False

        return np.array([self.state]), reward, done

    
    def __calculate_reward(self, error:int):
        comparison = math.fabs(self.state - self.T_desired)

        if comparison <= error:
            return 10
        elif comparison <= 2 * error:
            return -1 
        elif comparison <= 3 * error:
            return -5
        elif comparison > 3 * error:
            return -10


    def __calculate_temp(self):
        """
        Calculates the tempurature of the shower flow
        """
        # remember angle_hot is 1:1 with flow_hot
        output_temp = ( self.angle_hot*self.T_hot + self.cold_flow*self.T_cold )\
           / ( self.angle_hot + self.cold_flow )
        
        return output_temp
