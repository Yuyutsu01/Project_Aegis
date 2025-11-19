import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnvironment(gym.Env):
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000):
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_capital = initial_capital
        self.current_step = 0
        
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(10,))
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.capital = self.initial_capital
        self.shares = 0
        self.total_value = self.initial_capital
        return self._get_observation()
    
    def _get_observation(self):
        if self.current_step >= len(self.data):
            return np.zeros(10)
        
        return np.random.random(10)  # Mock observation
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Mock reward
        reward = np.random.normal(0, 0.01)
        
        return self._get_observation(), reward, done, {}