#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environnement implementation
"""
import gym
from gym import spaces

from board_class import Board
import rules
import policy

MAX_STEPS = 200

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()    
        self.action_space = spaces.Box(low=0, high=1, shape=
                        (8, 8, 2), dtype=np.uint8)   # Example for using image as input:

        self.observation_space = spaces.Box(low=0, high=255, shape=
                        (8, 8, 1), dtype=np.uint8)
        self.board = Board()
        self.turn = "W"

    def step(self, action):
        self._take_action(action)  
        self.current_step += 1  
        
        reward = policy.reward_function(legit_move, mat_next, piece_cap_value, piece_be_cap_value, pat)
        done = (not legit) | mat_next | pat
        obs = self.board.convert_to_array()
        return obs, reward, done, {}

    def reset(self):
        self.board.reset()
        self.turn = "W"
        return self.board.convert_to_array()

    def render(self, mode='human', close=False):
        print(self.board)

if __name__ == '__main__':
  pass