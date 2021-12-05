#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xmaster a.k.a. Yoann Poupart
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

    def __init__(self, player1, player2):
        super(CustomEnv, self).__init__()    
        self.action_space = spaces.Box(low=0, high=1, shape=
                        (8, 8, 2), dtype=np.uint8)   # Example for using image as input:

        self.observation_space = spaces.Box(low=0, high=255, shape=
                        (8, 8, 1), dtype=np.uint8)
        self.board = Board()
        self.players = {"W":player1, "B":player2}

    def step(self, action):
        init_pos = np.unravel_index(np.argmax(action[:,:,0]), (8,8))
        end_pos = np.unravel_index(np.argmax(action[:,:,1]), (8,8))
        move = (init_pos, end_pos)

        legit, old_piece, mat, pat = rules.apply_move(self.players[board.turn], board, move)
        reward = policy.reward_function(legit_move, mat, piece_cap_value, piece_be_cap_value, pat)
        done = (not legit) | mat | pat
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