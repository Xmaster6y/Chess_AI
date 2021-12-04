#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for the player.
"""
import numpy as np

class Player():

	__init__(self, is_AI, color, policy):
		self.is_AI = is_AI
		self.color = color
		self.policy = policy

	def choose_move(self, board_state):
		return self.policy(self, board_state)


if __name__ == '__main__':
	pass