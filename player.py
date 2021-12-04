#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for the player.
"""
import numpy as np

class Player():

	def __init__(self, is_AI, color, policy, is_policy_trainable=False):
		self.is_AI = is_AI
		self.color = color
		self.policy = policy
		self.is_policy_trainable = is_policy_trainable

	def choose_move(self, board):
		return self.policy(self, board)


if __name__ == '__main__':
	pass