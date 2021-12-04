#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for the player.
"""
import numpy as np

import rules

class Player():

	__init__(self, is_AI, color, policy):
		self.is_AI = is_AI
		self.color = color
		self.policy = policy

	def choose_move(self, board_state):
		return slef.policy(self, board_state)

def random_policy(player, board):
	moves = rules.possible_moves(board, player.color)
	return np.random.choice(moves)

def human_policy(player, board):
	if player.is_AI:
		return random_policy(player, board)
	while True:
		try:
			print(board)
			move_str = input("Give your move (exp:e4 e5)\n").split()
			move = (
					(int(move_str[0][1])-1, ord(move_str[0][0])-97), #The column index is the letter
					(int(move_str[1][1])-1, ord(move_str[1][0])-97),
				)
			assert rules.is_valid_move(player, board, move)
		except:
			print("Invalid move or format!")

	return np.random.choice(moves)

if __name__ == '__main__':
	pass