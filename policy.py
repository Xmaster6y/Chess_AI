#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Policy implementation
"""
import numpy as np
import random
import keras

import rules

def random_policy(player, board):
	moves = rules.possible_moves(board, player.color)
	if moves:
		return random.choice(moves)
	else:
		return ((0,0), (0,0))

def human_policy(player, board):
	if player.is_AI:
		return random_policy(player, board)
	while True:
		try:
			move = ((),())
			print(board)
			move_str = input("Give your move (exp:e2 e3)\n").split()
			move = (
					(int(move_str[0][1])-1, ord(move_str[0][0])-97), #The column index is the letter
					(int(move_str[1][1])-1, ord(move_str[1][0])-97),
				)
			assert rules.is_valid_move(player, board, move)
			break
		except KeyboardInterrupt:
			print("Let's stop for now")
			raise SystemExit
		except Exception as e:
			print(e)
			print(f"{move_str}->{move} is an invalid move or format!")
	return move

def create_rl_policy(path_file):
	def rl_policy(player, board):
		try:
			model = keras.models.load_model(path_file)
		except OSError:
			model = create_model()
		observation = board.convert_to_array()
		if player.color == "B": #Always learn to do white moves
			observation = 255 - observation
		result = model(observation)
		init_pos = tuple(np.argmax(result[:,:,0]))
		end_pos = tuple(np.argmax(result[:,:,1]))
		return (init_pos, end_pos)
	return rl_policy

def reward_function(legit_move, mat_next, piece_cap_value, piece_be_cap_value, pat):
	##If not a legit move -> -1000
	##If lost after -> -100
	##If lost after -> -50
	##Nothing -> 0
	##Piece captured -> piece_value
	##Piece being captured -> -piece_value
	return -1000*(not legit_move) - 100*mat_next - 50*pat + piece_cap_value - piece_be_cap_value

if __name__ == '__main__':
	pass