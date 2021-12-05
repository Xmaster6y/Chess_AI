#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Xmaster a.k.a. Yoann Poupart
Class to setup the board
"""

import numpy as np
import matplotlib.pyplot as plt


class Board():
	values = {
		"_":0, "P":1, "B":3, "Kn":3, "R":5, "Q":9, "Ki": 127.5
	}
	value_scale = 10

	def __init__(self):
		self.board = [
			["WR", "WKn", "WB", "WQ", "WKi", "WB", "WKn", "WR"],
			["WP",]*8,
			["_",]*8,
			["_",]*8,
			["_",]*8,
			["_",]*8,
			["BP",]*8,
			["BR", "BKn", "BB", "BQ", "BKi", "BB", "BKn", "BR"],
		]
		self.memory = {
			"W":{
				"has_king_move" : False,
				"has_left_rook_move" : False,
				"has_right_rook_move" : False,
				"last_double_pawn" : (),
			},
			"B":{
				"has_king_move" : False,
				"has_left_rook_move" : False,
				"has_right_rook_move" : False,
				"last_double_pawn" : (),
			}
		}
		self.turn = "W"

	def reset(self):
		self.__init__()

	def convert_to_array(self):
		array_board = np.ones((8,8))*127.5
		for i in range(8):
			for j in range(8):
				piece = self.board[i][j]
				if piece != "_":
					color = piece[0]
					value = type(self).values[piece[1:]] * type(self).value_scale
					scaled_value = min(value,127.5)
					array_board[i,j] += (1-2*(color=="B")) * scaled_value
		return array_board

	def __str__(self):
		s = "  " + "-"*32 + "\n"
		for i in range(8):
			s += f"{8-i} |" + "|".join(map(lambda s:s.ljust(3), self.board[7-i])) + "|\n"
			s += "  " + "-"*32 + "\n"
		s += "    " + " | ".join(["a","b", "c", "d", "e", "f", "g","h",])
		return s
	def plot(self):
		array = self.convert_to_array()
		plt.imshow(array, cmap='Greys')
		plt.show()

			

if __name__ == '__main__':
	board = Board()
	print(board)	
	print(board.convert_to_array())
	board.plot()

