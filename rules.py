#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the rules.
"""

BASE_MOVES = {
	"P":[(1,0),(2,0),(1,1),(1,-1)], "B":[(1,1), (-1,1), (1,-1), (-1,-1),], "R":[(0,1), (0,-1), (1,0), (-1,0),],
	"Ki":[(1,1), (-1,1), (1,-1), (-1,-1),(0,1), (0,-1), (1,0), (-1,0),(0,2), (0,-2),],
	"Q":[(1,1), (-1,1), (1,-1), (-1,-1),(0,1), (0,-1), (1,0), (-1,0),],
	"Kn":[(1,2), (-1,2), (1,-2), (-1,-2), (2,1), (-2,1), (2,-1), (-2,-1),],
}
INVERSE_COLOR = {"B":"W", "W":"B"}
UP = {"W":1, "B":-1}


def possible_moves(board, color):
	moves = []
	check = is_check(board, color)
	for i in range(8):
		for j in range(8):
			piece_color = board[i][j][0] #"_" will not be considered 
			if piece_color == color:
				piece = board[i][j][1:]
				if piece == "P":
					moves += _pawn_possible_move(board, color, (i,j), check)
				elif piece == "B":
					moves += _bishop_possible_move(board, color, (i,j), check)
				elif piece == "R":
					moves += _rook_possible_move(board, color, (i,j), check)
				elif piece == "Kn":
					moves += _knight_possible_move(board, color, (i,j), check)
				elif piece == "Ki":
					moves += _king_possible_move(board, color, (i,j), check)
				elif piece == "Q":
					moves += _queen_possible_move(board, color, (i,j), check)
				else:
					raise NotImplementedError
	if moves:
		mat = False
	else:
		mat = check
	return moves, mat


def is_check(board, color):
	king pos = ()
	for i in range(8):
		for j in range(8):
			piece_color = board[i][j][0] #"_" will not be considered 
			if (piece_color == color) & (board[i][j][1:] == "Ki"): #Hail the king
				king_pos = (i,j)
				break
		if king_pos:
			break
	inv_color = INVERSE_COLOR[color]
	for i in range(8):
		for j in range(8):
			piece_color = board[i][j][0] #"_" will not be considered 
			if piece_color == inv_color:
				piece = board[i][j][1:]
				if piece == "P":
					if _pawn_threat(board, inv_color, (i,j), king_pos):
						return True
				elif piece == "B":
					if _bishop_threat(board, inv_color, (i,j), king_pos):
						return True
				elif piece == "R":
					if _rook_threat(board, inv_color, (i,j), king_pos):
						return True
				elif piece == "Kn":
					if _knight_threat(board, inv_color, (i,j), king_pos):
						return True
				elif piece == "Ki":
					if _king_threat(board, inv_color, (i,j), king_pos):
						return True
				elif piece == "Q":
					if _queen_threat(board, inv_color, (i,j), king_pos):
						return True
				else:
					raise NotImplementedError
	return False

def in_board(pos):
	if (pos[0] < 0) | (pos[0] > 7) | (pos[1] < 0) | (pos[1] > 7):
		return False
	return True			

#Beware when calling the following functions, no test is run in these
def _pawn_possible_moves(board, color, pos, check):
	up = UP[color]
	moves = []
	for base_move in BASE_MOVES["P"]:
		end_pos = ((pos[0]+up*base_move[0]), (pos[1]+up*base_move[1]))
		if _pawn_valid_move(board, color, (pos,end_pos)):
			moves.append(move)
	return moves

def _bishop_possible_moves(board, color, pos, check):
	moves = []
	for base_move in BASE_MOVES["B"]:
		scale = 1
		end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
		move = (pos, end_pos)
		while in_board(end_pos) & board[end_pos[0]][end_pos[1]][0] != color:
			if _bishop_valid_move(board, color, move):
				moves.append(move)
			if board[end_pos[0]][end_pos[1]][0] == INVERSE_COLOR[color]:
				break
			scale += 1
			end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
			move = (pos, end_pos)
	return moves

def _rook_possible_moves(board, color, pos, check):
	moves = []
	for base_move in BASE_MOVES["R"]:
		scale = 1
		end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
		move = (pos, end_pos)
		while in_board(end_pos) & board[end_pos[0]][end_pos[1]][0] != color:
			if _rook_valid_move(board, color, move):
				moves.append(move)
			if board[end_pos[0]][end_pos[1]][0] == INVERSE_COLOR[color]:
				break
			scale += 1
			end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
			move = (pos, end_pos)
	return moves

def _king_possible_moves(board, color, pos, check):
	moves = []
	for base_move in BASE_MOVES["Ki"]:
		scale = 1
		end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
		move = (pos, end_pos)
		while in_board(end_pos) & board[end_pos[0]][end_pos[1]][0] != color:
			if _king_valid_move(board, color, move):
				moves.append(move)
			if board[end_pos[0]][end_pos[1]][0] == INVERSE_COLOR[color]:
				break
			scale += 1
			end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
			move = (pos, end_pos)
	return moves

def _queen_possible_moves(board, color, pos, check):
	moves = []
	for base_move in BASE_MOVES["Q"]:
		scale = 1
		end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
		move = (pos, end_pos)
		while in_board(end_pos) & board[end_pos[0]][end_pos[1]][0] != color:
			if _queen_valid_move(board, color, move):
				moves.append(move)
			if board[end_pos[0]][end_pos[1]][0] == INVERSE_COLOR[color]:
				break
			scale += 1
			end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
			move = (pos, end_pos)
	return moves

def _knight_possible_moves(board, color, pos, check):
	moves = []
	for base_move in BASE_MOVES["Kn"]:
		end_pos = ((pos[0]+base_move[0]), (pos[1]+base_move[1]))
		if _knight_valid_move(board, color, (pos,end_pos)):
			moves.append(move)
	return moves


def _pawn_threat(board, color, pos, pos_threat):#pos threatened
	up = UP[color]
	for base_move in BASE_MOVES["P"][2:]:
		end_pos = ((pos[0]+up*base_move[0]), (pos[1]+up*base_move[1]))
		if end_pos == pos_threat:
			return True
	return False

def _bishop_threat(board, color, pos, pos_threat):
	for base_move in BASE_MOVES["B"]:
		scale = 1
		end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
		while in_board(end_pos) & board[end_pos[0]][end_pos[1]][0] != color:
			if end_pos == pos_threat:
				return True
			scale += 1
			end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
	return False

def _rook_threat(board, color, pos, pos_threat):
	for base_move in BASE_MOVES["R"]:
		scale = 1
		end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
		while in_board(end_pos) & board[end_pos[0]][end_pos[1]][0] != color:
			if end_pos == pos_threat:
				return True
			scale += 1
			end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
	return False

def _king_threat(board, color, pos, pos_threat):
	for base_move in BASE_MOVES["Ki"]:
		end_pos = ((pos[0]+base_move[0]), (pos[1]+base_move[1]))
		if end_pos == pos_threat:
			return True
	return False

def _queen_threat(board, color, pos, pos_threat):
	for base_move in BASE_MOVES["Q"]:
		scale = 1
		end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
		while in_board(end_pos) & board[end_pos[0]][end_pos[1]][0] != color:
			if end_pos == pos_threat:
				return True
			scale += 1
			end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
	return False
	
def _knight_threat(board, color, pos, pos_threat):
	for base_move in BASE_MOVES["Kn"]:
		end_pos = ((pos[0]+base_move[0]), (pos[1]+base_move[1]))
		if end_pos == pos_threat:
			return True
	return False


def is_valid_move(player, board, move):
	pos = move[0]
	color = board[pos[0]][pos[1]][0]
	if color == player.color:
		check = is_check(board, color)
		piece = board[pos[0]][pos[1]][1:]
		if piece == "P":
			if move in _pawn_possible_moves(board, color, pos):
				return True
		elif piece == "B":
			if move in _bishop_possible_moves(board, color, pos):
				return True
		elif piece == "R":
			if move in  _rook_possible_moves(board, color, pos):
				return True
		elif piece == "Kn":
			if move in _knight_possible_moves(board, color, pos):
				return True
		elif piece == "Ki":
			if move in _king_possible_moves(board, color, pos):
				return True
		elif piece == "Q":
			if move in _queen_possible_moves(board, color, pos):
				return True
		else:
			raise NotImplementedError
	return False

#Beware when calling the following functions, no test is run in these (move assumed conform)
def _pawn_valid_move(board, color, move):
	delta = (move[1][0]-move[0][0], move[1][1]-move[0][1])
	up = UP[color]
	if delta[1] == 0:
		if delta[0] == 2*up:
			if !_pawn_at_sart(move[0], color):
				return False

		for pos_x in range(move[0][0]+1,move[1][0]+1):
			if board[pos_x][move[0][1]] != "_":
				return False
		return !_will_be_check(board, color, move)
	else:


def _bishop_valid_move(board, color, move):
	pass

def _rook_valid_move(board, color, move):
	pass

def _king_valid_move(board, color, move):
	pass

def _queen_valid_move(board, color, move):
	pass
	
def _knight_valid_move(board, color, move):
	pass

def _apply_move(board, move):#Different from apply_move because here the move is assumed conform with displacement rules
	piece = board[move[0][0]][move[0][1]]
	old_piece = board[move[1][0]][move[1][1]]
	board[move[1][0]][move[1][1]] = piece
	board[move[0][0]][move[0][1]] = "_"
	return old_piece

def _will_be_check(board, color, move):
	_apply_move(board, move)
	check = is_check(board, color)
	_apply_move(board, move[::-1])
	return check

def _pawn_at_start(pos, color):
	if (color == "W") & (pos[0] == 1):
		return True
	elif (color == "B") & (pos[0] == 6):
		return True
	return False

if __name__ == '__main__':
	empty_board = [
		["_",]*8,
		["_",]*8,
		["_",]*8,
		["_",]*8
		["_",]*8,
		["_",]*8,
		["_",]*8,
		["_",]*8,
	]
	pos = (4,4)
	empty_board[pos[0]][pos[1]] = "WP"
	print(possible_moves(empty_board, "W"))