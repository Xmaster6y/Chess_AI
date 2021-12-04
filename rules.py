#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the rules.
"""
import copy

from board_class import Board

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
	for i in range(8):
		for j in range(8):
			piece_color = board.board[i][j][0] #"_" will not be considered 
			if piece_color == color:
				piece = board.board[i][j][1:]
				if piece == "P":
					moves += _pawn_possible_moves(board, color, (i,j))
				elif piece == "B":
					moves += _bishop_possible_moves(board, color, (i,j))
				elif piece == "R":
					moves += _rook_possible_moves(board, color, (i,j))
				elif piece == "Kn":
					moves += _knight_possible_moves(board, color, (i,j))
				elif piece == "Ki":
					moves += _king_possible_moves(board, color, (i,j))
				elif piece == "Q":
					moves += _queen_possible_moves(board, color, (i,j))
				else:
					raise NotImplementedError
	return moves


def is_check(board, color):
	king_pos = ()
	for i in range(8):
		for j in range(8):
			piece_color = board.board[i][j][0] #"_" will not be considered 
			if (piece_color == color) & (board.board[i][j][1:] == "Ki"): #Hail the king
				king_pos = (i,j)
				break
		if king_pos:
			break
	inv_color = INVERSE_COLOR[color]
	for i in range(8):
		for j in range(8):
			piece_color = board.board[i][j][0] #"_" will not be considered 
			if piece_color == inv_color:
				piece = board.board[i][j][1:]
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
def _pawn_possible_moves(board, color, pos):
	up = UP[color]
	moves = []
	for base_move in BASE_MOVES["P"]:
		end_pos = ((pos[0]+up*base_move[0]), (pos[1]+up*base_move[1]))
		if not in_board(end_pos):
			continue
		if (board.board[end_pos[0]][end_pos[1]][0] == color):
			continue
		if _pawn_valid_move(board, color, (pos,end_pos)):
			move = (pos, end_pos)
			moves.append(move)
	return moves

def _bishop_possible_moves(board, color, pos):
	moves = []
	for base_move in BASE_MOVES["B"]:
		scale = 1
		end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
		if not in_board(end_pos):
			continue
		move = (pos, end_pos)
		while in_board(end_pos):
			if (board.board[end_pos[0]][end_pos[1]][0] == color):
				break
			if _bishop_valid_move(board, color, move):
				moves.append(move)
			if board.board[end_pos[0]][end_pos[1]][0] == INVERSE_COLOR[color]:
				break
			scale += 1
			end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
			if not in_board(end_pos):
				break
			move = (pos, end_pos)
	return moves

def _rook_possible_moves(board, color, pos):
	moves = []
	for base_move in BASE_MOVES["R"]:
		scale = 1
		end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
		if not in_board(end_pos):
			continue
		move = (pos, end_pos)
		while in_board(end_pos):
			if (board.board[end_pos[0]][end_pos[1]][0] == color):
				break
			if _rook_valid_move(board, color, move):
				moves.append(move)
			if board.board[end_pos[0]][end_pos[1]][0] == INVERSE_COLOR[color]:
				break
			scale += 1
			end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
			if not in_board(end_pos):
				break
			move = (pos, end_pos)
	return moves

def _king_possible_moves(board, color, pos):
	moves = []
	for base_move in BASE_MOVES["Ki"]:
		end_pos = ((pos[0]+base_move[0]), (pos[1]+base_move[1]))
		move = (pos, end_pos)
		if not in_board(end_pos):
			continue
		if (board.board[end_pos[0]][end_pos[1]][0] == color):
			continue	
		if _king_valid_move(board, color, move):
			moves.append(move)
	return moves

def _queen_possible_moves(board, color, pos):
	moves = []
	for base_move in BASE_MOVES["Q"]:
		scale = 1
		end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
		if not in_board(end_pos):
			continue
		move = (pos, end_pos)
		while in_board(end_pos):
			if (board.board[end_pos[0]][end_pos[1]][0] == color):
				break
			if _queen_valid_move(board, color, move):
				moves.append(move)
			if board.board[end_pos[0]][end_pos[1]][0] == INVERSE_COLOR[color]:
				break
			scale += 1
			end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
			if not in_board(end_pos):
				break
			move = (pos, end_pos)
	return moves

def _knight_possible_moves(board, color, pos):
	moves = []
	for base_move in BASE_MOVES["Kn"]:
		end_pos = ((pos[0]+base_move[0]), (pos[1]+base_move[1]))
		move = (pos, end_pos)
		if not in_board(end_pos):
			continue
		if (board.board[end_pos[0]][end_pos[1]][0] == color):
			continue
		if _knight_valid_move(board, color, move):
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
	inv_color = INVERSE_COLOR[color]
	for base_move in BASE_MOVES["B"]:
		scale = 1
		end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
		while in_board(end_pos):
			if (board.board[end_pos[0]][end_pos[1]][0] == color):
				break
			if end_pos == pos_threat:
				return True
			if (board.board[end_pos[0]][end_pos[1]][0] == inv_color):
				break
			scale += 1
			end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
	return False

def _rook_threat(board, color, pos, pos_threat):
	inv_color = INVERSE_COLOR[color]
	for base_move in BASE_MOVES["R"]:
		scale = 1
		end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
		while in_board(end_pos):
			if (board.board[end_pos[0]][end_pos[1]][0] == color):
				break
			if end_pos == pos_threat:
				return True
			if (board.board[end_pos[0]][end_pos[1]][0] == inv_color):
				break
			scale += 1
			end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
	return False

def _king_threat(board, color, pos, pos_threat):
	for base_move in BASE_MOVES["Ki"][:-2]:#except rock moves
		end_pos = ((pos[0]+base_move[0]), (pos[1]+base_move[1]))
		if end_pos == pos_threat:
			return True
	return False

def _queen_threat(board, color, pos, pos_threat):
	inv_color = INVERSE_COLOR[color]
	for base_move in BASE_MOVES["Q"]:
		scale = 1
		end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
		while in_board(end_pos):
			if (board.board[end_pos[0]][end_pos[1]][0] == color):
				break
			if end_pos == pos_threat:
				return True
			if (board.board[end_pos[0]][end_pos[1]][0] == inv_color):
				break
			scale += 1
			end_pos = ((pos[0]+scale*base_move[0]), (pos[1]+scale*base_move[1]))
	return False
	
def _knight_threat(board, color, pos, pos_threat):
	for base_move in BASE_MOVES["Kn"][:-2]:
		end_pos = ((pos[0]+base_move[0]), (pos[1]+base_move[1]))
		if end_pos == pos_threat:
			return True
	return False


def is_valid_move(player, board, move):
	pos = move[0]
	color = board.board[pos[0]][pos[1]][0]
	if color == player.color:
		piece = board.board[pos[0]][pos[1]][1:]
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
			if not _pawn_at_start(move[0], color):
				return False
		for pos_x in range(move[0][0]+1,move[1][0]+1):
			if board.board[pos_x][move[0][1]] != "_":
				return False
		return not _will_be_check(board, color, move)
	else:
		inv_color = INVERSE_COLOR[color]
		if board.board[move[1][0]][move[1][1]][0] != inv_color:
			if board.board[move[1][0]][move[1][1]][0] == "_":
				if _in_passing_valid(board, color, move):
					return not _will_be_check(board, color, move)
				else:
					return False
			else:
				return False
		return not _will_be_check(board, color, move)


def _bishop_valid_move(board, color, move):
	return not _will_be_check(board, color, move)

def _rook_valid_move(board, color, move):
	return not _will_be_check(board, color, move)

def _king_valid_move(board, color, move):
	delta_y = move[1][1]-move[0][1]
	if abs(delta_y) == 2:

		sign = delta_y//2
		if sign < 0:
			rock_allowed = (not board.memory[color]["has_left_rook_move"]) & (not board.memory[color]["has_king_move"])
		else:
			rock_allowed = (not board.memory[color]["has_right_rook_move"]) & (not board.memory[color]["has_king_move"])
		if not rock_allowed:
			return False
		scale = 1
		pos = (move[0][0], move[0][1]+sign)
		while board.board[pos[0]][pos[1]] == "_":
			scale += 1
			pos = (move[0][0], move[0][1]+scale*sign)
			if not in_board(pos):
				return False
		if board.board[pos[0]][pos[1]] != color+"R":
			return False
		inter_move = (move[0], (move[1][0],move[1][1]-sign))
		return (not is_check(board, color))&(not _will_be_check(board, color, inter_move))&(not _will_be_check(board, color, move))	
	return not _will_be_check(board, color, move)

def _queen_valid_move(board, color, move):
	return not _will_be_check(board, color, move)
	
def _knight_valid_move(board, color, move):
	return not _will_be_check(board, color, move)

def _apply_move(board, move):#Different from apply_move because here the move is assumed conform with displacement rules
	piece = board.board[move[0][0]][move[0][1]]
	old_piece = board.board[move[1][0]][move[1][1]]
	if piece[1:] == "P":
		delta_y = move[1][1]-move[0][1]
		if (old_piece == "_") & (delta_y != 0):
			up = UP[piece[0]]
			board.board[move[1][0]-up][move[1][1]] = "_" #In passing capture
		if _pawn_at_end(move[1], piece[0]):
			board.board[move[1][0]][move[1][1]] = piece[0] + "Q"
		else:
			board.board[move[1][0]][move[1][1]] = piece
	elif piece[1:] == "Ki":
		delta_y = move[1][1]-move[0][1]
		if abs(delta_y) == 2:
			if delta_y < 0:
				rook = board.board[move[0][0]][0]
				board.board[move[0][0]][0] = "_"
				board.board[move[0][0]][3] = rook
			else:
				rook = board.board[move[0][0]][7]
				board.board[move[0][0]][7] = "_"
				board.board[move[0][0]][5] = rook
		board.board[move[1][0]][move[1][1]] = piece#Move the king anyway
	else:
		board.board[move[1][0]][move[1][1]] = piece
	board.board[move[0][0]][move[0][1]] = "_"
	return old_piece

def _will_be_check(board, color, move):
	board_copy = copy.deepcopy(board)
	_apply_move(board_copy, move)
	check = is_check(board_copy, color)
	return check

def _pawn_at_start(pos, color):
	if (color == "W") & (pos[0] == 1):
		return True
	elif (color == "B") & (pos[0] == 6):
		return True
	return False

def _pawn_at_end(pos, color):
	if (color == "W") & (pos[0] == 7):
		return True
	elif (color == "B") & (pos[0] == 0):
		return True
	return False

def _in_passing_valid(board, color, move):
	inv_color = INVERSE_COLOR[color]
	pos_last_double = board.memory[inv_color]["last_double_pawn"]
	if pos_last_double:
		up = UP[color]
		if pos_last_double == (move[1][0]-up, move[1][1]):
			return True
	return False

def apply_move(player, board, move):
	legit = is_valid_move(player, board, move)
	if legit:
		mat = False
		pat = False
		old_piece = _apply_move(board, move)
	else:
		pat = not bool(possible_moves(board, player.color))
		mat = False
		if pat:
			legit = True
			mat = bool(is_check(board, player.color))
			if mat:
				pat = False
		old_piece = "_"
	return legit, old_piece, mat, pat

if __name__ == '__main__':
	empty = [
		["_",]*8,
		["_",]*8,
		["_",]*8,
		["_",]*8,
		["_",]*8,
		["_",]*8,
		["_",]*8,
		["_",]*8,
	]
	empty_board = Board()
	empty_board.board = empty

	#Displacement tests
	pos = (4,4)
	print("Pawn's moves:")
	empty_board.board[pos[0]][pos[1]] = "WP"
	moves = possible_moves(empty_board, "W")
	print(moves)
	print(f"Total number of moves : {len(moves)}")
	print("")
	print("King's moves:")
	empty_board.board[pos[0]][pos[1]] = "WKi"
	moves  = possible_moves(empty_board, "W")
	print(moves)
	print(f"Total number of moves : {len(moves)}")
	print("")
	print("Knight's moves:")
	empty_board.board[pos[0]][pos[1]] = "WKn"
	moves = possible_moves(empty_board, "W")
	print(moves)
	print(f"Total number of moves : {len(moves)}")
	print("")
	print("Bishop's moves:")
	empty_board.board[pos[0]][pos[1]] = "WB"
	moves = possible_moves(empty_board, "W")
	print(moves)
	print(f"Total number of moves : {len(moves)}")
	print("")
	print("Queen's moves:")
	empty_board.board[pos[0]][pos[1]] = "WQ"
	moves = possible_moves(empty_board, "W")
	print(moves)
	print(f"Total number of moves : {len(moves)}")
	print("")
	print("Rook's moves:")
	empty_board.board[pos[0]][pos[1]] = "WR"
	moves = possible_moves(empty_board, "W")
	print(moves)
	print(f"Total number of moves : {len(moves)}")
	print("")

	# Rock
	print("Rock's moves:")
	empty_board.board[pos[0]][pos[1]] = "_"
	empty_board.board[0][4] = "WKi"
	empty_board.board[0][0] = "WR"
	empty_board.board[0][7] = "WR"
	moves = _king_possible_moves(empty_board, "W", (0,4))
	print(moves)
	print(f"Total number of moves : {len(moves)}")
	print("")

	# In passing
	print("Pawn's moves (In passing):")
	empty_board.board[0][4] = "_"
	empty_board.board[0][0] = "_"
	empty_board.board[0][7] = "_"
	empty_board.board[4][4] = "WP"
	empty_board.board[4][5] = "BP"
	empty_board.memory["B"]["last_double_pawn"] = (4,5)
	moves = _pawn_possible_moves(empty_board, "W", (4,4))
	print(moves)
	print(f"Total number of moves : {len(moves)}")

