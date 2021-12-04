#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File to play against AI
"""

from board_class import Board
from player import Player
import rules
import policy

PLAYER1_FILE = "deep_policies/player1"
PLAYER2_FILE = "deep_policies/player2"
MAX_STEPS = 200


if __name__ == '__main__':
    human_color = input("Choose a color : (W/B)\n")
    print(f"You chose '{human_color}'!")
    ag_ai = input("Do you want to play against Deep Chess AI ? (Y/N)\n")
    if ag_ai == "N":
        if human_color == "W":
            player1 = Player(False, "W", policy.human_policy)
            player2 = Player(True, "B", policy.random_policy)
        else:
            player2 = Player(False, "B", policy.human_policy)
            player1 = Player(True, "W", policy.random_policy)
    else:
        if human_color == "W":
            try:
                ai_policy = policy.create_rl_policy(PLAYER2_FILE)
            except:
                print("AI not trained")
                raise SystemExit
            player1 = Player(False, "W", policy.human_policy)
            player2 = Player(True, "B", ai_policy)
        else:
            try:
                ai_policy = policy.create_rl_policy(PLAYER1_FILE)
            except:
                print("AI not trained")
                raise SystemExit
            player2 = Player(False, "B", policy.human_policy)
            player1 = Player(True, "W", ai_policy)

    done = False
    PLAYERS = {
        "W":player1,
        "B":player2,
    }
    board = Board()
    step = 1
    while (not done) & (step < MAX_STEPS):
        player = PLAYERS[board.turn]

        move = player.choose_move(board)
        print(f"Move chosen : {move}")
        legit, _, mat, pat = rules.apply_move(player, board, move)

        board.turn = rules.INVERSE_COLOR[board.turn]
        done = (not legit) | mat | pat
        step += 1

    if mat:
        if PLAYERS[board.turn].is_AI:
            print("You lost to AI!")
        else:
            print("Human superiority!")
    elif pat:
        print("Nobody wins...")
    elif (not legit):
        if PLAYERS[board.turn].is_AI:
            print("AI don't even know the rules!")
        else:
            print("Can't you learn the rules!")
    else:
        print("Too slow! Max number of steps reached")

