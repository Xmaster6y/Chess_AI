#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Xmaster a.k.a. Yoann Poupart
Policy implementation
"""

import numpy as np
import random

from keras.models import Sequential, Model, load_model, save_model
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import LocallyConnected2D
from keras.utils.vis_utils import plot_model

import rules

PLAYER1_FILE = "deep_policies/player1"

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

def create_rl_policy(path_file, is_policy_trainable=False):
    try:
        model = load_model(path_file)
    except OSError:
        model = create_model()
    def rl_policy(player, board):
        observation = board.convert_to_array()
        if player.color == "B": #Always learn to do white moves
            observation = 255 - observation
        result = model(np.expand_dims(observation, axis=0))# Rescale is done in the model
        init_pos = np.unravel_index(np.argmax(result[:,:,0]), (8,8))
        end_pos = np.unravel_index(np.argmax(result[:,:,1]), (8,8))
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



def create_model(w_1=16, w_2=24, w_3=32, w_4=16, w_end=8, s_2=1, s_3=1, s_4=1, d_2=2, d_3=8, d_4=4, act="relu", 
                            alpha=0.2, drop_rate=0.3, conv_drop_rate=0, batch_norm=True, name="player"):
    model = Sequential(name=name)
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Flatten())

    model.add(Dense(w_1**2))
    if act == "relu":
        model.add(Activation("relu"))
    else:
        model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(drop_rate))
    model.add(Reshape((w_1, w_1, 1)))

    ker_size = w_2 - (w_1 - 1) * s_2
    model.add(Conv2DTranspose(d_2, (ker_size, ker_size)))
    if act == "relu":
        model.add(Activation("relu"))
    else:
        model.add(LeakyReLU(alpha=alpha))

    ker_size = w_3 - (w_2 - 1) * s_3
    model.add(Conv2DTranspose(d_3, (ker_size, ker_size)))
    if act == "relu":
        model.add(Activation("relu"))
    else:
        model.add(LeakyReLU(alpha=alpha))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(conv_drop_rate))

    ker_size = w_3 - (w_4 - 1) * s_4
    model.add(Conv2D(d_4, (ker_size, ker_size)))
    if act == "relu":
        model.add(Activation("relu"))
    else:
        model.add(LeakyReLU(alpha=alpha))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(conv_drop_rate))
    model.add(Flatten())

    model.add(Dense(w_end**2*2))
    model.add(Activation("sigmoid"))
    model.add(Reshape((w_end, w_end, 2)))
    return model





if __name__ == '__main__':
    print("[INFO] building network...")
    policy = create_model()
    print('[INFO] test...')
    noise = np.random.uniform(size=(1,8,8))
    zeros = np.zeros((1,8, 8))
    rand_res = policy.predict(noise, verbose=1)
    bias_res = policy.predict(zeros, verbose=1)
    rand_move = (np.unravel_index(np.argmax(rand_res[0,:,:,0]), (8,8)), np.unravel_index(np.argmax(rand_res[0,:,:,1]), (8,8)))
    bias_move = (np.unravel_index(np.argmax(bias_res[0,:,:,0]), (8,8)), np.unravel_index(np.argmax(bias_res[0,:,:,1]), (8,8)))
    print(rand_move, bias_move)
    print(f"[INFO] network summary : ")
    policy.summary()
    save_model(policy, PLAYER1_FILE)
    plot_model(policy, to_file='./model.pdf', show_shapes=True,
               show_dtype=False,
               show_layer_names=True,
               rankdir="UR",
               expand_nested=False,
               dpi=None,
               )