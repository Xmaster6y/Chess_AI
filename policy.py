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
VALUE_FILE = "deep_policies/value"

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

def create_rl_policy(path_file, is_policy_trainable=False, name="player"):
    try:
        model = load_model(path_file)
    except OSError:
        model = create_policy_network(name=name)
    model.trainable = is_policy_trainable
    if is_policy_trainable:
        def rl_policy(player, board):
            observation = board.convert_to_array()
            if player.color == "B": #Always learn to do white moves
                observation = 255 - observation
            action = model(np.expand_dims(observation, axis=0))# Rescale is done in the model
            return action
    else:
        def rl_policy(player, board):
            observation = board.convert_to_array()
            if player.color == "B": #Always learn to do white moves
                observation = 255 - observation
            result = model(np.expand_dims(observation, axis=0))# Rescale is done in the model
            init_pos = np.unravel_index(np.argmax(result[:,:,0]), (8,8))
            end_pos = np.unravel_index(np.argmax(result[:,:,1]), (8,8))
            return (init_pos, end_pos)
    return model, rl_policy

def reward_function(legit_move, be_mat, mat, piece_cap_value, piece_be_cap_value, be_pat, pat):
    ##If not a legit move -> -1000
    ##If lost after -> -100
    ##If win after -> 100
    ##If lost after -> -50
    ##Nothing -> 0
    ##Piece captured -> piece_value
    ##Piece being captured -> -piece_value
    return -1000*(not legit_move) - 100*be_mat + 100*mat - 50*be_pat + 50*pat + piece_cap_value - piece_be_cap_value



def create_policy_network(w_1=16, w_2=24, w_3=32, w_4=16, w_end=8, s_2=1, s_3=1, s_4=1, d_2=2, d_3=8, d_4=4, act="relu", 
                            alpha=0.2, drop_rate=0.3, conv_drop_rate=0, batch_norm=True, name="player-policy"):
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


def create_value_network(w_1=16, w_2=24, w_3=32, w_4=16, w_end=8, s_2=1, s_3=1, s_4=1, d_2=4, d_3=8, d_4=4, act="relu", 
                            alpha=0.2, drop_rate=0.3, conv_drop_rate=0, batch_norm=True, name="player-value"):
    
    s = Input(shape = (8,8)) 
    if batch_norm:
        s1 = Flatten()(BatchNormalization()(s))
    else:
        s1 = Flatten()(s)
    if act == "relu":
        s2 = Dense(w_1**2, activation='relu')(s1)
    else:
        s2 = LeakyReLU(alpha=alpha)(Dense(w1**2)(s1))  
    s3 = Dropout(drop_rate)(s2)
    s4 = Reshape((w_1, w_1, 1))(s3)

    a = Input(shape = (8,8,2)) 
    if batch_norm:
        a1 = Flatten()(BatchNormalization()(a))
    else:
        a1 = Flatten()(a)
    if act == "relu":
        a2 = Dense(w_1**2, activation='relu')(a1)
    else:
        a2 = LeakyReLU(alpha=alpha)(Dense(w1**2)(a1))  
    a3 = Dropout(drop_rate)(a2)
    a4 = Reshape((w_1, w_1, 1))(a3)

    merged = Concatenate(axis = -1)([s4,a4])

    layer = Reshape((w_1, w_1, 2))(merged)
    ker_size = w_2 - (w_1 - 1) * s_2
    layer =Conv2DTranspose(d_2, (ker_size, ker_size))(layer)
    if act == "relu":
        layer = Activation("relu")(layer)
    else:
        layer = LeakyReLU(alpha=alpha)(layer)

    ker_size = w_3 - (w_2 - 1) * s_3
    layer = Conv2DTranspose(d_3, (ker_size, ker_size))(layer)
    if act == "relu":
        layer = Activation("relu")(layer)
    else:
        layer = LeakyReLU(alpha=alpha)(layer)
    if batch_norm:
        layer = BatchNormalization()(layer)
    layer = Dropout(conv_drop_rate)(layer)

    ker_size = w_3 - (w_4 - 1) * s_4
    layer = Conv2D(d_4, (ker_size, ker_size))(layer)
    if act == "relu":
        layer = Activation("relu")(layer)
    else:
        layer = LeakyReLU(alpha=alpha)(layer)
    if batch_norm:
        layer = BatchNormalization()(layer)
    layer = Dropout(conv_drop_rate)(layer)
    layer = Flatten()(layer)

    layer = Dense(1, activation="linear")(layer)
    return Model(inputs = [s, a], outputs = layer, name=name)








if __name__ == '__main__':
    print("[INFO] building network...")
    policy = create_policy_network()
    print('[INFO] test...')
    noise = np.random.uniform(size=(1,8,8))
    zeros = np.zeros((1,8, 8))
    rand_res = policy(noise)
    bias_res = policy.predict(zeros, verbose=True)
    rand_move = (np.unravel_index(np.argmax(rand_res[0,:,:,0]), (8,8)), np.unravel_index(np.argmax(rand_res[0,:,:,1]), (8,8)))
    bias_move = (np.unravel_index(np.argmax(bias_res[0,:,:,0]), (8,8)), np.unravel_index(np.argmax(bias_res[0,:,:,1]), (8,8)))
    print(rand_move, bias_move)
    print(f"[INFO] network summary : ")
    policy.summary()
    save_model(policy, PLAYER1_FILE)
    plot_model(policy, to_file='./policy_network.pdf', show_shapes=True,
               show_dtype=False,
               show_layer_names=True,
               rankdir="UR",
               expand_nested=False,
               dpi=None,
               )
    print("[INFO] building network...")
    value = create_value_network()

    print('[INFO] test...')
    noise_s = np.random.uniform(size=(1,8,8))
    noise_a = np.random.uniform(size=(1,8,8,2))
    rand_res = value([noise_s,noise_a])
    print(rand_res)
    print(f"[INFO] network summary : ")
    value.summary()
    save_model(value, VALUE_FILE)
    plot_model(value, to_file='./value_network.pdf', show_shapes=True,
               show_dtype=False,
               show_layer_names=True,
               rankdir="UR",
               expand_nested=False,
               dpi=None,
               )