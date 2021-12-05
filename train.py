#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""""
@author: Xmaster a.k.a. Yoann Poupart
File containing an example of GAN training.
"""

import time
import numpy as np
import scipy.signal
import tensorflow as tf 
from statistics import mean
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.optimizers as ko
import tensorflow.keras.losses as kls
from tensorflow.keras.callbacks import TensorBoard
from keras.models import load_model, save_model

import policy
import rules
from player import Player
from board_class import Board

# Initialize tensorboard object
name = f'VPG_logs_{time.time()}'
summary_writer = tf.summary.create_file_writer(logdir = f'logs/{name}/')
PLAYER1_FILE = "deep_policies/player1"
PLAYER2_FILE = "deep_policies/player2"
VALUE_FILE = "deep_policies/value"

def normalize(adv):
    g_n = len(adv)
    mean = np.mean(adv)
    std = np.std(adv)

    return (adv-mean)/std

if __name__ == '__main__':
    gamma = 0.99 #discount
    p_lr = 0.01
    v_lr = 0.01
    lam = 0.97
    epochs = 500
    train_value_iterations = 80
    max_steps_per_epoch = 1000
    render = False
    render_time = 100
    train_policy_iterations = 1
    reward_scale = 1000

    training_color = "W"
    against_random = True

    if training_color == "W":
        try:
            policy_net, ai_policy = policy.create_rl_policy(PLAYER1_FILE, True)
        except Exception as e:
            print(e)
            print("AI not trained")
            raise SystemExit
        if against_random:
            player2 = Player(True, "B", policy.random_policy)
        else:
            _, ai_policy2 = policy.create_rl_policy(PLAYER2_FILE)
            player2 = Player(True, "B", policy2)
        player1 = Player(True, "W", ai_policy, True)
    else:
        try:
            policy_net, ai_policy = policy.create_rl_policy(PLAYER2_FILE, True)
        except:
            print("AI not trained")
            raise SystemExit
        if against_random:
            player2 = Player(True, "W", policy.random_policy)
        else:
            _, ai_policy1 = policy.create_rl_policy(PLAYER1_FILE)
            player2 = Player(True, "W", ai_policy1)
        player1 = Player(True, "B", ai_policy, True)
    board = Board()

    try:
        value_net = load_model(VALUE_FILE)
    except OSError:
        value_net, _ = policy.create_rl_policy(VALUE_FILE, True)
        value_net.layers.pop()
        value_net.add(kl.Dense(1, activation = 'linear', name="laste_dense"))

    # Optimizers for the model
    optimizer_policy_net = tf.optimizers.Adam(p_lr)
    optimizer_value_net = tf.optimizers.Adam(v_lr)

    test_state = np.zeros((1,8,8))
    value_net(test_state)
    policy_net(test_state)

    # Main Loop
    for epoch in range(epochs):
        # Reset the environment and observe the state
        board.reset()
        done = False
        rewards = []
        states = []
        actions = []
        values = []

        if training_color == "B":
            move = player2.choose_move(board)
            legit, piece_be_cap, mat, pat = rules.apply_move(player2, board, move)

        for t in range(max_steps_per_epoch):

            if render and t%render_time == 0:
                plot(board)
                
            # Select action using current policy
            state = board.convert_to_array()
            action = player1.choose_move(board) # deep player
            value = value_net(np.expand_dims(state.astype('float32'), axis=0))

            init_pos = np.unravel_index(np.argmax(action[:,:,0]), (8,8))
            end_pos = np.unravel_index(np.argmax(action[:,:,1]), (8,8))
            move = (init_pos, end_pos)

            legit, piece_cap, be_mat, be_pat = rules.apply_move(player1, board, move)
            piece_cap_value = Board.values[piece_cap] * Board.value_scale
            done = (not legit) | be_mat | be_pat
            if done:
                reward = policy.reward_function(legit, be_mat, False, piece_cap_value, 0, be_pat, False)

            else:
                move = player2.choose_move(board)
                legit, piece_be_cap, mat, pat = rules.apply_move(player2, board, move)
                piece_be_cap_value = Board.values[piece_be_cap] * Board.value_scale
                done = (not legit) | mat | pat
                reward = policy.reward_function(True, be_mat, mat, piece_cap_value, piece_be_cap_value, be_pat, pat)

            # Store the data in memory for policy update
            actions.append(action)
            states.append(state)
            rewards.append(reward/reward_scale)
            values.append(value)

            if done or (t+1 == local_steps_per_epoch):

                # Compute Rewards to Go
                values.append(0)
                rewards = np.array(rewards)
                actions = np.array(actions)
                states = np.array(states)
                values = np.array(values)

                returns = np.zeros(t+1, dtype = 'float32')
                for i in reversed(range(t+1)):
                    returns[i] = rewards[i] + (gamma*returns[i+1] if i+1 < t+1 else 0)


                delta = rewards + gamma * values[1:] - values[:-1]
                advantage = list(scipy.signal.lfilter([1], [1, float(-gamma*lam)], delta[::-1], axis=0)[::-1])

                advantage = normalize(advantage)

                total_reward = sum(rewards)
                break

        with tf.GradientTape() as policy_tape, tf.GradientTape() as value_tape:
            logits = tf.nn.log_softmax(policy_net(np.array(states).astype('float32')))
            #one_hot_values = tf.one_hot(np.array(actions), 64*64)
            log_probs = tf.math.reduce_sum(logits, axis=[1,2,3])
            policy_loss = -tf.math.reduce_mean(advantage * log_probs)
            value_loss = -tf.math.reduce_mean(kls.MSE(returns,value_net(np.array(states).astype('float32'))))

        policy_variables = policy_net.trainable_variables
        value_variables = value_net.trainable_variables
        policy_gradients = policy_tape.gradient(policy_loss, policy_variables)
        value_gradients = value_tape.gradient(value_loss, value_variables)

        # Update the policy network weights using ADAM
        optimizer_policy_net.apply_gradients(zip(policy_gradients, policy_variables))

        for iteration in range(train_value_iterations):
            optimizer_value_net.apply_gradients(zip(value_gradients, value_variables))
        
        # Book-keeping
        with summary_writer.as_default():
            tf.summary.scalar('Episode_returns', sum(returns), step = epoch)
            tf.summary.scalar('Running_total_reward', total_reward, step = epoch)
            tf.summary.scalar('Losses', policy_loss, step = epoch)

        if epoch%50 == 0:
            print(f"[INFO] Episode: {epoch} Policy Loss: {policy_loss: 0.5e} \
Value Loss: {value_loss: 0.5e} Total_reward: {total_reward: 0.2e}")


    
    save_model(policy_net, PLAYER1_FILE)
    save_model(value_net, VALUE_FILE)
    # To render the environment after the training to check how the model performs.
    render_var = input("Do you want to render the env(Y/N) ?")
    if render_var == 'Y' or render_var == 'y':
        n_render_iter = int(input("How many episodes? "))
        
        for i in range(n_render_iter):
            board.reset()
            while True:
                print(board)
                if training_color == "B":
                    move = player2.choose_move(board)
                    legit, piece_be_cap, mat, pat = rules.apply_move(player2, board, move)
                    print(f"Random move {move}")
                state = board.convert_to_array()
                action = player1.choose_move(board) 
                init_pos = np.unravel_index(np.argmax(action[:,:,0]), (8,8))
                end_pos = np.unravel_index(np.argmax(action[:,:,1]), (8,8))
                move = (init_pos, end_pos)
                print(f"AI move {move}")

                legit, piece_cap, be_mat, be_pat = rules.apply_move(player1, board, move)
                piece_cap_value = Board.values[piece_cap] * Board.value_scale
                done = (not legit) | be_mat | be_pat
                if not done:
                    move = player2.choose_move(board)
                    legit, piece_be_cap, mat, pat = rules.apply_move(player2, board, move)
                    print(f"Random move {move}")
                    piece_be_cap_value = Board.values[piece_be_cap] * Board.value_scale
                    done = (not legit) | mat | pat

                if done:
                    break
    else:
        print("Thankyou for using!")

    env.close()




