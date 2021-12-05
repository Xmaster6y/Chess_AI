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

# Initialize tensorboard object
#name = f'VPG_logs_{time.time()}'
#summary_writer = tf.summary.create_file_writer(logdir = f'logs/{name}/')

if __name__ == '__main__':
    gamma = 0.99 #discount
    p_lr = 0.01
    lam = 0.97
    epochs = 300
    max_steps_per_epoch = 1000
    render = False
    render_time = 100
    train_policy_iterations = 1
    reward_scale = 1000

    training_color = "W"
    against_random = True

    if training_color == "W":
        try:
            model, ai_policy = policy.create_rl_policy(PLAYER1_FILE, True)
        except:
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
            model, ai_policy = policy.create_rl_policy(PLAYER2_FILE, True)
        except:
            print("AI not trained")
            raise SystemExit
        if against_random:
            player1 = Player(True, "W", policy.random_policy)
        else:
            _, ai_policy1 = policy.create_rl_policy(PLAYER1_FILE)
            player1 = Player(True, "W", ai_policy1)
        player2 = Player(True, "B", ai_policy, True)
    board = Board()



    # Optimizers for the model
    optimizer_policy_net = tf.optimizers.Adam(p_lr)

    # Main Loop
    for epoch in range(epochs):
        # Reset the environment and observe the state
        board.reset()
        done = False
        rewards = []
        memory = []
        returns = []
        log_probs = []

        for t in range(max_steps_per_epoch):

            if render and t%render_time == 0:
                plot(board)
                
            # Select action using current policy
            state = board.convert_to_array()
            action = player1.choose_move(board) # deep player
            init_pos = np.unravel_index(np.argmax(result[:,:,0]), (8,8))
            end_pos = np.unravel_index(np.argmax(result[:,:,1]), (8,8))
            move = (init_pos, end_pos)

            legit, piece_cap, be_mat, be_pat = apply_move(player1, board, move)
            piece_cap_value = Board.values[piece_cap] * Board.value_scale
            done = (not legit) | be_mat | be_pat
            if done:
                reward = reward_function(legit_move, be_mat, False, piece_cap_value, 0, be_pat, False)

            else:
                move = player2.choose_move(board)
                legit, piece_be_cap, mat, pat = apply_move(player2, board, move)
                piece_be_cap_value = Board.values[piece_be_cap] * Board.value_scale
                done = (not legit) | mat | pat
                reward = reward_function(True, be_mat, mat, piece_cap_value, piece_be_cap_value, be_pat, pat)

            # Store the data in memory for policy update
            memory.append((state, action))
            rewards.append(reward/reward_scale)

            if done or (t+1 == local_steps_per_epoch):

                # Compute Rewards to Go
                returns = np.zeros(t+1, dtype = 'float32')
                for i in reversed(range(t+1)):
                    returns[i] = rewards[i] + (gamma*returns[i+1] if i+1 < n else 0)

                temp = zip(*memory)
                states, actions = np.asarray(temp[0]),np.asarray(temp[1])
                total_reward = sum(ep_rewards)
                break

        with tf.GradientTape() as policy_tape:
            logits = tf.nn.log_softmax(model(np.atleast_2d(np.array(states)).astype('float32')))

            one_hot_values = tf.squeeze(tf.one_hot(np.array(actions), actions[0].size))
            log_probs = tf.math.reduce_sum(logits * one_hot_values, axis=1)
            policy_loss = -tf.math.reduce_mean(returns * log_probs)

        policy_variables = model.trainable_variables
        policy_gradients = policy_tape.gradient(policy_loss, policy_variables)

        # Update the policy network weights using ADAM
        optimizer_policy_net.apply_gradients(zip(policy_gradients, policy_variables))

        for iteration in range(train_policy_iterations):
            optimizer_policy_net.apply_gradients(zip(policy_gradients, policy_variables))
        
        # Book-keeping
        with summary_writer.as_default():
            tf.summary.scalar('Episode_returns', sum(returns), step = epoch)
            tf.summary.scalar('Running_total_reward', total_rewards, step = epoch)
            tf.summary.scalar('Losses', policy_loss, step = epoch)

        if epoch%1 == 0:
            print(f"Episode: {epoch} Losses: {policy_loss: 0.2f} Avg_reward: {total_rewards: 0.2f}")


    # To render the environment after the training to check how the model performs.
    # You can save the weights for further use using model.save_weights() function from TF2
    render_var = input("Do you want to render the env(Y/N) ?")
    if render_var == 'Y' or render_var == 'y':
        n_render_iter = int(input("How many episodes? "))
        
        for i in range(n_render_iter):
            state = env.reset()
            while True:
                action = agent.select_action(state, policy_net)
                env.render()
                n_state, reward, done, _ = env.step(action.numpy())
                if done:
                    break
    else:
        print("Thankyou for using!")

    env.close()





















































# Libraries imports
from dcgan import FLEXI_DCGAN

import scipy.stats as sc
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

from keras.models import Model, load_model
from keras.layers import Input
from keras.optimizers import Adam
from sklearn.utils import shuffle

# Class definition

# Functions definition

# Constants definition
BLOCK_SIZE = 90
SAMPLING_SIZE = 156
DATA_SET_SIZE = 3 ** 4 * 2 ** 9
FILE = './Samples/real_ds_dist_pluiv_glob.npy'

FILE_GEN_STEP = "./Networks/gan_dist_pluiv_step_gen"
FILE_DISC_STEP = "./Networks/gan_dist_pluiv_step_disc"

NUM_EPOCHS = 1  # 50_000
BATCH_SIZE = 3 ** 1 * 2 ** 7
INIT_LR = 4e-4
LAST_ACTI = "linear"
H = 10
FACT_LR = 1
GRAD_NORM = 1

TYPE = 'student'
SHAPE = 5
SIGMA_NOISE_DATA = 1e-5

SIMU_GAN = True
SIMU_DIST = True
SIMU_TREND = False
SAVE_STEPS = False
SAVE_FIG = False
SAVE_TEX = False
FS = 30

if __name__ == '__main__':
    if SIMU_GAN:
        X = np.load(FILE)

        if SAMPLING_SIZE == 165:
            print("[INFO] building generator...")
            gen = FLEXI_DCGAN.build_generator(w_end=SAMPLING_SIZE, w_start=SAMPLING_SIZE,
                                              w_1=110, s_1=1, w_2=150, final_act=LAST_ACTI, h=H)
            print("[INFO] building discriminator...")
            disc = FLEXI_DCGAN.build_discriminator(w_start=SAMPLING_SIZE,
                                                   w_1=150, w_2=90, w_3=15, h=H)
        elif SAMPLING_SIZE == 156:
            print("[INFO] building generator...")
            gen = FLEXI_DCGAN.build_generator(w_end=SAMPLING_SIZE, w_start=SAMPLING_SIZE,
                                              w_1=100, s_1=1, w_2=140, final_act=LAST_ACTI, h=H)
            print("[INFO] building discriminator...")
            disc = FLEXI_DCGAN.build_discriminator(w_start=SAMPLING_SIZE,
                                                   w_1=140, w_2=80, w_3=15, h=H)
        else:
            raise NotImplementedError
        try:
            gen = load_model(FILE_GEN_STEP, compile=False)
            disc = load_model(FILE_DISC_STEP, compile=False)
        except:
            pass
        print('[INFO] test...')
        discOpt = Adam(learning_rate=INIT_LR, beta_1=0.999, decay=INIT_LR / NUM_EPOCHS, clipnorm=GRAD_NORM)
        disc.compile(loss="binary_crossentropy", optimizer=discOpt)

        zeros = np.zeros((BATCH_SIZE, SAMPLING_SIZE))
        _ = gen.predict(zeros, verbose=1)
        _ = disc.predict(zeros, verbose=1)
        print(f"[INFO] discriminator summary : ")
        disc.summary()
        print(f"[INFO] generator summary : ")
        gen.summary()

        print("[INFO] building GAN...")
        disc.trainable = False
        ganInput = Input(shape=(SAMPLING_SIZE))
        ganOutput = disc(gen(ganInput))
        gan = Model(ganInput, ganOutput)
        ganOpt = Adam(learning_rate=INIT_LR / 4, beta_1=0.5, decay=INIT_LR / 4 / NUM_EPOCHS, clipnorm=GRAD_NORM)
        gan.compile(loss="binary_crossentropy", optimizer=discOpt)

        print("[INFO] starting training...")
        benchmark_noise = np.random.normal(size=(1, SAMPLING_SIZE))
        disc_losses = []
        gen_losses = []

        for epoch in range(0, NUM_EPOCHS):
            print("[INFO] starting epoch {} of {}...".format(epoch + 1,
                                                             NUM_EPOCHS))
            X = shuffle(X)
            batchesPerEpoch = int(X.shape[0] / BATCH_SIZE)
            for i in range(0, batchesPerEpoch):
                batch_data = X[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :] + np.random.normal(loc=SIGMA_NOISE_DATA, size=(
                    BATCH_SIZE, SAMPLING_SIZE))
                noise = np.random.normal(size=(BATCH_SIZE, SAMPLING_SIZE))
                gen_data = gen.predict(noise, verbose=0)

                X_tr = np.concatenate((batch_data, gen_data))
                y_tr = np.array(([1] * BATCH_SIZE) + ([0] * BATCH_SIZE))
                (X_tr, y_tr) = shuffle(X_tr, y_tr)
                disc_loss = disc.train_on_batch(X_tr, y_tr)
                disc_losses.append(disc_loss)

                noise = np.random.normal(size=(BATCH_SIZE, SAMPLING_SIZE))
                fake_labels = np.array([1] * BATCH_SIZE)
                gen_loss = gan.train_on_batch(noise, fake_labels)
                gen_losses.append(gen_loss)

            if NUM_EPOCHS >= 10:
                if epoch % (NUM_EPOCHS // 10) == 0:
                    if SIMU_DIST:
                        gen_sample = gen.predict(benchmark_noise, verbose=0)
                        plt.subplot(2, 5, epoch // (NUM_EPOCHS // 10) + 1)
                        plt.hist(gen_sample.flatten(), bins=100)
                    if SIMU_TREND:
                        gen_sample = gen.predict(benchmark_noise, verbose=0)
                        plt.subplot(2, 5, epoch // (NUM_EPOCHS // 10) + 1)
                        plt.plot(gen_sample.mean(axis=0))
            if SAVE_STEPS:
                gen.save(FILE_GEN_STEP, include_optimizer=True)
                disc.save(FILE_DISC_STEP, include_optimizer=True)

        if SAVE_FIG:
            if SAVE_TEX:
                tikzplotlib.save("./Images/states.tex")
            else:
                plt.savefig("./Images/States.png")
            plt.close()
        else:
            plt.show()

        plt.figure(figsize=(15, 15))
        steps = 1 + np.arange(len(gen_losses))
        plt.plot(steps, gen_losses, label="Loss of the generator")
        plt.plot(steps, disc_losses, label="Loss of the discriminator")
        plt.xticks(fontsize=FS)
        plt.yticks(fontsize=FS)
        plt.xlabel("Steps", fontsize=FS)
        plt.ylabel(r"Loss", fontsize=FS)
        plt.legend(loc='best', fontsize=FS)
        if SAVE_FIG:
            if SAVE_TEX:
                tikzplotlib.save("Images/Loss.tex")
            else:
                plt.savefig("./Images/Loss.png")
            plt.close()
        else:
            plt.show()