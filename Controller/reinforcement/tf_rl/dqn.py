from __future__ import print_function
from collections import deque

from reinforcement.tf_rl.rl_implementations.neural_q_learner import NeuralQLearner
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from utils.activations import get_activation_tf
import numpy as np
import time
from reinforcement.environment import Environment
import utils.miscellaneous
import constants
import os
import matplotlib.pyplot as plt
import json
from reinforcement.reinforcement_parameters import DQNParameters

STD = 0.01
MAX_EPISODES = 1000000
MAX_STEPS = 500000


class DQN():
    def __init__(self, game):

        #
        # Set parameters of the model
        #
        self.q_net_hidden_layers = [500, 500, 500]
        self.activation_f = "relu"
        self.parameters = DQNParameters(batch_size=100,
                                        init_exp=0.5,
                                        final_exp=0.1,
                                        anneal_steps=100000,
                                        replay_buffer_size=10000,
                                        store_replay_every=5,
                                        discount_factor=0.9,
                                        target_update_rate=0.01,
                                        reg_param=0.01,
                                        max_gradient=5,
                                        double_q_learning=False)

        self.optimizer_params = {}
        self.optimizer_params["name"] = "rmsprop"
        self.optimizer_params["learning_rate"] = 0.005
        self.optimizer_params["decay"] = 0.9
        self.optimizer_params["momentum"] = 0.95

        #
        #
        #

        lr = self.optimizer_params["learning_rate"]
        if self.optimizer_params["name"] == "rmsprop":
            d = self.optimizer_params["decay"]
            m = self.optimizer_params["momentum"]
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=d, momentum=m)
        if self.optimizer_params["name"] == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        self.game = game
        self.game_config = utils.miscellaneous.get_game_config(game)
        self.game_class = utils.miscellaneous.get_game_class(game)
        self.state_size = self.game_config["input_sizes"][0]

        self.actions_count = self.game_config["output_sizes"]
        self.actions_count_sum = sum(self.actions_count)
        self.create_dirs_and_logs()

        self.sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=8,
                                                     intra_op_parallelism_threads=8,
                                                     allow_soft_placement=True))

        self.writer = tf.summary.FileWriter(logdir=self.logdir,
                                            graph=self.sess.graph,
                                            flush_secs=10)

        self.state_dim = 16
        self.num_actions = 4

        self.q_learner = NeuralQLearner(self.sess,
                                        self.optimizer,
                                        self.q_network,
                                        self.state_dim,
                                        self.num_actions,
                                        summary_writer=self.writer,
                                        summary_every=100,
                                        batch_size=100,
                                        anneal_steps=100000,
                                        replay_buffer_size=10000,
                                        target_update_rate=0.1,
                                        store_replay_every=1,  # how frequent to store experience
                                        discount_factor=0.9,  # discount future rewards
                                        reg_param=0.01,  # regularization constants
                                        max_gradient=5,  # max gradient norms
                                        double_q_learning=False)

    def create_dirs_and_logs(self):
        dir = constants.loc + "/logs/" + self.game + "/dqn"
        current = time.localtime()
        t_string = "{}-{}-{}_{}-{}-{}".format(str(current.tm_year).zfill(2), str(current.tm_mon).zfill(2),
                                              str(current.tm_mday).zfill(2), str(current.tm_hour).zfill(2),
                                              str(current.tm_min).zfill(2), str(current.tm_sec).zfill(2))
        self.logdir = dir + "/logs_" + t_string
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        with open(os.path.join(self.logdir, "metadata.json"), "w") as f:
            data = {}
            data["model_name"] = "DQN"
            data["game"] = self.game

            q_net = {}
            q_net["activation"] = self.activation_f
            q_net["hidden_layers"] = self.q_net_hidden_layers
            data["q_network"] = q_net

            data["parameters"] = self.parameters.to_dictionary()
            f.write(json.dumps(data))

    def q_network(self, input):
        # Hidden fully connected layers
        x = None
        for i, dim in enumerate(self.q_net_hidden_layers):
            x = tf_layers.fully_connected(inputs=input,
                                          num_outputs=dim,
                                          activation_fn=get_activation_tf(self.activation_f),
                                          weights_initializer=tf.random_normal_initializer(mean=0, stddev=STD),
                                          scope="fully_connected_{}".format(i))

        # Output logits
        logits = tf_layers.fully_connected(inputs=x,
                                           num_outputs=self.num_actions,
                                           activation_fn=None,
                                           weights_initializer=tf.random_normal_initializer(mean=0, stddev=STD),
                                           scope="output_layer")

        return logits

    def run(self):
        data = []
        start = time.time()
        for i_episode in range(MAX_EPISODES):

            self.env = Environment(game_class=self.game_class,
                                   seed=np.random.randint(0, 2 ** 30),
                                   observations_count=self.state_size,
                                   actions_in_phases=self.actions_count,
                                   discrete=True)
            # initialize
            state = self.env.reset()
            total_rewards = 0

            for t in range(MAX_STEPS):
                action = self.q_learner.eGreedyAction(state[np.newaxis, :])
                next_state, reward, done, info = self.env.step(action)

                total_rewards += reward
                # reward = -10 if done else 0.1 # normalize reward
                self.q_learner.storeExperience(state, action, reward, next_state, done)

                self.q_learner.updateModel()
                state = next_state

                if done:
                    data.append(info)
                    break

            if i_episode % 100 == 0:
                with open(self.logdir + "/logbook.txt", "w") as f:
                    for x in data:
                        f.write(str(x))
                        f.write("\n")

                plt.figure()
                plt.plot(data)
                plt.savefig(self.logdir + "/plot.png")

            if time.time() - start > 1:
                print("Episode: {}, Steps: {}, Score: {}".format(i_episode, t + 1, info))
                start = time.time()

            self.q_learner.measure_summaries(i_episode, info, t + 1)
