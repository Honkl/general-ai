from __future__ import print_function

import json
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

import constants
import utils.miscellaneous
from reinforcement.abstract_reinforcement import AbstractReinforcement
from reinforcement.ddpg2.ddpg_learner import DeepDeterministicPolicyGradient
from reinforcement.environment import Environment
from utils.activations import get_activation_tf

STD = 0.01
MAX_EPISODES = 1000000
MAX_STEPS = 50000


class DDPG(AbstractReinforcement):
    """
    Represents a DDPG (gradient policy).
    """

    def __init__(self, game, parameters, optimizer_parameters, test_every=10):
        """
        Initializes a new instance of DDPG reinforcement learning model.
        :param game: Game to play.
        :param parameters: DQNParameters instance.
        :param optimizer_parameters: Dictionary; Optimizer parameters.
        :param test_every: Testing every n-th episode.
        """

        self.parameters = parameters
        self.optimizer_params = optimizer_parameters
        self.test_every = test_every

        self.checkpoint_name = "dqn.ckpt"
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
        self.init_directories()

        self.sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=16,
                                                     intra_op_parallelism_threads=16,
                                                     allow_soft_placement=True))

        self.writer = tf.summary.FileWriter(logdir=self.logdir,
                                            graph=self.sess.graph,
                                            flush_secs=10)

        self.num_actions = self.actions_count_sum

        self.agent = DeepDeterministicPolicyGradient(
            session=self.sess,
            optimizer=self.optimizer,
            actor_network=self.actor_network,
            critic_network=self.critic_network,
            state_dim=self.state_size,
            action_dim=self.actions_count_sum,
            batch_size=100,
            replay_buffer_size=1000000,
            store_replay_every=1,
            discount_factor=0.99,
            noise_sigma=0.20,
            noise_theta=0.15,
            summary_writer=self.writer)

    def actor_network(self, input):
        """
        Define actor network for DDPG reinforcement learning algorithm.
        :param input: Inputs.
        :return: Logits.
        """
        ACTOR_DROPOUT = None

        x = None
        for i, dim in enumerate([400, 300]):
            x = tf_layers.fully_connected(inputs=input,
                                          num_outputs=dim,
                                          activation_fn=get_activation_tf("relu"),
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                          biases_initializer=tf.constant_initializer(0),
                                          scope="fully_connected_actor_{}".format(i))

            if ACTOR_DROPOUT:
                x = tf_layers.dropout(x, keep_prob=ACTOR_DROPOUT)

        # Output logits
        logits = tf_layers.fully_connected(inputs=x,
                                           num_outputs=self.num_actions,
                                           activation_fn=None,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           biases_initializer=tf.constant_initializer(0),
                                           scope="output_layer_actor")

        # we assume actions range from [-1, 1]
        # you can scale action outputs with any constant here
        logits = tf.nn.tanh(logits)

        return logits

    def critic_network(self, states, action):
        """
        Defines a critic network for DDPG reinforcement learning algorithm.
        """
        h1_dim = 400
        h2_dim = 300
        state_dim = self.state_size
        action_dim = self.actions_count_sum

        # define policy neural network
        W1 = tf.get_variable("W1", [state_dim, h1_dim],
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", [h1_dim],
                             initializer=tf.constant_initializer(0))
        h1 = tf.nn.relu(tf.matmul(states, W1) + b1)
        # skip action from the first layer
        h1_concat = tf.concat(1, [h1, action])

        W2 = tf.get_variable("W2", [h1_dim + action_dim, h2_dim],
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", [h2_dim],
                             initializer=tf.constant_initializer(0))
        h2 = tf.nn.relu(tf.matmul(h1_concat, W2) + b2)

        W3 = tf.get_variable("W3", [h2_dim, 1],
                             initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable("b3", [1],
                             initializer=tf.constant_initializer(0))
        v = tf.matmul(h2, W3) + b3
        return v

    def init_directories(self, dir_name=None):
        """
        Initializes directories used for logging.
        """
        self.test_logbook_data.append("Testing every {} episodes".format(self.test_every))

        dir = constants.loc + "/logs/" + self.game + "/dqn"
        t_string = utils.miscellaneous.get_pretty_time()

        self.logdir = dir + "/logs_" + t_string
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        with open(os.path.join(self.logdir, "metadata.json"), "w") as f:
            data = {}
            data["model_name"] = "DDPG"
            data["game"] = self.game
            data["actor_critic"] = "Default settings"
            data["parameters"] = self.parameters.to_dictionary()
            data["optimizer_parameters"] = self.optimizer_params
            f.write(json.dumps(data))

    def run(self):
        data = []
        data.append("Episode Steps Score Exploration_rate Time")
        start = time.time()
        tmp = time.time()
        line = ""
        for i_episode in range(MAX_EPISODES):

            self.env = Environment(game_class=self.game_class,
                                   seed=np.random.randint(0, 2 ** 30),
                                   observations_count=self.state_size,
                                   actions_in_phases=self.actions_count)

            # initialize
            state = self.env.state
            total_rewards = 0
            self.negative_reward = 0
            game_start = time.time()

            for t in range(MAX_STEPS):
                action = self.q_learner.eGreedyAction(state[np.newaxis, :])
                next_state, reward, done, info = self.env.step(self.convert_to_sequence(action))

                total_rewards += reward
                if reward < 0:
                    self.negative_reward += 1
                self.q_learner.storeExperience(state, action, reward, next_state, done)

                self.q_learner.updateModel()
                state = next_state
                if done:
                    game_time = time.time() - game_start
                    line = "Episode: {}, Steps: {}, Score: {}, Current exploration rate: {}, Time: {}".format(
                        i_episode, t + 1, info, self.q_learner.exploration, game_time)
                    data.append("{} {} {} {}".format(i_episode, t + 1, info, self.q_learner.exploration, game_time))
                    break

            if (t + 1) == MAX_STEPS:
                print("Maximum number of steps within single game exceeded. ")

            if time.time() - tmp > 1:
                print(line)
                tmp = time.time()

            if i_episode % self.test_every == 0:
                self.test_and_save(data, start, i_episode)

            self.q_learner.measure_summaries(i_episode, info, t + 1, self.negative_reward)

    def convert_to_sequence(self, action):
        """
        From specified action, creates a list of n outputs, onehot encoding.
        """
        result = np.zeros(self.num_actions)
        result[action] = 1
        return result

    def test(self, n_iterations):
        avg_test_score = 0

        for i_episode in range(n_iterations):

            self.env = Environment(game_class=self.game_class,
                                   seed=np.random.randint(0, 2 ** 30),
                                   observations_count=self.state_size,
                                   actions_in_phases=self.actions_count)
            # initialize
            state = self.env.state

            for t in range(MAX_STEPS):
                action = self.q_learner.eGreedyAction(state[np.newaxis, :])
                next_state, reward, done, info = self.env.step(self.convert_to_sequence(action))
                state = next_state

                if done:
                    avg_test_score += info
                    break

        avg_test_score /= n_iterations
        return avg_test_score
