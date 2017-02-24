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

STD = 0.01
MAX_EPISODES = 1000000
MAX_STEPS = 1000


class DQN():
    def __init__(self, game):
        self.game_config = utils.miscellaneous.get_game_config(game)
        self.game_class = utils.miscellaneous.get_game_class(game)
        self.state_size = self.game_config["input_sizes"][0]

        self.actions_count = self.game_config["output_sizes"]
        self.actions_count_sum = sum(self.actions_count)

        self.sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=8,
                                                     intra_op_parallelism_threads=8,
                                                     allow_soft_placement=True))

        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9, momentum=0.95)
        self.writer = tf.summary.FileWriter("test_summary/{}".format(time.time()),
                                            graph=self.sess.graph,
                                            flush_secs=10)

        self.env = Environment(game_class=self.game_class,
                               seed=np.random.randint(0, 2 ** 30),
                               observations_count=self.state_size,
                               actions_in_phases=self.actions_count,
                               discrete=True)

        self.state_dim = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.q_learner = NeuralQLearner(self.sess,
                                        self.optimizer,
                                        self.observation_to_action,
                                        self.state_dim,
                                        self.num_actions,
                                        summary_writer=self.writer,
                                        summary_every=100,
                                        batch_size=100,
                                        replay_buffer_size=10000,
                                        store_replay_every=5,  # how frequent to store experience
                                        discount_factor=0.9,  # discount future rewards
                                        reg_param=0.01,  # regularization constants
                                        max_gradient=10,  # max gradient norms
                                        double_q_learning=False)

    def observation_to_action(self, input):
        # Hidden fully connected layers
        for i, dim in enumerate([256, 256]):
            x = tf_layers.fully_connected(inputs=input,
                                          num_outputs=dim,
                                          activation_fn=get_activation_tf("relu"),
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
        episode_history = deque(maxlen=100)
        for i_episode in range(MAX_EPISODES):

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

                if done: break

            episode_history.append(total_rewards)
            mean_rewards = np.mean(episode_history)

            print("Episode: {}, Steps: {}, TReward: {}, Score: {}".format(i_episode, t + 1, total_rewards, info))
            report_measures = ([tf.Summary.Value(tag='score', simple_value=info),
                                tf.Summary.Value(tag='number_of_steps', simple_value=t + 1)])
            self.writer.add_summary(tf.Summary(value=report_measures), i_episode)
