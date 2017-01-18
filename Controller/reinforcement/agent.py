import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import tensorflow as tf


class Agent():
    def __init__(self, reinfoce_params, q_network, state_size, actions_count, expname, threads):
        self.batch_size = reinfoce_params.batch_size
        batch_size = self.batch_size
        self.dropout = reinfoce_params.dropout
        self.gamma = reinfoce_params.gamma
        self.state_size = state_size
        self.actions_count = actions_count
        self.q_network = q_network

        with tf.device('/cpu:0'):
            with tf.variable_scope('agent') as scope:
                self.state = tf.placeholder(shape=[batch_size, state_size], dtype=tf.float32)
                self.estimated_rewards = self.Q(self.state)
                scope.reuse_variables()

                self.selected_action, self.estimated_reward = self.select_best_action(self.state)
                self.new_state = tf.placeholder(shape=[batch_size, state_size], dtype=tf.float32)
                self.last_action = tf.placeholder(shape=[batch_size], dtype=tf.int32)
                self.last_reward = tf.placeholder(shape=[batch_size], dtype=tf.float32)
                self.last_estimated_reward = tf.placeholder(shape=[batch_size], dtype=tf.float32)
                _, new_estimated_reward = self.select_best_action(self.new_state)

                # loss = (r + γ*max_a'Q(s',a';θ) - Q(s,a;θ))^2
                self.losses = (self.last_reward + self.gamma * self.last_estimated_reward - new_estimated_reward) ** 2
                self.loss = tf.reduce_mean(self.losses)
                self.training = tf.train.AdamOptimizer().minimize(self.loss)

                self.session = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                intra_op_parallelism_threads=threads,
                                                                allow_soft_placement=True))

                self.session.run(tf.global_variables_initializer())

                self.summary_writer = tf.train.SummaryWriter('train_{}'.format(expname),
                                                             graph=self.session.graph,
                                                             flush_secs=10)

    def play(self, env_states):
        selected_action, estimated_reward = self.session.run([self.selected_action, self.estimated_reward],
                                                             {self.state: np.array(env_states).reshape(self.batch_size,
                                                                                                       self.state_size)})
        return selected_action, estimated_reward

    def select_best_action(self, state):
        result = self.Q(state)
        selected_action = tf.argmax(result, 1)
        estimated_reward = tf.reduce_max(result, 1)
        return selected_action, estimated_reward

    def Q(self, state):
        return self.q_network.forward_pass(state)

    def learn(self, env_states, last_action, last_reward, last_estimated_reward):
        _, loss = self.session.run([self.training, self.loss],
                                   {self.new_state: np.array(env_states).reshape(self.batch_size, self.state_size),
                                    self.last_action: last_action,
                                    self.last_estimated_reward: last_estimated_reward,
                                    self.last_reward: last_reward})
        return loss
