import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import tensorflow as tf


class GreedyPolicyAgent():
    def __init__(self, reinfoce_params, q_network, state_size, logdir, threads):
        self.batch_size = reinfoce_params.batch_size
        batch_size = self.batch_size
        self.gamma = reinfoce_params.gamma
        self.random_action_prob = reinfoce_params.rand_action_prob
        self.state_size = state_size
        self.q_network = q_network
        self.logdir = logdir
        self.saver = None

        self.sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                     intra_op_parallelism_threads=threads,
                                                     allow_soft_placement=True))
        with tf.device('/cpu:0'):
            with tf.variable_scope('agent') as scope:
                self.state = tf.placeholder(shape=[None, state_size], dtype=tf.float32, name="state")

                self.selected_action, self.estimated_reward = self.select_best_action(self.state)
                self.selected_action = tf.squeeze(self.selected_action)
                self.estimated_reward = tf.squeeze(self.estimated_reward)
                scope.reuse_variables()

                self.new_state = tf.placeholder(shape=[batch_size, state_size], dtype=tf.float32, name="new_state")
                self.last_reward = tf.placeholder(shape=[batch_size], dtype=tf.float32, name="last_reward")
                self.last_estimated_reward = tf.placeholder(shape=[batch_size], dtype=tf.float32,
                                                            name="last_estimated_reward")

                _, new_estimated_reward = self.select_best_action(self.new_state)

                self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")

                # loss = (r + γ*max_a'Q(s',a';θ) - Q(s,a;θ))^2
                self.losses = (self.last_reward + self.gamma * self.last_estimated_reward - new_estimated_reward) ** 2
                self.loss = tf.reduce_mean(self.losses)
                self.optimizer = self.get_optimizer(reinfoce_params.optimizer)
                self.training = self.optimizer(reinfoce_params.learning_rate).minimize(self.loss,
                                                                                       global_step=self.global_step)

                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

                self.summary_writer = tf.summary.FileWriter(logdir,
                                                            graph=self.sess.graph,
                                                            flush_secs=10)

    def get_optimizer(self, opt_str):
        if opt_str == "adam":
            return tf.train.AdamOptimizer
        if opt_str == "rmsprop":
            return tf.train.RMSPropOptimizer
        raise NotImplementedError

    def play(self, env_state, i_episode):
        # Probability of random action selection will decay in time
        if np.random.random() < self.random_action_prob ** i_episode:
            estimated_reward = self.sess.run(self.estimated_reward, {self.state: [env_state]})
            n_actions = self.q_network.output_size
            selected_action = []
            for i in range(n_actions):
                selected_action.append(np.random.random())
        else:
            selected_action, estimated_reward = self.sess.run([self.selected_action, self.estimated_reward],
                                                              {self.state: [env_state]})
        return selected_action, estimated_reward

    def select_best_action(self, state):
        result = self.Q(state)
        # selected_action = tf.argmax(result, 1)
        selected_action = result
        estimated_reward = tf.reduce_max(result, 1)
        return selected_action, estimated_reward

    def Q(self, state):
        return self.q_network.forward_pass(state)

    # def learn(self, env_states, last_action, last_reward, last_estimated_reward):
    def learn(self, states, rewards, estimated_rewards):
        _, loss = self.sess.run([self.training, self.loss],
                                {self.new_state: np.array(states).reshape(self.batch_size, self.state_size),
                                 self.last_reward: np.array(rewards),
                                 self.state: np.array(states).reshape(self.batch_size, self.state_size),
                                 self.last_estimated_reward: np.array(estimated_rewards).reshape(self.batch_size)})
        return loss
