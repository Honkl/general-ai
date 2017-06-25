"""
The MIT License (MIT)

Copyright (c) 2016 Flood Sung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This library is used and modified in General Artificial Intelligence for Game Playing project by Jan Kluj.
See the original license above.
"""
import numpy as np
import tensorflow as tf

from reinforcement.ddpg.actor_network_bn import ActorNetwork
from reinforcement.ddpg.critic_network import CriticNetwork
from reinforcement.ddpg.ou_noise import OUNoise
from reinforcement.replay_buffer import ReplayBuffer


class DDPGAgent():
    """
    DDPG Agent.
    """

    def __init__(self, replay_buffer_size, gamma, batch_size, state_size, actions_count, logdir):
        self.name = 'DDPG'  # name for uploading results
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = state_size
        self.action_dim = actions_count
        self.batch_size = batch_size
        self.gamma = gamma

        self.sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=8,
                                                     intra_op_parallelism_threads=8,
                                                     allow_soft_placement=True))
        with self.sess.graph.as_default():
            self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim)
            self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim)

            # initialize replay buffer
            self.replay_buffer = ReplayBuffer(replay_buffer_size)

            # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
            self.exploration_noise = OUNoise(self.action_dim)

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            self.summary_writer = tf.summary.FileWriter(logdir,
                                                        graph=self.sess.graph,
                                                        flush_secs=10)

            self.actor_parameters = self.actor_network.get_parameters()
            self.critic_parameters = self.critic_network.get_parameters()

    def train(self):
        # print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(self.batch_size)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch, [self.batch_size, self.action_dim])

        # Calculate y_batch
        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch, next_action_batch)
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.gamma * q_value_batch[i])
        y_batch = np.resize(y_batch, [self.batch_size, 1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch, state_batch, action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch, action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch, state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def play(self, state):
        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.action(state)
        return action + self.exploration_noise.noise()

    def action(self, state):
        action = self.actor_network.action(state)
        return action

    def perceive(self, state, action, reward, next_state, done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > self.batch_size:
            self.train()

            # if self.time_step % 10000 == 0:
            # self.actor_network.save_network(self.time_step)
            # self.critic_network.save_network(self.time_step)

        # Re-initialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()
