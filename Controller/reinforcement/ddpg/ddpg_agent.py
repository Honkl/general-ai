# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Flood Sung, Modified by Jan Kluj
# Date: 2016.5.4
# -----------------------------------
import gym
import tensorflow as tf
import numpy as np
from reinforcement.ddpg.ou_noise import OUNoise
from reinforcement.ddpg.critic_network_bn import CriticNetwork
from reinforcement.ddpg.actor_network_bn import ActorNetwork
from reinforcement.ddpg.replay_buffer import ReplayBuffer

# Hyper Parameters:

REPLAY_BUFFER_SIZE = 10000
REPLAY_START_SIZE = 500
GAMMA = 0.99
LEARN_EVERY = 1


class DDPGAgent():
    """
    DDPG Agent.
    """

    def __init__(self, batch_size, state_size, actions_count, logdir):
        self.name = 'DDPG'  # name for uploading results
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = state_size
        self.action_dim = actions_count
        self.batch_size = batch_size
        self.total_steps = 0

        graph = tf.Graph()
        graph.seed = np.random.randint(low=0, high=2 ** 16)
        self.sess = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=8,
                                                                  intra_op_parallelism_threads=8,
                                                                  allow_soft_placement=True))
        with self.sess.graph.as_default():
            self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim)
            self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim)

            # initialize replay buffer
            self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

            # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
            self.exploration_noise = OUNoise(self.action_dim)

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            self.summary_writer = tf.summary.FileWriter(logdir,
                                                        graph=self.sess.graph,
                                                        flush_secs=10)

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
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
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

    def play(self, state, i_episode):
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
        if self.replay_buffer.count() > REPLAY_START_SIZE:
            self.total_steps += 1
            if self.total_steps % LEARN_EVERY == 0:
                self.train()

                # if self.time_step % 10000 == 0:
                # self.actor_network.save_network(self.time_step)
                # self.critic_network.save_network(self.time_step)

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()
