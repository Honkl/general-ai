from __future__ import print_function
from collections import deque

from reinforcement.tf_rl.rl.neural_q_learner import NeuralQLearner
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from utils.activations import get_activation_tf
from utils.miscellaneous import get_rnn_cell
import numpy as np
import gym
from reinforcement.environment import Environment
import utils.miscellaneous

# env_name = 'CartPole-v0'
# env = gym.make(env_name)

game = "2048"
game_config = utils.miscellaneous.get_game_config(game)
game_class = utils.miscellaneous.get_game_class(game)
state_size = game_config["input_sizes"][0]

actions_count = game_config["output_sizes"]
actions_count_sum = sum(actions_count)

env = Environment(game_class=game_class,
                  seed=np.random.randint(0, 2 ** 16),
                  observations_count=state_size,
                  actions_in_phases=actions_count,
                  discrete=True)
import time

sess = tf.Session()
#optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
writer = tf.summary.FileWriter("test_summary/{}".format(time.time()),
                               graph=sess.graph,
                               flush_secs=10)

state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
STD = 0.01

def observation_to_action(states):
    # Hidden fully connected layers
    for i, dim in enumerate([256, 256]):
        x = tf_layers.fully_connected(inputs=states,
                                      num_outputs=dim,
                                      activation_fn=get_activation_tf("relu"),
                                      weights_initializer=tf.random_normal_initializer(mean=0, stddev=STD),
                                      biases_initializer=tf.random_normal_initializer(mean=0, stddev=STD),
                                      scope="fully_connected_{}".format(i))

    # Output logits
    logits = tf_layers.fully_connected(inputs=x,
                                       num_outputs=num_actions,
                                       activation_fn=None,
                                       weights_initializer=tf.random_normal_initializer(mean=0, stddev=STD),
                                       biases_initializer=tf.random_normal_initializer(mean=0, stddev=STD),
                                       scope="output_layer")

    return logits


q_learner = NeuralQLearner(sess,
                           optimizer,
                           observation_to_action,
                           state_dim,
                           num_actions,
                           summary_writer=writer,
                           summary_every=100,
                           batch_size=100,
                           replay_buffer_size=10000,
                           store_replay_every=5,  # how frequent to store experience
                           discount_factor=0.9,  # discount future rewards
                           reg_param=0.01,  # regularization constants
                           max_gradient=10,  # max gradient norms
                           double_q_learning=False)

MAX_EPISODES = 1000000
MAX_STEPS = 1000

episode_history = deque(maxlen=100)
for i_episode in range(MAX_EPISODES):

    # initialize
    state = env.reset()
    total_rewards = 0

    for t in range(MAX_STEPS):
        action = q_learner.eGreedyAction(state[np.newaxis, :])
        next_state, reward, done, info = env.step(action)

        total_rewards += reward
        # reward = -10 if done else 0.1 # normalize reward
        q_learner.storeExperience(state, action, reward, next_state, done)

        q_learner.updateModel()
        state = next_state

        if done: break

    episode_history.append(total_rewards)
    mean_rewards = np.mean(episode_history)

    print("Episode: {}, Steps: {}, TReward: {}, Score: {}".format(i_episode, t + 1, total_rewards, info))
    report_measures = ([tf.Summary.Value(tag='score', simple_value=info),
                        tf.Summary.Value(tag='number_of_steps', simple_value=t + 1)])
    writer.add_summary(tf.Summary(value=report_measures), i_episode)
