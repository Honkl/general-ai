from  reinforcement.ddpg.filter_env import *
from reinforcement.ddpg.ddpgagent import DDPGAgent
from reinforcement.environment import Environment
import utils.miscellaneous
import tensorflow as tf
import gym
import gc
import os
import matplotlib.pyplot as plt
import numpy as np
gc.enable()

EPISODES = 100000
TEST = 10

def logs(scores):
    plt.figure()
    if len(scores) <= 100:
        plt.title("AVG (last 100): {}".format(np.mean(scores)))
    else:
        plt.title("AVG (last 100): {}".format(np.mean(scores[-100:])))
    plt.plot(scores, label="score")
    plt.savefig("plot.jpg")


def main():
    game = "2048"

    game_config = utils.miscellaneous.get_game_config(game)
    game_class = utils.miscellaneous.get_game_class(game)
    state_size = game_config["input_sizes"][0]  # inputs for all phases are the same in our games
    actions_count = game_config["output_sizes"]

    env = Environment(game_class, np.random.randint(0, 2 ** 16), state_size, actions_count)
    env = makeFilteredEnv(env)
    agent = DDPGAgent(env)

    scores = []
    for episode in range(EPISODES):
        state = env.reset()
        # print "episode:",episode
        # Train
        for step in range(100000):
            action = agent.noise_action(state)
            next_state, reward, done, score = env.step(action)
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        scores.append(score)
        print("Episode {}, Score: {}, Steps: {}".format(episode, score, step))
        if episode % 10:
            logs(scores)
    env.close()


if __name__ == '__main__':
    with tf.device('/gpu:0'):
        main()
