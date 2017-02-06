from  reinforcement.ddpg.filter_env import *
from reinforcement.ddpg.ddpg_agent import DDPGAgent
from reinforcement.environment import Environment
import utils.miscellaneous
import tensorflow as tf
import gc
import os
import matplotlib.pyplot as plt
import numpy as np
import constants
import time
import json

gc.enable()


def logs(scores):
    plt.figure()
    if len(scores) <= 100:
        plt.title("AVG (last 100): {}".format(np.mean(scores)))
    else:
        plt.title("AVG (last 100): {}".format(np.mean(scores[-100:])))
    plt.plot(scores, label="score")
    plt.savefig("plot.jpg")


class DDPGReinforcement():
    def __init__(self, game, episodes, batch_size):
        self.game = game
        self.episodes = episodes
        self.batch_size = batch_size
        self.game_config = utils.miscellaneous.get_game_config(game)
        self.game_class = utils.miscellaneous.get_game_class(game)
        self.state_size = self.game_config["input_sizes"][0]  # inputs for all phases are the same in our games

        actions_count = self.game_config["output_sizes"]
        self.actions_count_sum = sum(actions_count)
        self.logdir = self.init_directories()

        # DDPG (deep deterministic gradient policy)
        self.env = Environment(discrete=False,
                               game_class=self.game_class,
                               seed=np.random.randint(0, 2 ** 16),
                               observations_count=self.state_size,
                               actions_in_phases=actions_count)
        self.agent = DDPGAgent(self.env, batch_size, self.state_size, self.actions_count_sum)

    def init_directories(self):
        self.dir = constants.loc + "/logs/" + self.game + "/deep_deterministic_gradient_policy"
        # create name for directory to store logs
        current = time.localtime()
        t_string = "{}-{}-{}_{}-{}-{}".format(str(current.tm_year).zfill(2),
                                              str(current.tm_mon).zfill(2),
                                              str(current.tm_mday).zfill(2),
                                              str(current.tm_hour).zfill(2),
                                              str(current.tm_min).zfill(2),
                                              str(current.tm_sec).zfill(2))
        logdir = self.dir + "/logs_" + t_string
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        return logdir

    def log_metadata(self):
        with open(os.path.join(self.logdir, "metadata.json"), "w") as f:
            data = {}
            data["model_name"] = "reinforcement_learning_ddpg"
            data["game"] = self.game
            data["batch_size"] = self.batch_size
            data["episodes"] = self.episodes
            f.write(json.dumps(data))

    def run(self):
        game_config = utils.miscellaneous.get_game_config(self.game)
        game_class = utils.miscellaneous.get_game_class(self.game)

        with tf.device('/gpu:0'):
            scores = []
            for episode in range(self.episodes):
                state = self.env.reset()
                # print "episode:",episode
                # Train
                for step in range(100000):
                    action = self.agent.play(state, episode)
                    next_state, reward, done, score = self.env.step(action)
                    self.agent.perceive(state, action, reward, next_state, done)
                    state = next_state
                    if done:
                        break
                scores.append(score)
                print("Episode {}, Score: {}, Steps: {}".format(episode, score, step))
                if episode % 10:
                    logs(scores)
            self.env.close()
