from  reinforcement.ddpg.filter_env import *
from reinforcement.ddpg.ddpg_agent import DDPGAgent
from reinforcement.environment import Environment
import utils.miscellaneous
import tensorflow as tf
import gc
import os
import numpy as np
import constants
import time
import json

gc.enable()


class DDPGReinforcement():
    def __init__(self, game, episodes, batch_size, logs_every=10):
        self.game = game
        self.episodes = episodes
        self.batch_size = batch_size
        self.game_config = utils.miscellaneous.get_game_config(game)
        self.game_class = utils.miscellaneous.get_game_class(game)
        self.state_size = self.game_config["input_sizes"][0]  # inputs for all phases are the same in our games
        self.logs_every = logs_every

        actions_count = self.game_config["output_sizes"]
        self.actions_count_sum = sum(actions_count)
        self.logdir = self.init_directories()

        # DDPG (deep deterministic gradient policy)
        self.env = Environment(discrete=False,
                               game_class=self.game_class,
                               seed=np.random.randint(0, 2 ** 16),
                               observations_count=self.state_size,
                               actions_in_phases=actions_count)
        self.agent = DDPGAgent(self.env, batch_size, self.state_size, self.actions_count_sum, self.logdir)

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
        self.log_metadata()

        with tf.device('/cpu:0'):

            max_score = 0
            for episode in range(self.episodes):
                state = self.env.reset()
                for step in range(100000):
                    action = self.agent.play(state, episode)
                    next_state, reward, done, score = self.env.step(action)
                    score = score[0]
                    self.agent.perceive(state, action, reward, next_state, done)
                    state = next_state
                    if done:
                        break

                report_measures = ([tf.Summary.Value(tag='score', simple_value=score),
                                    tf.Summary.Value(tag='number_of_steps', simple_value=step)])
                self.agent.summary_writer.add_summary(tf.Summary(value=report_measures), episode)

                if episode % self.logs_every == 0:
                    checkpoint_path = os.path.join(self.logdir, "ddpg.ckpt")
                    self.agent.saver.save(self.agent.sess, checkpoint_path)
                print("Episode {}, Score: {}, Steps: {}".format(episode, score, step))
            self.env.close()

    def load_checkpoint(self, checkpoint):
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring model: {}'.format(ckpt.model_checkpoint_path))
            saver.restore(self.agent.sess, ckpt.model_checkpoint_path)
        else:
            raise IOError('No model found in {}.'.format(checkpoint))
