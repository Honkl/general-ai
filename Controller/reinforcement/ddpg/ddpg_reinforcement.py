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
    """
    Represents a DDPG model.
    """

    def __init__(self, game, parameters, logs_every=10):
        """
        Initializes a new DDPG model using specified parameters.
        :param game: Game that will be played.
        :param parameters: Parameters of the model. (DDPGParameters)
        :param logs_every: At each n-th episode model will be saved.
        """
        self.game = game
        self.episodes = parameters.episodes
        self.batch_size = parameters.batch_size
        self.game_config = utils.miscellaneous.get_game_config(game)
        self.game_class = utils.miscellaneous.get_game_class(game)
        self.state_size = self.game_config["input_sizes"][0]  # inputs for all phases are the same in our games
        self.logs_every = logs_every

        self.actions_count = self.game_config["output_sizes"]
        self.actions_count_sum = sum(self.actions_count)
        self.logdir = self.init_directories()

        # DDPG (deep deterministic gradient policy)
        self.agent = DDPGAgent(self.batch_size, self.state_size, self.actions_count_sum, self.logdir)

    def init_directories(self):
        """
        Initializes a directories for log files.
        :return: Current logdir.
        """
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
        """
        Saves model metadata into a logdir.
        """
        with open(os.path.join(self.logdir, "metadata.json"), "w") as f:
            data = {}
            data["model_name"] = "reinforcement_learning_ddpg"
            data["game"] = self.game
            data["batch_size"] = self.batch_size
            data["episodes"] = self.episodes
            f.write(json.dumps(data))

    def run(self):
        """
        Starts an evaluation of DDPG model.
        """
        self.log_metadata()

        start = time.time()
        data = []
        for episode in range(self.episodes):

            self.env = Environment(game_class=self.game_class,
                                   seed=np.random.randint(0, 2 ** 16),
                                   observations_count=self.state_size,
                                   actions_in_phases=self.actions_count)
            state = self.env.state
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
                with open(os.path.join(self.logdir, "logbook.txt"), "w") as f:
                    for line in data:
                        f.write(line)
                        f.write('\n')

            now = time.time()
            t = now - start
            h = t // 3600
            m = (t % 3600) // 60
            s = t - (h * 3600) - (m * 60)
            elapsed_time = "{}h {}m {}s".format(int(h), int(m), s)
            line = "Episode {}/{}, Score: {}, Steps: {}, Total Time: {}".format(episode, self.episodes, score, step,
                                                                                elapsed_time)
            print(line)
            data.append(line)

        self.env.close()

    def load_checkpoint(self, checkpoint):
        """
        Loads tensorflow checkpoint.
        :param checkpoint: Checkpoint to be loaded.
        """
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring model: {}'.format(ckpt.model_checkpoint_path))
            saver.restore(self.agent.sess, ckpt.model_checkpoint_path)
        else:
            raise IOError('No model found in {}.'.format(checkpoint))
