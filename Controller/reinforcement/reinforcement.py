import json
import os
import time

import numpy as np
import tensorflow as tf

import constants
from reinforcement.environment import Environment


class Reinforcement():
    """
    Interface for DDPGReinforcement and GreedyPolicyReinforcement
    """

    STEP_LIMIT = 1000000  # 1M

    """ Overwritten in subclasses """
    game = None
    game_class = None
    checkpoint_name = None
    actions_count = None
    state_size = None
    agent = None

    def init_directories(self, dir_name):
        """
        Initializes a directories for log files.
        :param dir_name: Name of the directory (usually model name).
        :return: Current logdir.
        """
        self.dir = constants.loc + "/logs/" + self.game + "/" + dir_name
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
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def load_checkpoint(self, checkpoint):
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring model: {}'.format(checkpoint))
            saver.restore(self.agent.sess, os.path.join(checkpoint, self.checkpoint_name))
        else:
            raise IOError('No model found in {}.'.format(checkpoint))

    def test(self, n_iterations):
        avg_test_score = 0
        for i in range(n_iterations):
            env = Environment(game_class=self.game_class,
                              seed=np.random.randint(0, 2 ** 16),
                              observations_count=self.state_size,
                              actions_in_phases=self.actions_count)
            game_steps = 0
            while game_steps < self.STEP_LIMIT:
                game_steps += 1

                old_state = env.state
                selected_action, estimated_reward = self.agent.play(env.state)

                # Perform the action
                new_state, reward, done, score = env.step(selected_action)

                if done:
                    avg_test_score += score[0]
                    break

        avg_test_score /= n_iterations
        return avg_test_score
