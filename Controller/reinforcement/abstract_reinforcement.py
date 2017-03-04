import os
import time
import numpy as np
import tensorflow as tf

import constants
import utils.miscellaneous


class AbstractReinforcement():
    """
    Interface for DDPGReinforcement and GreedyPolicyReinforcement
    """

    STEP_LIMIT = 50000

    """ Overwritten in subclasses """
    game = None
    game_class = None
    checkpoint_name = None
    actions_count = None
    state_size = None
    agent = None
    parameters = None
    logdir = None

    best_test_score = -np.Inf
    test_logbook_data = []
    test_every = None

    def init_directories(self, dir_name):
        """
        Initializes a directories for log files.
        :param dir_name: Name of the directory (usually model name).
        :return: Current logdir.
        """

        self.dir = constants.loc + "/logs/" + self.game + "/" + dir_name
        # create name for directory to store logs
        logdir = self.dir + "/logs_" + utils.miscellaneous.get_pretty_time()
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

    def test_and_save(self, log_data, start_time, i_episode):

        print("Testing model... [{} runs]".format(self.parameters.test_size))
        current_score = self.test(self.parameters.test_size)
        line = "Current score: {}, Best score: {}".format(current_score, self.best_test_score)
        print(line)
        self.test_logbook_data.append(line)
        if (current_score > self.best_test_score):
            print("Saving model...")
            checkpoint_path = os.path.join(self.logdir, self.checkpoint_name)
            self.agent.saver.save(self.agent.sess, checkpoint_path)
            self.best_test_score = current_score

        elapsed_time = utils.miscellaneous.get_elapsed_time(start_time)
        t = "Total time: {}".format(elapsed_time)
        print(t)
        self.test_logbook_data.append(t)

        with open(os.path.join(self.logdir, "logbook.txt"), "w") as f:
            for line in log_data:
                f.write(line)
                f.write('\n')

        with open(os.path.join(self.logdir, "test_logbook.txt"), "w") as f:
            for line in self.test_logbook_data:
                f.write(line)
                f.write('\n')

        test_measure = ([tf.Summary.Value(tag='score_test', simple_value=current_score)])
        self.agent.summary_writer.add_summary(tf.Summary(value=test_measure), i_episode)

    def test(self, n_iterations):
        raise NotImplementedError
