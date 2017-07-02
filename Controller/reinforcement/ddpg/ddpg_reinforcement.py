from  reinforcement.ddpg.filter_env import *
from reinforcement.ddpg.ddpg_agent import DDPGAgent
from reinforcement.environment import Environment
from reinforcement.abstract_reinforcement import AbstractReinforcement
import utils.miscellaneous
import tensorflow as tf
import gc
import os
import numpy as np
import constants
import concurrent.futures
import time
import json

gc.enable()

EPISODE_TIME_LIMIT_SEC = 1000


class DDPGReinforcement(AbstractReinforcement):
    """
    Represents a DDPG model.
    """

    def __init__(self, game, parameters, logs_every=10, checkpoint=None):
        """
        Initializes a new DDPG model using specified parameters.
        :param game: Game that will be played.
        :param parameters: Parameters of the model. (DDPGParameters)
        :param logs_every: At each n-th episode model will be saved.
        :param checkpoint: Checkpoint to begin with.
        """
        self.game = game
        self.parameters = parameters
        self.episodes = parameters.episodes
        self.batch_size = parameters.batch_size
        self.game_config = utils.miscellaneous.get_game_config(game)
        self.game_class = utils.miscellaneous.get_game_class(game)
        self.state_size = self.game_config["input_sizes"][0]  # inputs for all phases are the same in our games
        self.logs_every = logs_every
        self.checkpoint_name = "ddpg.ckpt"

        self.actions_count = self.game_config["output_sizes"]
        self.actions_count_sum = sum(self.actions_count)
        self.logdir = self.init_directories(dir_name="ddpg")

        # DDPG (deep deterministic gradient policy)
        self.agent = DDPGAgent(parameters.replay_buffer_size, parameters.discount_factor, self.batch_size,
                               self.state_size, self.actions_count_sum, self.logdir)


        self.start_episode = 1
        if checkpoint:
            print("Train started with checkpoint: {}".format(checkpoint))
            with open(checkpoint + "logbook.txt", "r") as f:
                lines = len(f.readlines())
                print(lines)
                self.start_episode = lines

            self.logdir = checkpoint
            self.load_checkpoint(checkpoint + "/last")

    def log_metadata(self):
        """
        Saves model metadata into a logdir.
        """
        with open(os.path.join(self.logdir, "metadata.json"), "w") as f:
            data = {}
            data["model_name"] = "reinforcement_learning_ddpg"
            data["game"] = self.game
            data["parameters"] = self.parameters.to_dictionary()
            data["actor_network"] = self.agent.actor_parameters
            data["critic_network"] = self.agent.critic_parameters
            f.write(json.dumps(data))

    def run(self):
        """
        Starts an evaluation of DDPG model.
        """
        self.log_metadata()

        start = time.time()
        data = []
        tmp = time.time()
        for i_episode in range(self.start_episode, self.episodes + 1):

            success = False
            # Avoiding game internal error (subprocess fail etc.)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                while not success:
                    episode_start_time = time.time()
                    future = executor.submit(self.get_episode_results)
                    try:
                        score, step = future.result(EPISODE_TIME_LIMIT_SEC)
                        success = True
                    except concurrent.futures.TimeoutError:
                        print("Episode limit time limit exceeded ({} sec).".format(EPISODE_TIME_LIMIT_SEC))
                        self.env.shut_down(internal_error=True)
                        time.sleep(3)
                        print("Starting new game...")

            episode_time = utils.miscellaneous.get_elapsed_time(episode_start_time)
            line = "Episode {}, Score: {}, Steps: {}, Episode Time: {}".format(i_episode, score, step,
                                                                               episode_time)

            if time.time() - tmp > 1:
                print(line)
                tmp = time.time()
            data.append(line)

            if i_episode % self.logs_every == 0:
                self.test_and_save(log_data=data, start_time=start, i_episode=i_episode)

            report_measures = ([tf.Summary.Value(tag='score', simple_value=score),
                                tf.Summary.Value(tag='number_of_steps', simple_value=step)])
            self.agent.summary_writer.add_summary(tf.Summary(value=report_measures), i_episode)

        self.env.close()

    def get_episode_results(self, test=False):
        self.env = Environment(game_class=self.game_class,
                               seed=np.random.randint(0, 2 ** 30),
                               observations_count=self.state_size,
                               actions_in_phases=self.actions_count,
                               test=test)

        state = self.env.state
        for step in range(self.STEP_LIMIT):
            action = self.agent.play(state)
            next_state, reward, done, score = self.env.step(action)

            if not test:
                self.agent.perceive(state, action, reward, next_state, done)

            state = next_state
            if done:
                return score, step

    def test(self, n_iterations):
        avg_test_score = 0
        for i in range(n_iterations):

            success = False
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                while not success:
                    future = executor.submit(self.get_episode_results, test=True)
                    try:
                        score, step = future.result(EPISODE_TIME_LIMIT_SEC)
                        success = True
                    except concurrent.futures.TimeoutError:
                        print("Episode [test] limit time limit exceeded ({} sec).".format(EPISODE_TIME_LIMIT_SEC))
                        self.env.shut_down(internal_error=True)
                        time.sleep(3)

            avg_test_score += score

        avg_test_score /= n_iterations
        return avg_test_score
