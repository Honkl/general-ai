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
import time
import json

gc.enable()


class DDPGReinforcement(AbstractReinforcement):
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
        self.agent = DDPGAgent(self.batch_size, self.state_size, self.actions_count_sum, self.logdir)

    def log_metadata(self):
        """
        Saves model metadata into a logdir.
        """
        with open(os.path.join(self.logdir, "metadata.json"), "w") as f:
            data = {}
            data["model_name"] = "reinforcement_learning_ddpg"
            data["game"] = self.game
            data["parameters"] = self.parameters.to_dictionary()
            f.write(json.dumps(data))

    def run(self):
        """
        Starts an evaluation of DDPG model.
        """
        self.log_metadata()

        start = time.time()
        data = []
        tmp = time.time()
        for i_episode in range(1, self.episodes + 1):

            episode_start_time = time.time()
            self.env = Environment(game_class=self.game_class,
                                   seed=np.random.randint(0, 2 ** 30),
                                   observations_count=self.state_size,
                                   actions_in_phases=self.actions_count)

            state = self.env.state
            for step in range(self.STEP_LIMIT):
                action = self.agent.play(state)
                next_state, reward, done, score = self.env.step(action)
                self.agent.perceive(state, action, reward, next_state, done)
                state = next_state

                if done:
                    break

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

    def test(self, n_iterations):
        avg_test_score = 0
        for i in range(n_iterations):
            env = Environment(game_class=self.game_class,
                              seed=np.random.randint(0, 2 ** 30),
                              observations_count=self.state_size,
                              actions_in_phases=self.actions_count)
            game_steps = 0
            state = env.state
            while game_steps < self.STEP_LIMIT:
                game_steps += 1

                # Get the action
                selected_action = self.agent.play(state)

                # Perform the action
                new_state, reward, done, score = env.step(selected_action)
                state = new_state

                if done:
                    avg_test_score += score
                    break

        avg_test_score /= n_iterations
        return avg_test_score
