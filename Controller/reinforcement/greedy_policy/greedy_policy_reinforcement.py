import json
import os
import time

import numpy as np
import tensorflow as tf

import constants
import utils.miscellaneous
from reinforcement.environment import Environment
from reinforcement.greedy_policy.greedy_policy_agent import GreedyPolicyAgent
from reinforcement.abstract_reinforcement import AbstractReinforcement
from copy import deepcopy


class GreedyPolicyReinforcement(AbstractReinforcement):
    def __init__(self, game, parameters, q_network, threads=8, logs_every=10):
        self.game = game
        self.parameters = parameters
        self.q_network = q_network
        self.threads = threads
        self.logs_every = logs_every

        self.game_config = utils.miscellaneous.get_game_config(game)
        self.game_class = utils.miscellaneous.get_game_class(game)
        self.state_size = self.game_config["input_sizes"][0]  # inputs for all phases are the same in our games

        # we will train only one network inside the "Q-network" (not different networks, each for each game phase)
        self.actions_count = self.game_config["output_sizes"]
        self.actions_count_sum = sum(self.actions_count)
        self.logdir = self.init_directories(dir_name="greedy_policy")
        self.checkpoint_name = "greedy_policy.ckpt"

        q_network.init(self.actions_count_sum, self.parameters.batch_size)
        self.agent = GreedyPolicyAgent(parameters, q_network, self.state_size, self.actions_count_sum, self.logdir,
                                       threads)

    def log_metadata(self):
        with open(os.path.join(self.logdir, "metadata.json"), "w") as f:
            data = {}
            data["model_name"] = "reinforcement_learning_greedy_policy"
            data["game"] = self.game
            data["q-network"] = self.q_network.to_dictionary()
            data["parameters"] = self.parameters.to_dictionary()
            f.write(json.dumps(data))

    def run(self):
        self.log_metadata()
        episodes = self.parameters.episodes

        start = time.time()
        data = []
        print_timer = time.time()

        # One epoch = One episode = One game played
        for i_episode in range(1, episodes + 1):

            self.env = Environment(game_class=self.game_class,
                                   seed=np.random.randint(0, 2 ** 16),
                                   observations_count=self.state_size,
                                   actions_in_phases=self.actions_count,
                                   discrete=True)

            epoch_loss = 0.0
            epoch_reward = 0.0
            epoch_score = 0.0
            epoch_estimated_reward = 0.0
            game_steps = 0
            episode_start_time = time.time()

            # Running the game until it is not done (big step limit for safety)
            while game_steps < self.STEP_LIMIT:
                game_steps += 1

                old_state = deepcopy(self.env.state)
                selected_action, estimated_reward = self.agent.play(old_state)
                epoch_estimated_reward += estimated_reward

                # Perform the action
                new_state, reward, done, score = self.env.step(selected_action)
                epoch_reward += reward

                loss = self.agent.learn(old_state, selected_action, reward, new_state, done)

                if loss:
                    # Waiting until we'll get enough experiences in replay buffer
                    epoch_loss += loss

                if done:
                    epoch_score = score
                    break

            episode_time = utils.miscellaneous.get_elapsed_time(episode_start_time)
            line = "Episode: {}/{}, Score: {}, Avg Loss: {}, Episode Time: {}".format(i_episode, episodes, epoch_score,
                                                                                      "{0:.5f}".format(
                                                                                          epoch_loss / game_steps),
                                                                                      episode_time)
            data.append(line)
            if time.time() - print_timer > 1 or i_episode == episodes:
                print(line)
                print_timer = time.time()

            if i_episode % self.logs_every == 0:
                self.test_and_save(log_data=data, start_time=start, i_episode=i_episode)

            report_measures = ([tf.Summary.Value(tag='loss_total', simple_value=epoch_loss),
                                tf.Summary.Value(tag='loss_average', simple_value=float(epoch_loss) / game_steps),
                                tf.Summary.Value(tag='score', simple_value=epoch_score),
                                tf.Summary.Value(tag='reward_total', simple_value=epoch_reward),
                                tf.Summary.Value(tag='reward_average', simple_value=float(epoch_reward) / game_steps),
                                tf.Summary.Value(tag='estimated_reward_total', simple_value=epoch_estimated_reward),
                                tf.Summary.Value(tag='estimated_reward_average',
                                                 simple_value=float(epoch_estimated_reward) / game_steps),
                                tf.Summary.Value(tag='number_of_steps', simple_value=game_steps)])
            self.agent.summary_writer.add_summary(tf.Summary(value=report_measures), i_episode)

        self.env.shut_down()

    def test(self, n_iterations):
        avg_test_score = 0
        for i in range(n_iterations):
            env = Environment(game_class=self.game_class,
                              seed=np.random.randint(0, 2 ** 16),
                              observations_count=self.state_size,
                              actions_in_phases=self.actions_count,
                              discrete=True)
            game_steps = 0
            while game_steps < self.STEP_LIMIT:
                game_steps += 1

                old_state = env.state
                selected_action, estimated_reward = self.agent.play(env.state)

                # Perform the action
                new_state, reward, done, score = env.step(selected_action)

                if done:
                    avg_test_score += score
                    break

        avg_test_score /= n_iterations
        return avg_test_score
