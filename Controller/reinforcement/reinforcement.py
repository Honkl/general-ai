import numpy as np
import constants
import json
import os
import time
from reinforcement.agent import Agent
from reinforcement.environment import Environment
import tensorflow as tf


class Reinforcement():
    def __init__(self, game, reinforce_params, q_network, threads):
        self.game = game
        self.reinforce_params = reinforce_params
        self.q_network = q_network
        self.threads = threads

        game_config_file = ""
        if game == "alhambra":
            game_config_file = constants.ALHAMBRA_CONFIG_FILE
        if game == "2048":
            game_config_file = constants.GAME2048_CONFIG_FILE
        if game == "mario":
            game_config_file = constants.MARIO_CONFIG_FILE
        if game == "torcs":
            game_config_file = constants.TORCS_CONFIG_FILE

        with open(game_config_file, "r") as f:
            self.game_config = json.load(f)
        self.agents = []
        self.state_size = self.game_config["input_sizes"][0]  # inputs for all phases are the same in our games

        # we will train only one network inside the Q-network
        self.actions_count = sum(self.game_config["output_sizes"])

        expname = "game{}-penalty{}-gamma{}-base_reward{}".format(game, reinforce_params.penalty,
                                                                  reinforce_params.gamma,
                                                                  reinforce_params.base_reward)

        self.agent = Agent(reinforce_params, q_network, self.state_size, self.actions_count, expname, threads)

    def init_directories(self):
        self.dir = constants.loc + "/logs/" + self.game + "/q-network"
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        # create name for directory to store logs
        current = time.localtime()
        t_string = "{}-{}-{}_{}-{}-{}".format(str(current.tm_year).zfill(2),
                                              str(current.tm_mon).zfill(2),
                                              str(current.tm_mday).zfill(2),
                                              str(current.tm_hour).zfill(2),
                                              str(current.tm_min).zfill(2),
                                              str(current.tm_sec).zfill(2))

        return self.dir + "/logs_" + t_string

    def run(self):
        logdir = self.init_directories()
        epochs = self.reinforce_params.epochs

        for i_epoch in range(epochs):
            print("Epoch {}/{}".format(i_epoch, epochs))

            # Gym Environment
            envs = [Environment(self.reinforce_params.base_reward,
                                self.reinforce_params.penalty,
                                np.random.randint(0, 2 ** 16),
                                self.state_size,
                                self.actions_count) for _ in range(self.reinforce_params.batch_size)]

            for env in envs:
                env.reset()

            epoch_loss = 0.0
            epoch_reward = 0.0
            epoch_estimated_reward = 0.0
            max_reward = 0
            step_id = -1

            while True:
                step_id += 1
                # TODO: Set step limit?

                # Evaluate action (forward pass in Q-net) and apply it
                selected_actions, estimated_rewards = self.agent.play([env.state for env in envs])
                epoch_estimated_reward += estimated_rewards.mean()

                _, rewards, dones, _ = zip(*[env.step(action) for env, action in zip(envs, selected_actions)])
                epoch_reward += np.array(rewards).mean()

                # Train Q-net
                epoch_loss += self.agent.learn([env.state for env in envs], selected_actions, rewards,
                                               estimated_rewards)
                max_reward = max(max_reward, np.max(rewards))

                if sum(dones):
                    break

            report_measures = ([tf.Summary.Value(tag='loss_total', simple_value=epoch_loss),
                                tf.Summary.Value(tag='loss_average', simple_value=float(epoch_loss) / step_id),
                                tf.Summary.Value(tag='reward_total', simple_value=epoch_reward),
                                tf.Summary.Value(tag='reward_average', simple_value=float(epoch_reward) / step_id),
                                tf.Summary.Value(tag='estimated_reward_total', simple_value=epoch_estimated_reward),
                                tf.Summary.Value(tag='estimated_reward_average',
                                                 simple_value=float(epoch_estimated_reward) / step_id),
                                tf.Summary.Value(tag='number_of_turns', simple_value=step_id),
                                tf.Summary.Value(tag='max_reward', simple_value=max_reward)])
            self.agent.summary_writer.add_summary(tf.Summary(value=report_measures), i_epoch)
            print('Avg loss: {}'.format(float(epoch_loss) / step_id))

        # TODO: Make better
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        with open(os.path.join(logdir, self.expname), "w") as f:
            f.write(str(epoch_loss) + "\n")
            f.write(str(float(epoch_loss) / step_id) + "\n")
            f.write(str(epoch_reward) + "\n")
            f.write(str(float(epoch_reward) / step_id) + "\n")
            f.write(str(epoch_estimated_reward) + "\n")
            f.write(str(float(epoch_estimated_reward) / step_id) + "\n")
