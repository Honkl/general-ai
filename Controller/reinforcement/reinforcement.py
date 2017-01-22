import numpy as np
import constants
import json
import os
import time
from reinforcement.agent import Agent
from reinforcement.environment import Environment
import tensorflow as tf

import utils.miscellaneous
from games.alhambra import Alhambra
from games.torcs import Torcs
from games.mario import Mario
from games.game2048 import Game2048


class Reinforcement():
    def __init__(self, game, reinforce_params, q_network, threads):
        self.game = game
        self.reinforce_params = reinforce_params
        self.q_network = q_network
        self.threads = threads

        self.game_config = utils.miscellaneous.get_game_config(game)
        self.game_class = utils.miscellaneous.get_game_class(game)
        self.agents = []
        self.state_size = self.game_config["input_sizes"][0]  # inputs for all phases are the same in our games

        # we will train only one network inside the Q-network
        self.actions_count = sum(self.game_config["output_sizes"])
        q_network.set_output_size(self.actions_count)

        self.logdir = self.init_directories()
        self.agent = Agent(reinforce_params, q_network, self.state_size, self.actions_count, self.logdir, threads)

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

    def log_metadata(self):
        with open(os.path.join(self.logdir, "metadata.json"), "w") as f:
            data = {}
            data["model_name"] = "Q-Network"
            data["game"] = self.game
            data["q_network"] = self.q_network.to_dictionary()
            data["reinforce_params"] = self.reinforce_params.to_dictionary()
            f.write(json.dumps(data))

    def run(self):
        self.log_metadata()
        epochs = self.reinforce_params.epochs
        max_score = 0.0

        for i_epoch in range(1, epochs + 1):
            # Gym Environment
            envs = [Environment(self.game_class,
                                self.reinforce_params.base_reward,
                                self.reinforce_params.penalty,
                                np.random.randint(0, 2 ** 16),
                                self.state_size,
                                self.actions_count) for _ in range(self.reinforce_params.batch_size)]

            epoch_loss = 0.0
            epoch_reward = 0.0
            epoch_score = 0.0
            epoch_estimated_reward = 0.0
            max_reward = 0
            step_id = 0

            while True:
                step_id += 1
                # TODO: Set step limit?
                # Evaluate action (forward pass in Q-net) and apply it
                selected_actions, estimated_rewards = self.agent.play([env.state for env in envs])
                epoch_estimated_reward += estimated_rewards.mean()

                _, rewards, dones, scores = zip(*[env.step(action) for env, action in zip(envs, selected_actions)])
                epoch_reward += np.array(rewards).mean()

                # Train Q-net
                epoch_loss += self.agent.learn([env.state for env in envs], selected_actions, rewards,
                                               estimated_rewards)
                max_reward = max(max_reward, np.max(rewards))

                if sum(dones):
                    epoch_score = max(scores)[0]  # Max score from batch
                    break

            report_measures = ([tf.Summary.Value(tag='loss_total', simple_value=epoch_loss),
                                tf.Summary.Value(tag='loss_average', simple_value=float(epoch_loss) / step_id),
                                tf.Summary.Value(tag='score', simple_value=epoch_score),
                                tf.Summary.Value(tag='reward_total', simple_value=epoch_reward),
                                tf.Summary.Value(tag='reward_average', simple_value=float(epoch_reward) / step_id),
                                tf.Summary.Value(tag='estimated_reward_total', simple_value=epoch_estimated_reward),
                                tf.Summary.Value(tag='estimated_reward_average',
                                                 simple_value=float(epoch_estimated_reward) / step_id),
                                tf.Summary.Value(tag='number_of_turns', simple_value=step_id),
                                tf.Summary.Value(tag='max_reward', simple_value=max_reward)])
            self.agent.summary_writer.add_summary(tf.Summary(value=report_measures), i_epoch)

            for env in envs:
                env.shut_down()

            if epoch_score >= max_score:
                checkpoint_path = os.path.join(self.logdir, "q-net-model.ckpt")
                self.agent.saver.save(self.agent.session, checkpoint_path)

            # print("Epoch: {}/{} Avg loss: {}".format(i_epoch, epochs, float(epoch_loss) / step_id))
            print("Epoch: {}/{} Avg score: {}".format(i_epoch, epochs, epoch_score))


        # TODO: Make better
        """
        if not os.path.isdir(logdir):
            os.mkdir(logdir)
        with open(os.path.join(logdir, self.expname + ".txt"), "w") as f:
            f.write(str(epoch_loss) + "\n")
            f.write(str(float(epoch_loss) / step_id) + "\n")
            f.write(str(epoch_reward) + "\n")
            f.write(str(float(epoch_reward) / step_id) + "\n")
            f.write(str(epoch_estimated_reward) + "\n")
            f.write(str(float(epoch_estimated_reward) / step_id) + "\n")
        """

    def load_checkpoint(self, checkpoint):
        #tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring model: {}'.format(ckpt.model_checkpoint_path))
            saver.restore(self.agent.session, ckpt.model_checkpoint_path)
        else:
            raise IOError('No model found in {}.'.format(checkpoint))