import json
import os
import numpy as np

from models.abstract_model import AbstractModel
from reinforcement.dqn.dqn import DQN
from reinforcement.reinforcement_parameters import DQNParameters


class LearnedDQN(AbstractModel):
    """
    Represents a learned greedy-policy reinforcement learning model. This is only an interface wrapper.
    Uses tensorflow internally.
    """

    def __init__(self, logdir):
        """
        Initializes tensorflow greedy-policy RL model, using specified directory.
        :param logdir: Directory to tensorflow checkpoint.
        """
        self.metadata = None
        for file in os.listdir(logdir):
            if file.endswith(".json"):
                with open(os.path.join(logdir, file), "r") as f:
                    self.metadata = json.load(f)
                    break

        net = self.metadata["q_network"]
        params = self.metadata["parameters"]
        optimizer_params = self.metadata["optimizer_parameters"]
        self.game = self.metadata["game"]
        self.dqn = DQN(self.game, DQNParameters.from_dict(params), net, optimizer_params)
        self.dqn.load_checkpoint(os.path.join(logdir, "last"))

    def get_new_instance(self, weights, game_config):
        raise NotImplementedError

    def evaluate(self, input, current_phase):
        """
        Evaluates the model result, using the specified input and current game phase.
        :param input: Input for the model.
        :param current_phase: Current game phase.
        :return: Action.
        """
        action = self.dqn.agent.eGreedyAction(input[np.newaxis, :])
        return self.dqn.convert_to_sequence(action)

    def get_name(self):
        """
        Returns a string representation of the current model.
        :return: a string representation of hte current model.
        """
        return "Learned Greedy Policy (Reinforcement Learning) [DQN]"

    def get_class_name(self):
        """
        Returns a class name of the current model.
        :return: a class name of the current model.
        """
        return "LearnedDQN"
