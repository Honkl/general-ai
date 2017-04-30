import json
import os

from models.abstract_model import AbstractModel
from reinforcement.ddpg.ddpg_reinforcement import DDPGReinforcement
from reinforcement.reinforcement_parameters import DDPGParameters


class LearnedDDPG(AbstractModel):
    """
    Represents a learned DDPG model. This is only an interface wrapper. Uses tensorflow internally.
    """

    def __init__(self, logdir):
        """
        Initializes tensorflow DDPG model, using specified directory.
        :param logdir: Directory to tensorflow checkpoint.
        """
        self.metadata = None
        for file in os.listdir(logdir):
            if file.endswith(".json"):
                with open(os.path.join(logdir, file), "r") as f:
                    self.metadata = json.load(f)
                    break

        self.game = self.metadata["game"]
        self.parameters = DDPGParameters.from_dict(self.metadata["parameters"])
        self.ddpg = DDPGReinforcement(self.game, self.parameters)
        self.ddpg.load_checkpoint(os.path.join(logdir, "last"))

    def get_new_instance(self, weights, game_config):
        raise NotImplementedError

    def evaluate(self, input, current_phase):
        """
        Evaluates the model result, using the specified input and current game phase.
        :param input: Input for the model.
        :param current_phase: Current game phase.
        :return: Action.
        """
        action = self.ddpg.agent.play(input)
        return action

    def get_name(self):
        """
        Returns a string representation of the current model.
        :return: a string representation of hte current model.
        """
        return "Learned DDPG Agent (Reinforcement Learning)"

    def get_class_name(self):
        """
        Returns a class name of the current model.
        :return: a class name of the current model.
        """
        return "LearnedDDPG"
