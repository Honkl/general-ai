import json
import os

from models.model import Model
from reinforcement.greedy_policy.greedy_policy_reinforcement import GreedyPolicyReinforcement
from reinforcement.greedy_policy.q_network import QNetwork
from reinforcement.reinforcement_parameters import GreedyPolicyParameters


class LearnedGreedyRL(Model):
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

        q_net_params = self.metadata["q-network"]
        rl_params = self.metadata["parameters"]
        self.game = self.metadata["game"]
        self.q_network = QNetwork(q_net_params["hidden_layers"], q_net_params["activation"])

        self.rl = GreedyPolicyReinforcement(self.game, GreedyPolicyParameters.from_dict(rl_params), self.q_network,
                                            threads=1)
        self.rl.load_checkpoint(logdir)

    def get_new_instance(self, weights, game_config):
        raise NotImplementedError

    def evaluate(self, input, current_phase):
        """
        Evaluates the model result, using the specified input and current game phase.
        :param input: Input for the model.
        :param current_phase: Current game phase.
        :return: Action.
        """
        action, estimated_reward = self.rl.agent.play(input, None)
        action_string = ""
        for a in action:
            action_string += str(a) + " "
        return action_string

    def get_name(self):
        """
        Returns a string representation of the current model.
        :return: a string representation of hte current model.
        """
        return "Learned Greedy Policy (Reinforcement Learning)"

    def get_class_name(self):
        """
        Returns a class name of the current model.
        :return: a class name of the current model.
        """
        return "LearnedGreedyRL"
