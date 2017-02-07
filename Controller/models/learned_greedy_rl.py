import json
import os

from models.model import Model
from reinforcement.greedy_policy.greedy_policy_reinforcement import GreedyPolicyReinforcement
from reinforcement.greedy_policy.q_network import QNetwork
from reinforcement.reinforcement_parameters import ReinforcementParameters


class LearnedGreedyRL(Model):
    def __init__(self, logdir):
        self.metadata = None
        for file in os.listdir(logdir):
            if file.endswith(".json"):
                with open(os.path.join(logdir, file), "r") as f:
                    self.metadata = json.load(f)
                    break

        q_net_params = self.metadata["q-network"]
        rl_params = self.metadata["reinforcement_params"]
        self.game = self.metadata["game"]
        self.q_network = QNetwork(q_net_params["hidden_layers"], q_net_params["activation"])

        self.rl = GreedyPolicyReinforcement(self.game, ReinforcementParameters.from_dict(rl_params), self.q_network, threads=1)
        self.rl.load_checkpoint(logdir)

    def get_new_instance(self, weights, game_config):
        raise NotImplementedError

    def evaluate(self, input, current_phase):
        action, estimated_reward = self.rl.agent.play(input, None)
        action_string = ""
        for a in action:
            action_string += str(a) + " "
        return action_string

    def get_name(self):
        return "Learned Greedy Policy (Reinforcement Learning)"

    def get_class_name(self):
        return "LearnedGreedyRL"
