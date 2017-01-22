import os
import json

from models.model import Model
from reinforcement.reinforcement import Reinforcement
from reinforcement.reinforcement_parameters import ReinforcementParameters
from reinforcement.q_network import QNetwork

class LearnedQNet(Model):
    def __init__(self, logdir):
        self.metadata = None
        for file in os.listdir(logdir):
            if file.endswith(".json"):
                with open(os.path.join(logdir, file), "r") as f:
                    self.metadata = json.load(f)
                    break

        q_net_params = self.metadata["q_network"]
        rl_params = self.metadata["reinforce_params"]
        self.game = self.metadata["game"]
        self.q_network = QNetwork(q_net_params["hidden_layers"], q_net_params["activations"])

        self.rl = Reinforcement(self.game, ReinforcementParameters.from_dict(rl_params), self.q_network, threads=1)
        self.rl.load_checkpoint(logdir)

    def get_new_instance(self, weights, game_config):
        raise NotImplementedError

    def evaluate(self, input, current_phase):
        action, estimated_reward = self.rl.agent.play(input)
        action_string = ""
        for i in range(self.rl.actions_count):
            if i == action:
                action_string += "1 "
            else:
                action_string += "0 "
        return action_string

    def get_name(self):
        return "RL Q-net"

    def get_class_name(self):
        return "LearnedQNet"
