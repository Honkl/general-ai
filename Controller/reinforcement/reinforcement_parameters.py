class ReinforcementParameters():
    """
    Encapsulates parameters of for reinforcement learning.
    """

    @staticmethod
    def from_dict(data):
        params = ReinforcementParameters(
            data["batch_size"],
            data["episodes"],
            data["gamma"],
            data["optimizer"],
            data["rand_action_prob"],
            data["learning_rate"])

        return params

    def __init__(self,
                 batch_size,
                 episodes,
                 gamma,
                 optimizer,
                 rand_action_prob,
                 learning_rate=0.001):
        self.batch_size = batch_size
        self.episodes = episodes
        self.gamma = gamma
        self.optimizer = optimizer
        self.rand_action_prob = rand_action_prob
        self.learning_rate = learning_rate

    def to_dictionary(self):
        data = {}
        data["batch_size"] = self.batch_size
        data["episodes"] = self.episodes
        data["gamma"] = self.gamma
        data["optimizer"] = self.optimizer
        data["rand_action_prob"] = self.rand_action_prob
        data["learning_rate"] = self.learning_rate
        return data

    def to_string(self):
        return "gamma: {}, optimizer: {}, learning rate: {}, rand action prob: {}, batch_size: {}, episodes: {}".format(
            self.gamma, self.optimizer, self.learning_rate, self.rand_action_prob, self.batch_size, self.episodes)
