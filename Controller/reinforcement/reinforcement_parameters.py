class GreedyPolicyParameters():
    """
    Encapsulates parameters of for reinforcement learning.
    """

    @staticmethod
    def from_dict(data):
        params = GreedyPolicyParameters(
            data["batch_size"],
            data["episodes"],
            data["gamma"],
            data["optimizer"],
            data["epsilon"],
            data["learning_rate"])

        return params

    def __init__(self,
                 batch_size,
                 episodes,
                 gamma,
                 optimizer,
                 epsilon,
                 learning_rate=0.001):
        self.batch_size = batch_size
        self.episodes = episodes
        self.gamma = gamma
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.learning_rate = learning_rate

    def to_dictionary(self):
        data = {}
        data["batch_size"] = self.batch_size
        data["episodes"] = self.episodes
        data["gamma"] = self.gamma
        data["optimizer"] = self.optimizer
        data["epsilon"] = self.epsilon
        data["learning_rate"] = self.learning_rate
        return data

    def to_string(self):
        return "gamma: {}, optimizer: {}, learning rate: {}, epsilon: {}, batch_size: {}, episodes: {}".format(
            self.gamma, self.optimizer, self.learning_rate, self.epsilon, self.batch_size, self.episodes)


class DDPGParameters():
    """
    Encapsulates parameters of for reinforcement learning.
    """

    @staticmethod
    def from_dict(data):
        params = DDPGParameters(
            data["batch_size"],
            data["episodes"])
        return params

    def __init__(self,
                 batch_size,
                 episodes):
        self.batch_size = batch_size
        self.episodes = episodes

    def to_dictionary(self):
        data = {}
        data["batch_size"] = self.batch_size
        data["episodes"] = self.episodes
        return data

    def to_string(self):
        return "batch_size: {}, episodes: {}".format(self.batch_size, self.episodes)
