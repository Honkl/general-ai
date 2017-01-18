class ReinforcementParameters():
    """
    Encapsulates parameters of for reinforcement learning.
    """

    @staticmethod
    def from_dict(data):
        """
        params = ReinforcementParameters(
            data["pop_size"],
            data["cxpb"],
            data["mut"],
            data["ngen"],
            data["game_batch_size"],
            data["cxindpb"],
            data["hof_size"],
            data["elite"],
            data["selection"])
        return params
        """
        raise NotImplementedError("TODO")

    def __init__(self,
                 batch_size,
                 epochs,
                 penalty,
                 gamma,
                 base_reward,
                 dropout,
                 optimizer,
                 learning_rate=0.001):
        self.batch_size = batch_size
        self.epochs = epochs
        self.penalty = penalty
        self.gamma = gamma
        self.base_reward = base_reward
        self.dropout = dropout
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def to_dictionary(self):
        data = {}
        data["batch_size"] = self.batch_size
        data["epochs"] = self.epochs
        data["penalty"] = self.penalty
        data["gamma"] = self.gamma
        data["base_reward"] = self.base_reward
        data["dropout"] = self.dropout
        data["optimizer"] = self.optimizer
        data["learning_rate"] = self.learning_rate
        return data

    def to_string(self):
        return "penalty: {}, gamma: {}, base_reward: {}, dropout: {}, optimizer: {}, lr: {}, batch_size: {}, epochs: {}".format(
            self.penalty, self.gamma, self.base_reward, self.dropout, self.optimizer, self.learning_rate,
            self.base_reward, self.epochs)
