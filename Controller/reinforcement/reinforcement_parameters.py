class ReinforcementParameters():
    """
    Encapsulates parameters of for reinforcement learning.
    """

    @staticmethod
    def from_dict(data):
        params = ReinforcementParameters(
            data["batch_size"],
            data["epochs"],
            data["gamma"],
            data["dropout"],
            data["optimizer"],
            data["learning_rate"])
        return params

    def __init__(self,
                 batch_size,
                 epochs,
                 gamma,
                 dropout,
                 optimizer,
                 learning_rate=0.001):
        self.batch_size = batch_size
        self.epochs = epochs
        self.gamma = gamma
        self.dropout = dropout
        self.optimizer = optimizer
        self.learning_rate = learning_rate

    def to_dictionary(self):
        data = {}
        data["batch_size"] = self.batch_size
        data["epochs"] = self.epochs
        data["gamma"] = self.gamma
        data["dropout"] = self.dropout
        data["optimizer"] = self.optimizer
        data["learning_rate"] = self.learning_rate
        return data

    def to_string(self):
        return "gamma: {}, dropout: {}, optimizer: {}, learning rate: {}, batch_size: {}, epochs: {}".format(
            self.gamma, self.dropout, self.optimizer, self.learning_rate, self.batch_size, self.epochs)
