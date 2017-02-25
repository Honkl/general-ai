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
            data["test_size"],
            data["learning_rate"])

        return params

    def __init__(self,
                 batch_size,
                 episodes,
                 gamma,
                 optimizer,
                 epsilon,
                 test_size,
                 learning_rate=0.001):
        self.batch_size = batch_size
        self.episodes = episodes
        self.gamma = gamma
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.test_size = test_size
        self.learning_rate = learning_rate

    def to_dictionary(self):
        data = {}
        data["batch_size"] = self.batch_size
        data["episodes"] = self.episodes
        data["gamma"] = self.gamma
        data["optimizer"] = self.optimizer
        data["epsilon"] = self.epsilon
        data["test_size"] = self.test_size
        data["learning_rate"] = self.learning_rate
        return data

    def to_string(self):
        return "gamma: {}, optimizer: {}, learning rate: {}, epsilon: {}, batch_size: {}, episodes: {}, test_size: {}".format(
            self.gamma, self.optimizer, self.learning_rate, self.epsilon, self.batch_size, self.episodes,
            self.test_size)


class DDPGParameters():
    """
    Encapsulates parameters of for reinforcement learning.
    """

    @staticmethod
    def from_dict(data):
        params = DDPGParameters(
            data["batch_size"],
            data["episodes"],
            data["test_size"])
        return params

    def __init__(self,
                 batch_size,
                 episodes,
                 test_size):
        self.batch_size = batch_size
        self.episodes = episodes
        self.test_size = test_size

    def to_dictionary(self):
        data = {}
        data["batch_size"] = self.batch_size
        data["episodes"] = self.episodes
        data["test_size"] = self.test_size
        return data

    def to_string(self):
        return "batch_size: {}, episodes: {}, test_size: {}".format(self.batch_size, self.episodes, self.test_size)


class DQNParameters():
    """
    Encapsulates parameters fo DQN reinforcement learning.
    """

    @staticmethod
    def from_dict(data):
        raise NotImplementedError("Still not implemented")

    def __init__(self,
                 batch_size,
                 init_exp=0.5,  # initial exploration prob
                 final_exp=0.1,  # final exploration prob
                 anneal_steps=10000,  # N steps for annealing exploration
                 replay_buffer_size=10000,
                 store_replay_every=5,  # how frequent to store experience
                 discount_factor=0.9,  # discount future rewards
                 target_update_rate=0.01,
                 reg_param=0.01,  # regularization constants
                 max_gradient=5,  # max gradient norms
                 double_q_learning=False):
        self.batch_size = batch_size
        self.init_exp = init_exp
        self.final_exp = final_exp
        self.anneal_steps = anneal_steps
        self.replay_buffer_size = replay_buffer_size
        self.store_replay_every = store_replay_every
        self.discount_factor = discount_factor
        self.target_update_rate = target_update_rate
        self.reg_param = reg_param
        self.max_gradient = max_gradient
        self.double_q_learning = double_q_learning

    def to_dictionary(self):
        data = {}
        data["batch_size"] = self.batch_size
        data["init_exp"] = self.init_exp
        data["final_exp"] = self.final_exp
        data["anneal_steps"] = self.anneal_steps
        data["replay_buffer_size"] = self.replay_buffer_size
        data["store_replay_every"] = self.store_replay_every
        data["discount_factor"] = self.discount_factor
        data["target_update_rate"] = self.target_update_rate
        data["reg_param"] = self.reg_param
        data["max_gradient"] = self.max_gradient
        data["double_q_learning"] = self.double_q_learning
        return data

    def to_string(self):
        return "batch_size: {}, init_exp: {}, final_exp: {}, final_exp: {}, replay_buffer_size: {}, " \
               "store_replay_every: {}. discount_factor: {}, target_update_rate: {}, reg_param: {}, " \
               "max_gradient: {}, double_q_learning: {}".format(self.batch_size, self.init_exp, self.final_exp,
                                                                self.final_exp, self.replay_buffer_size,
                                                                self.store_replay_every, self.discount_factor,
                                                                self.target_update_rate, self.reg_param,
                                                                self.max_gradient, self.double_q_learning)
