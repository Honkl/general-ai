class Model():
    """
    Wrapper for all models for evaluating input from a game.
    """

    def get_new_instance(self, weights, game_config):
        raise NotImplementedError("Calling interface.")

    def evaluate(self, input):
        raise NotImplementedError("Calling interface.")

    def get_number_of_parameters(self, game):
        raise NotImplementedError("Calling interface.")

    def get_name(self):
        raise NotImplementedError("Calling interface.")

    def get_class_name(self):
        raise NotImplementedError("Calling interface.")
