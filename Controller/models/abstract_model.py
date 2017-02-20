class AbstractModel():
    """
    Wrapper for all models for evaluating input from a game.
    """

    @staticmethod
    def load_from_file(file_name, game):
        raise NotImplementedError

    def get_new_instance(self, weights, game_config):
        raise NotImplementedError

    def evaluate(self, input, current_phase):
        raise NotImplementedError

    def get_number_of_parameters(self, game):
        raise NotImplementedError

    def get_name(self):
        raise NotImplementedError

    def get_class_name(self):
        raise NotImplementedError
