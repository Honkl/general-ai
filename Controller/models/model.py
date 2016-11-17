class Model():
    """
    Wrapper for all models for evaluating input from a game.
    """
    def __init__(self, game_config, model_config):
        self.game_config = game_config
        self.model_config = model_config

    def evaluate(self, input):
        pass
