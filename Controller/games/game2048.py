from games.game import Game
from constants import *
import importlib.util
import numpy as np


class Game2048(Game):
    """
    Represents a single 2048 game.
    """

    def __init__(self, model, game_batch_size, seed):
        """
        Initializes a new instance of 2048 game.
        :param model: Model which will be playing this game.
        :param game_batch_size: Number of games that will be played immediately (one after one) within the single game
        instance. Result is averaged.
        :param seed: A random seed for random generator within the game.
        :param use_advanced_tool: True if some advanced results are wanted (using different process).
        """
        super(Game2048, self).__init__()
        self.model = model
        self.game_batch_size = game_batch_size
        self.seed = seed
        self.phase = 0

    def init_process(self):
        """
        Initializes a new 2048 game.
        """
        spec = importlib.util.spec_from_file_location("Game", GAME2048_PY_PATH)
        game_2048 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(game_2048)
        self.game = game_2048.Game()
        state = self.game.get_state()
        return state, self.phase

    def run(self, advanced_results=False):
        """
        Runs a whole game and returns result.
        :return: Game result.
        """
        score_total = 0
        for _ in range(self.game_batch_size):
            state, phase = self.init_process()
            while not self.game.end:
                result = self.model.evaluate(state, phase)
                result = np.argsort(np.array(result))[::-1]
                for a in result:
                    moved, _ = self.game.move(a)
                    if moved:
                        break

                state = self.game.get_state()
            score_total += self.game.score
        return score_total / self.game_batch_size

    def step(self, action):
        """
        Performs a single step within the game.
        :param action: Action to make.
        :return: New state, current phase, reward, done
        """
        reward = None
        result = np.argsort(np.array(action))[::-1]
        assert(len(result) == 4) # TODO
        for a in result:
            moved, reward = self.game.move(a)
            if moved:
                break
        new_state = self.game.get_state()
        self.score = self.game.score

        if self.game.end:
            return new_state, self.phase, reward, True

        return new_state, self.phase, reward, False

    def finalize(self):
        pass
