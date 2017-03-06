import numpy as np
import gym
from gym.utils import seeding
import matplotlib.pyplot as plt


class Environment(gym.Env):
    """
    Environment for reinforcement learning algorithms. This single environment is used for all games.
    """

    def __init__(self, game_class, seed, observations_count, actions_in_phases):
        """
        Initializes a new instance of Environment.
        :param game_class: A game class, implementing games.abstract_game.
        :param seed: Seed for the environment.
        :param observations_count: Num of observations.
        :param actions_in_phases: List of actions for game phases.
        """
        self.game_class = game_class
        self.game_instance = None
        self.state = None
        self.actions_in_phases = actions_in_phases
        self.last_phase = 0
        self.done = False

        self.actions_total = sum(actions_in_phases)
        self._seed(seed)
        self.reset()
        self.viewer = None
        self._configure()

    def _step(self, action):
        """
        Performs a single step in the game.
        :param action: Action to make. This is a list of actions (every game defines a number of actions they need
        in every step).
        :return: By Gym-interface, returns observation (new state), reward, done, info
        """

        # Need to determine proper game phase and use only specific action subset
        actions_count = self.actions_total
        if len(self.actions_in_phases) > 1:
            # Games with multiple phases
            begin = sum(self.actions_in_phases[:self.last_phase])
            end = begin + self.actions_in_phases[self.last_phase]
            action = action[begin:end]
            actions_count = end - begin

        new_state, self.last_phase, reward, done = self.game_instance.step(action)
        self.state = np.array(new_state)
        self.done = done
        return self.state, reward, done, float(self.game_instance.score)

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, s = seeding.np_random(seed)
        return s

    def _reset(self):
        if self.game_instance != None:
            self.game_instance.finalize()
        model = None
        game_batch = 1
        seed = np.random.randint(0, 2 ** 30)
        self.game_instance = self.game_class(model, game_batch, seed)
        self.done = False
        self.last_phase = 0
        self.state, _ = self.game_instance.init_process()  # First state of the game
        self.state = np.array(self.state)
        return self.state

    def _render(self, mode='human', close=False):
        pass

    def _close(self):
        self.shut_down()

    def shut_down(self):
        if self.game_instance:
            self.game_instance.finalize()
