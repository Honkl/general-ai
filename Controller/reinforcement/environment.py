import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class Environment(gym.Env):
    def __init__(self, game_class, base_reward, penalty, seed, observations_count, actions_count):
        self.game_class = game_class
        self.state = None
        self.base_reward = base_reward
        self.penalty = penalty
        self.actions_count = actions_count

        self.observation_space = spaces.Discrete(n=observations_count)
        self.action_space = spaces.Discrete(n=actions_count)

        self._seed(seed)
        self.reset()
        self.viewer = None
        self._configure()

    def _step(self, action):
        """
        Performs a single step in the game.
        :param action: Action to make.
        :return: By Gym-interface, returns observation (new state), reward, done, info
        """
        new_state, reward, done = self.game_instance.step(action)
        self.state = new_state

        return self.state, reward, done, {}  # info

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, s = seeding.np_random(seed)
        return s

    def _reset(self):
        model = None
        game_batch = 1
        seed = np.random.randint(0, 2 ** 16)
        self.game_instance = self.game_class(model, game_batch, seed)
        self.state = self.game_instance.init_process()  # First state of the game
        return self.state

    def _render(self, mode='human', close=False):
        pass

    def _close(self):
        pass
