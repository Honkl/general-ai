import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class Environment(gym.Env):
    def __init__(self, base_reward, penalty, seed, observations_count, actions_count):
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
        raise NotImplementedError

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, s = seeding.np_random(seed)
        return s

    def _reset(self):
        pass

    def _render(self, mode='human', close=False):
        pass

    def _close(self):
        pass
