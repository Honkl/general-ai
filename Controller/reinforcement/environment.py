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
        self.last_phase = 0

        self.observation_space = spaces.Discrete(n=observations_count)
        self.action_space = spaces.Discrete(n=sum(actions_count))

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

        # Need to determine proper game phase and use only specific action subset
        if len(self.actions_count) > 0:
            begin = sum(self.actions_count[:self.last_phase])
            end = begin + self.actions_count[self.last_phase]
            action = action[begin:end]

        action_string = ""
        # for i in range(self.actions_count):
        #    if i == action:
        #        action_string += "1 "
        #    else:
        #        action_string += "0 "
        for a in action:
            action_string += str(a) + " "

        new_state, self.last_phase, reward, done = self.game_instance.step(action_string)
        self.state = new_state

        return self.state, reward, done, self.game_instance.score

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
        self.state, _ = self.game_instance.init_process()  # First state of the game
        return self.state

    def _render(self, mode='human', close=False):
        pass

    def _close(self):
        pass

    def shut_down(self):
        self.game_instance.finalize()
