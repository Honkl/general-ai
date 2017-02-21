import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt


class Environment(gym.Env):
    def __init__(self, game_class, seed, observations_count, actions_in_phases, discrete):
        self.game_class = game_class
        self.game_instance = None
        self.state = None
        self.actions_in_phases = actions_in_phases
        self.last_phase = 0
        self.done = False
        self.discrete = discrete

        self.actions_total = sum(actions_in_phases)
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(observations_count,))
        if discrete:
            self.action_space = spaces.Discrete(n=self.actions_total)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.actions_total,))

        self._seed(seed)
        self.reset()
        self.viewer = None
        self._configure()

        self.results = []

    def _step(self, action):
        """
        Performs a single step in the game.
        :param action: Action to make.
        :return: By Gym-interface, returns observation (new state), reward, done, info
        """

        # Need to determine proper game phase and use only specific action subset
        actions_count = self.actions_total
        if len(self.actions_in_phases) > 1:
            begin = sum(self.actions_in_phases[:self.last_phase])
            end = begin + self.actions_in_phases[self.last_phase]
            action = action[begin:end]
            actions_count = end - begin

        new_state, self.last_phase, reward, done = self.game_instance.step(action)
        self.state = new_state
        self.done = done
        if done:
            self.results.append(self.game_instance.score)
            #self.save_results()
            return self.state, reward, done, int(self.game_instance.score)
            #return self.state, reward, done,
        else:
            return self.state, reward, done, {}

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
        seed = np.random.randint(0, 2 ** 16)
        self.game_instance = self.game_class(model, game_batch, seed)
        self.done = False
        self.last_phase = 0
        self.state, _ = self.game_instance.init_process()  # First state of the game
        return self.state

    def _render(self, mode='human', close=False):
        pass

    def _close(self):
        self.shut_down()

    def shut_down(self):
        if self.game_instance:
            self.game_instance.finalize()

    def save_results(self):
        plt.figure()
        plt.plot(self.results)
        print(self.results)
        plt.xlim(20)
        plt.xlabel("Episode")
        plt.ylim(5000)
        plt.ylabel("Score")
        plt.savefig("score_plot.jpg")
        plt.close()

