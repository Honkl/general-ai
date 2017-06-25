"""
The MIT License (MIT)

Copyright (c) 2016 Flood Sung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import gym

def makeFilteredEnv(env):
  """ crate a new environment class with actions and states normalized to [-1,1] """
  acsp = env.action_space
  obsp = env.observation_space
  if not type(acsp)==gym.spaces.box.Box:
    raise RuntimeError('Environment with continous action space (i.e. Box) required.')
  if not type(obsp)==gym.spaces.box.Box:
    raise RuntimeError('Environment with continous observation space (i.e. Box) required.')

  env_type = type(env)

  class FilteredEnv(env_type):
    def __init__(self):
      self.__dict__.update(env.__dict__) # transfer properties

      # Observation space
      if np.any(obsp.high < 1e10):
        h = obsp.high
        l = obsp.low
        sc = h-l
        self.o_c = (h+l)/2.
        self.o_sc = sc / 2.
      else:
        self.o_c = np.zeros_like(obsp.high)
        self.o_sc = np.ones_like(obsp.high)

      # Action space
      h = acsp.high
      l = acsp.low
      sc = (h-l)
      self.a_c = (h+l)/2.
      self.a_sc = sc / 2.

      # Rewards
      self.r_sc = 0.1
      self.r_c = 0.

      # Special cases
      '''
      if self.spec.id == "Reacher-v1":
        print "is Reacher!!!"
        self.o_sc[6] = 40.
        self.o_sc[7] = 20.
        self.r_sc = 200.
        self.r_c = 0.
    '''
      # Check and assign transformed spaces
      self.observation_space = gym.spaces.Box(self.filter_observation(obsp.low),
                                              self.filter_observation(obsp.high))
      self.action_space = gym.spaces.Box(-np.ones_like(acsp.high),np.ones_like(acsp.high))
      def assertEqual(a,b): assert np.all(a == b), "{} != {}".format(a,b)
      assertEqual(self.filter_action(self.action_space.low), acsp.low)
      assertEqual(self.filter_action(self.action_space.high), acsp.high)

    def filter_observation(self,obs):
      return (obs-self.o_c) / self.o_sc

    def filter_action(self,action):
      return self.a_sc*action+self.a_c

    def filter_reward(self,reward):
      ''' has to be applied manually otherwise it makes the reward_threshold invalid '''
      return self.r_sc*reward+self.r_c

    def step(self,action):

      ac_f = np.clip(self.filter_action(action),self.action_space.low,self.action_space.high)

      obs, reward, term, info = env_type.step(self,ac_f) # super function

      obs_f = self.filter_observation(obs)

      return obs_f, reward, term, info

  fenv = FilteredEnv()

  print('True action space: ' + str(acsp.low) + ', ' + str(acsp.high))
  print('True state space: ' + str(obsp.low) + ', ' + str(obsp.high))
  print('Filtered action space: ' + str(fenv.action_space.low) + ', ' + str(fenv.action_space.high))
  print('Filtered state space: ' + str(fenv.observation_space.low) + ', ' + str(fenv.observation_space.high))

  return fenv