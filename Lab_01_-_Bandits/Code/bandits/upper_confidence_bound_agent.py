'''
Created on 14 Jan 2022

@author: ucacsjj
'''

import math

import numpy as np

from .agent import Agent


class UpperConfidenceBoundAgent(Agent):

    def __init__(self, environment, c):
        super().__init__(environment)
        self._c = c

    # Q6a:
    # Implement UCB
    def _choose_action(self):
        def cts():
            return np.array(
                [self._c * (math.log10(np.sum(self.number_of_pulls)) / bandit_pulls) ** 0.5 for bandit_pulls in
                 self.number_of_pulls])

        q = np.divide(self.total_reward, self.number_of_pulls) + cts()
        best_action = np.where(q == np.amax(q))[0]
        action = best_action[0]
        return action
