#!/usr/bin/env python3

'''
Created on 13 Jan 2022

@author: ucacsjj
'''
import numpy as np
import gym

from bandits.bandit import Bandit
from bandits.bandit import BanditEnvironment


# Q2b:
# Write a method which takes only takes the bandit environment
# as input, runs all the bandits, and computes the mean and
# covariance. Print out the result for each bandit using the
# form specified below

def run_bandits(environment, number_of_steps):
    for bandit_number in range(environment.number_of_bandits()):
        b = bandit_number
        rewards = np.zeros(number_of_steps)
        for i in range(number_of_steps):
            rewards[i] = environment.bandit(bandit_number=b).pull_arm()
        print(f'bandit = {b}, mean = {np.mean(rewards)}, sigma = {np.std(rewards)}')


if __name__ == '__main__':
    # Q2a:
    # Change to implement four bandits with the mean and
    # covariance specified in the question
    environment = BanditEnvironment(4)

    # Add some bandits
    environment.set_bandit(0, Bandit(mean=1, sigma=1))
    environment.set_bandit(1, Bandit(mean=1, sigma=2))
    environment.set_bandit(2, Bandit(mean=2, sigma=1))
    environment.set_bandit(3, Bandit(mean=2, sigma=2))

    # Q2b:
    # Vary the number of steps if you like to validate your code
    run_bandits(environment, 1000)
