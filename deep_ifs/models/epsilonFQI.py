from ifqi.algorithms.fqi import FQI
from random import random, choice


class EpsilonFQI:
    def __init__(self, fqi_params, epsilon=1.0, epsilon_rate=0.99):
        self.epsilon = epsilon
        self.epsilon_rate = epsilon_rate
        self.fqi_params = fqi_params
        self.initial_actions = self.fqi_params['discrete_actions']
        self.fqi = FQI(**self.fqi_params)

    def fit_on_dataset(self, sast, r, state_dim):
        self.fqi_params['state_dim'] = state_dim
        self.fqi = FQI(**self.fqi_params)
        self.fqi.fit(sast, r)

    def epsilon_step(self, epsilon_rate=None):
        if epsilon_rate is None:
            self.epsilon *= self.epsilon_rate
        else:
            self.epsilon *= epsilon_rate

    def draw_action(self, state, absorbing, evaluation=False):
        if not evaluation and random() <= self.epsilon:
            return choice(self.initial_actions)
        else:
            return self.fqi.draw_action(state, absorbing, evaluation=evaluation)
