from ifqi.algorithms.fqi import FQI
from random import random, choice


class EpsilonFQI:
    def __init__(self, fqi_params, nn_stack, epsilon=1.0, epsilon_rate=0.99,
                 min_epsilon=0.05, fqi=None):
        self.fqi_params = fqi_params
        self.nn_stack = nn_stack
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_rate = epsilon_rate
        if fqi is None:
            self.initial_actions = self.fqi_params['discrete_actions']
            self.fqi = FQI(**self.fqi_params)
        else:
            self.fqi = fqi
            self.initial_actions = self.fqi._actions

    def fit_on_dataset(self, sast, r, state_dim, **kwargs):
        self.fqi_params['state_dim'] = state_dim
        self.fqi = FQI(**self.fqi_params)
        self.fqi.fit(sast, r, **kwargs)

    def partial_fit_on_dataset(self, sast=None, r=None, **kwargs):
        self.fqi.partial_fit(sast, r, **kwargs)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def epsilon_step(self, epsilon_rate=None):
        if self.epsilon * self.epsilon_rate >= self.min_epsilon:
            if epsilon_rate is None:
                self.epsilon *= self.epsilon_rate
            else:
                self.epsilon *= epsilon_rate

    def draw_action(self, state, absorbing, evaluation=False):
        if not evaluation and random() <= self.epsilon:
            return choice(self.initial_actions)
        else:
            preprocessed_state = self.nn_stack.s_features(state)
            return self.fqi.draw_action(preprocessed_state, absorbing,
                                        evaluation=evaluation)
