from random import random, choice

import joblib
from ifqi.algorithms.fqi import FQI


class EpsilonFQI:
    def __init__(self, fqi, fe, epsilon=0.05):
        """
        Creates an epsilon-greedy policy from the given FQI policy object. 
        :param fqi: an FQI instance from the ifqi package
        :param fe: a feature extractor (method s_features(x) is expected)
        :param epsilon: exploration rate for the policy (0 <= epsilon <= 1)
        """
        self.epsilon = epsilon
        self.fqi = None
        self.actions = None
        self.fe = None

        self.load_fe(fe)

        if isinstance(fqi, dict):
            self.fqi = FQI(**fqi)
            self.actions = fqi['discrete_actions']
        else:
            self.load_fqi(fqi)

    def fit(self, sast, r, **kwargs):
        """
        Fits the policy on the given sast, r dataset, for the amounts of iterations
        defined when creating FQI.
        :param sast: a dataset of (state, action, state, terminal) transitions
        :param r: the rewards corresponding to the transitions in sast
        :param kwargs: additional arguments for the fit() function of FQI
        """
        self.fqi.fit(sast, r, **kwargs)

    def partial_fit(self, sast=None, r=None, **kwargs):
        """
        Fits the policy on the given sast, r dataset, for one iteration.
        :param sast: a dataset of (state, action, state, terminal) transitions
        :param r: the rewards corresponding to the transitions in sast
        :param kwargs: additional arguments for the partial_fit() function of FQI
        """
        self.fqi.partial_fit(sast, r, **kwargs)

    def draw_action(self, state, absorbing, evaluation=False, fully_deterministic=False):
        """
        Picks an action according to the epsilon-greedy choice
        :param state: a state
        :param absorbing: bool, whether the state is absorbing
        :param evaluation: bool, whether to use the epsilon defined for evaluation
        :param fully_deterministic: whether to use FQI, deterministically, to
        select the action
        :return: the selected action
        """
        if not fully_deterministic and random() <= self.epsilon:
            return choice(self.actions)
        else:
            preprocessed_state = self.fe.s_features(state)
            return self.fqi.draw_action(preprocessed_state, absorbing,
                                        evaluation=evaluation)

    def set_epsilon(self, epsilon):
        """
        :param epsilon: the exploration rate to use 
        """
        self.epsilon = epsilon

    def get_epsilon(self):
        """
        :return: the current exploration rate 
        """
        return self.epsilon

    def load_fqi(self, fqi):
        """
        Loads an FQI policy from file, sets the action space accordingly
        :param fqi: str or file-like object from which to load the policy
        """
        if isinstance(fqi, str):
            self.fqi = joblib.load(fqi)
        else:
            self.fqi = fqi
        # Set the correct action space
        self.actions = self.fqi._actions

    def save_fqi(self, filename):
        """
        Saves the FQI object to file
        :param filename: filename to which save the model
        """
        joblib.dump(self.fqi, filename)

    def load_fe(self, fe):
        """
        Loads the feature extractor from file
        :param fe: str or file-like object from which to load the model
        """
        if isinstance(fe, str):
            self.fe = joblib.load(fe)
            self.fe.load(fe)
        else:
            self.fe = fe

    def save_fe(self, filename):
        """
        Saves the feature extractor to file
        :param filename: filename to which save the model
        """
        if hasattr(self.fe, 'save'):
            self.fe.save(filename)
        else:
            joblib.dump(self.fe, filename)