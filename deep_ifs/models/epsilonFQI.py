from ifqi.algorithms.fqi import FQI
from deep_ifs.extraction.NNStack import NNStack
from random import random, choice
import joblib


class EpsilonFQI:
    def __init__(self, fqi, nn_stack, epsilon=0.05):
        self.epsilon = epsilon

        self.fqi = None
        self.actions = None

        self.fe = None
        self.load_fe(nn_stack)

        if isinstance(fqi, dict):
            self.fqi = FQI(**fqi)
            self.actions = fqi['discrete_actions']
        else:
            self.load_fqi(fqi)

    def fit(self, sast, r, **kwargs):
        self.fqi.fit(sast, r, **kwargs)

    def partial_fit(self, sast=None, r=None, **kwargs):
        self.fqi.partial_fit(sast, r, **kwargs)

    def draw_action(self, state, absorbing, evaluation=False, fully_deterministic=False):
        if not fully_deterministic and random() <= self.epsilon:
            return choice(self.actions)
        else:
            preprocessed_state = self.fe.s_features(state)
            return self.fqi.draw_action(preprocessed_state, absorbing,
                                        evaluation=evaluation)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_epsilon(self):
        return self.epsilon

    def load_fqi(self, fqi):
        if isinstance(fqi, str):
            self.fqi = joblib.load(fqi)
        else:
            self.fqi = fqi
        # Set the correct action space
        self.actions = self.fqi._actions

    def save_fqi(self, filepath):
        joblib.dump(self.fqi, filepath)

    def load_fe(self, fe):
        if isinstance(fe, str):
            self.fe = joblib.load(fe)
            self.fe.load(fe)
        else:
            self.fe = fe

    def save_fe(self, filepath):
        if hasattr(self.fe, 'save'):
            self.fe.save(filepath)
        else:
            joblib.dump(self.fe, filepath)