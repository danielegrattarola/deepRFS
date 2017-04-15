from ifqi.algorithms.fqi import FQI
from deep_ifs.extraction.NNStack import NNStack
from random import random, choice
import joblib


class EpsilonFQI:
    def __init__(self, fqi_params, nn_stack, epsilon=1.0, epsilon_rate=0.99,
                 min_epsilon=0.05, fqi=None):
        self.fqi_params = fqi_params
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_rate = epsilon_rate

        self.nn_stack = None
        self.load_nn_stack(nn_stack)

        if fqi is None:
            self.initial_actions = self.fqi_params['discrete_actions']
            self.fqi = FQI(**self.fqi_params)
        else:
            self.fqi = None
            self.load_fqi(fqi)

    def fit(self, sast, r, state_dim, **kwargs):
        """
        Fits the policy from scratch on the given dataset for a number of steps
         as defined in self.fqi_params.

        Args
            sast (np.array): the dataset on which to fit
            r (np.array): the reward of each sast transition in the dataset
            state_dim (int): the dimensionality of the state space
            **kwargs: parameters to pass to the fit function of FQI
        """
        self.reset(state_dim)
        self.fqi.fit(sast, r, **kwargs)

    def partial_fit(self, sast=None, r=None, **kwargs):
        """
        Fits the policy from scratch on the given dataset for one step.
        Assumes that the dimensionality of the state is already correct (you
        can set it with the set_state_dim method).
        The parameters sast and r must only be passed at the first iteration
        and can be omitted on subsequent calls.

        Args
            sast (np.array, None): the dataset on which to fit
            r (np.array, None): the reward of each sast transition in the dataset
            **kwargs: parameters to pass to the fit function of FQI
        """
        self.fqi.partial_fit(sast, r, **kwargs)

    def draw_action(self, state, absorbing, evaluation=False):
        """
        Draws an action from the action space using the epsilon-greedy policy.

        Args
            state (np.array): the current state of the MDP
            absorbing (bool): whether the state is an absorbing state
            evaluation (bool, False): if True, the policy is set to fully
                greedy
        """
        if not evaluation and random() <= self.epsilon:
            return choice(self.initial_actions)
        else:
            preprocessed_state = self.nn_stack.s_features(state)
            return self.fqi.draw_action(preprocessed_state, absorbing,
                                        evaluation=evaluation)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def epsilon_step(self, epsilon_rate=None):
        """
        Decreases epsilon by multiplying it for self.epsilon_rate if no value is
        passed as parameter (otherwise, the parameter is used as factor).
        """
        if self.epsilon * self.epsilon_rate >= self.min_epsilon:
            if epsilon_rate is None:
                self.epsilon *= self.epsilon_rate
            else:
                self.epsilon *= epsilon_rate

    def reset(self, state_dim):
        """
        Sets the dimension of the states to the given parameter and resets the
        policy.
        """
        self.fqi_params['state_dim'] = state_dim
        self.fqi = FQI(**self.fqi_params)

    def load_fqi(self, fqi):
        """
        Loads an FQI object from memory or file.

        Args:
            fqi (FQI or str): if an FQI instance is passed, then it is assigned
                directly to self.fqi. Otherwise, if a string is passed, the
                corresponding pickle file is loaded from disk.
        """
        if isinstance(fqi, str):
            self.fqi = joblib.load(fqi)
        else:
            self.fqi = fqi
        # Set the correct action space
        self.initial_actions = self.fqi._actions

    def save_fqi(self, filepath):
        """
        Dumps the fqi instance to file at filepath.
        """
        joblib.dump(self.fqi, filepath)

    def load_nn_stack(self, nn_stack):
        """
        Loads an NNStack object from memory or file.

        Args:
            nn_stack (NNStack or str): if an NNStack instance is passed, then
                it is assigned directly to self.nn_stack. Otherwise, if a
                string is passed, the model is loaded from the corresponding
                folder.
        """
        if isinstance(nn_stack, str):
            self.nn_stack = NNStack()
            self.nn_stack.load(nn_stack)
        else:
            self.nn_stack = nn_stack

    def save_nn_stack(self, folder):
        """
        Dumps the nn_stack instance into folder.
        """
        self.nn_stack.save(folder)