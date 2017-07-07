from random import random, choice
import numpy as np


class GenericPolicy:
    def __init__(self, fe, q_model, action_values, epsilon=0.05):
        self.fe = fe
        self.q_model = q_model
        self.epsilon = epsilon
        self.action_values = action_values

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
            return choice(self.action_values)
        else:
            preprocessed_state = self.fe.s_features(state).reshape(1, -1)
            Q = self.q_model.predict(preprocessed_state) * (1 - absorbing)

            if Q.shape[0] > 1:
                amax = np.argmax(Q, axis=1)
            else:
                q = Q[0]
                amax = np.array([np.random.choice(np.argwhere(q == np.max(q)).ravel())]).ravel()

            return amax

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

