import numpy as np


class NNStack:
    def __init__(self):
        self.stack = []
        self.support_dim = 0

    def add(self, model, support):
        d = {'model': model, 'support': np.array(support)}
        self.stack.append(d)
        self.support_dim += d['support'].sum()

    def s_features(self, state):
        # Runs all neural networks on the given state,
        # returns the selected features of each NN as a single array.
        output = []
        for d in self.stack:
            prediction = d['model'].flat_encode(state)[0]
            prediction = prediction[d['support']]  # Keep only support features
            output.extend(prediction)
        return np.array(output)

    def get_support_dim(self, index=None):
        # index is used to get the support dim of a specific network
        return self.support_dim if index is None else self.stack[index]['support'].sum()

    # TODO Save NNStack
    def save(self):
        pass