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
            prediction = d['model'].s_features(state, d['support'])
            output.extend(prediction)
        return np.array(output)

    def get_support_dim(self, index=None):
        # index is used to get the support dim of a specific network
        if index is None:
            return self.support_dim
        else:
            return self.stack[index]['support'].sum()

    def save(self):
        for idx, d in enumerate(self.stack):
            d['model'].save(filename='network_%d.h5' % idx)  # Save network
            np.save('support_%d.npy' % idx, d['support'])  # Save support array
