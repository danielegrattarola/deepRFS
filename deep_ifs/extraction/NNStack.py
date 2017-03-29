import numpy as np
import gc
import glob
from deep_ifs.extraction.GenericEncoder import GenericEncoder


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
            return sum([d['support'].sum() for d in self.stack])
        else:
            return self.stack[index]['support'].sum()

    def reset(self):
        self.stack = []
        self.support_dim = 0
        gc.collect()

    def save(self, folder):
        if not folder.endswith('/'):
            folder += '/'
        for idx, d in enumerate(self.stack):
            d['model'].save_encoder(folder + 'encoder_%d.h5' % idx)  # Save network
            np.save(folder + 'support_%d.npy' % idx, d['support'])  # Save support array

    def load(self, folder):
        # Get all filepaths
        models = glob.glob(folder + 'encoder_*.h5')
        supports = glob.glob(folder + 'support_*.npy')
        nb_models = len(models)
        nb_supports = len(supports)
        assert nb_models == nb_supports and nb_models != 0

        self.reset()

        # Build all models (and their supports) for the stack
        for i in range(nb_models):
            m = GenericEncoder(folder + 'encoder_%s.h5' % i)
            s = np.load(folder + 'support_%s.npy' % i)
            self.stack.append({'model': m, 'support': s})

        self.support_dim = self.get_support_dim()
