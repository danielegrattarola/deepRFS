import numpy as np
import gc
import glob
from deep_ifs.extraction.GenericEncoder import GenericEncoder
from keras import backend as K


class NNStack:
    def __init__(self):
        self.stack = []
        self.support_dim = 0

    def add(self, model, support):
        """
        Add a ConvNet and its support to the stack.

        Args
            model (ConvNet): a ConvNet object to store
            support (np.array): boolean mask for the model s_features and
                all_features methods
        """
        d = {'model': model, 'support': np.array(support)}
        self.stack.append(d)
        self.support_dim += d['support'].sum()

    def s_features(self, x):
        """
        Runs all neural networks on the given state, returns the selected
        features of each NN as a single array.

        Args
            state (np.array): the current state of the MDP.
        """
        if x.shape[0] == 1:
            output = []
        else:
            output = np.empty((x.shape[0], 0))

        for d in self.stack:
            prediction = d['model'].s_features(x, d['support'])
            if prediction.ndim == 1:
                output = np.concatenate([output, prediction])
            else:
                output = np.column_stack((output, prediction))
        return np.array(output)

    def get_support_dim(self, index=None):
        """
        Returns the cumulative dimension of all supports in the stack, or the
        dimension of the index-th support if index is given.
        """
        if index is None:
            return sum([d['support'].sum() for d in self.stack])
        else:
            return self.stack[index]['support'].sum()

    def reset(self):
        """
        Empties the stack and forcibly frees memory.
        """
        self.stack = []
        self.support_dim = 0
        K.clear_session()
        gc.collect()

    def save(self, folder):
        """
        Saves the encoders of all models in the stack and their supports
        in folder, as .h5 and .npy files respectively.
        """
        if not folder.endswith('/'):
            folder += '/'
        for idx, d in enumerate(self.stack):
            d['model'].save_encoder(folder + 'encoder_%d.h5' % idx)  # Save network
            np.save(folder + 'support_%d.npy' % idx, d['support'])  # Save support array

    def load(self, folder):
        """
        Loads all models (as .h5 files) and their supports (as .npy files) from
        folder.
        Note that the loaded models are instantiated as GenericEncoder models
        and are not trainable.
        """
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
