import numpy as np
from keras.models import load_model
from keras.optimizers import Adam


class GenericEncoder:
    def __init__(self, path, binarize=False):
        self.encoder = load_model(path)

        self.support = None
        self.binarize = binarize

        # Optimization algorithm
        self.optimizer = Adam()
        print 'Compiling...'
        self.encoder.compile(optimizer=self.optimizer, loss='mse',
                             metrics=['accuracy'])

    def all_features(self, x):
        """ Embeds the given array using the encoder
        :param x: samples to encode, ignoring the support 
        :return: the encoded samples
        """
        # Feed input to the model, return encoded images flattened
        x = np.asarray(x).astype('float32') / 255  # To 0-1 range
        if self.binarize:
            x[x < 0.1] = 0
            x[x >= 0.1] = 1

        if x.shape[0] == 1:
            # x is a singe sample
            return np.asarray(self.encoder.predict_on_batch(x)).flatten()
        else:
            return np.asarray(self.encoder.predict(x))

    def s_features(self, x, support=None):
        """
        Runs the given samples on the model and returns the features of the last
        dense layer filtered by the support mask.
        :param x: samples to encode
        :param support: boolean mask with which to filter the embedding
        :return: th encoded samples
        """
        if support is None:
            support = self.support

        prediction = self.all_features(x)
        if x.shape[0] == 1:
            # x is a singe sample
            prediction = prediction[support]  # Keep only support features
        else:
            prediction = prediction[:, support]  # Keep only support features
        return prediction

    def save_encoder(self, filename):
        """
        Save the encoder model
        :param filename: filename to which save the model
        :return: 
        """
        self.encoder.save(filename)

    def set_support(self, support):
        """
        :param support: np.array, boolean mask to use as support 
        """
        self.support = support
