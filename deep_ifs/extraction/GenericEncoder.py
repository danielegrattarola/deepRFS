import numpy as np
from keras.models import load_model
from keras.optimizers import Adam

class GenericEncoder:
    def __init__(self, path):
        self.model = load_model(path)
        # Optimization algorithm
        self.optimizer = Adam()
        self.model.compile(optimizer=self.optimizer, loss='mse',
                           metrics=['accuracy'])

    def all_features(self, sample):
        """
        Runs the given sample on the model and returns the features of the last
        dense layer in a 1d array.
        :param sample: a single sample to encode.
        :return: the encoded sample.
        """
        # Feed input to the model, return encoded images flattened
        sample = np.asarray(sample).astype('float32') / 255  # To 0-1 range
        return np.asarray(self.model.predict_on_batch(sample)).flatten()

    def s_features(self, sample, support):
        """
        Runs the given sample on the model and returns the features of the last
        dense layer filtered by the support mask.
        :param sample: a single sample to encode.
        :param support: a boolean mask with which to filter the output.
        :return: the encoded sample.
        """
        prediction = self.all_features(sample)
        prediction = prediction[support]  # Keep only support features
        return prediction

    def save_encoder(self, filepath):
        self.model.save(filepath)
