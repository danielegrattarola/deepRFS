import numpy as np
from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from sklearn import preprocessing

from deep_ifs.extraction.GatherLayer import GatherLayer


class ConvNet:
    def __init__(self, input_shape, target_size, nb_actions=1, encoding_dim=512,
                 nb_epochs=10, dropout_prob=0.5, binarize=False,
                 class_weight=None, sample_weight=None, load_path=None,
                 logger=None):
        self.dim_ordering = 'th'  # (samples, filters, rows, cols)
        self.input_shape = input_shape
        self.target_size = target_size
        self.nb_actions = nb_actions
        self.encoding_dim = encoding_dim
        self.nb_epochs = nb_epochs
        self.dropout_prob = dropout_prob
        self.binarize = binarize
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.logger = logger

        # Build network
        self.input = Input(shape=self.input_shape)
        self.u = Input(shape=(1,), dtype='int32')

        self.hidden = Convolution2D(32, 8, 8, border_mode='valid',
                                    activation='relu', subsample=(4, 4),
                                    dim_ordering='th')(self.input)
        self.hidden = BatchNormalization()(self.hidden)
        self.hidden = Convolution2D(64, 4, 4, border_mode='valid',
                                    activation='relu', subsample=(2, 2),
                                    dim_ordering='th')(self.hidden)
        self.hidden = BatchNormalization()(self.hidden)
        self.hidden = Convolution2D(64, 3, 3, border_mode='valid',
                                    activation='relu', subsample=(1, 1),
                                    dim_ordering='th')(self.hidden)
        self.hidden = BatchNormalization()(self.hidden)

        self.hidden = Flatten()(self.hidden)
        self.features = Dense(self.encoding_dim, activation='relu')(self.hidden)
        self.hidden = BatchNormalization()(self.features)
        self.output = Dense(self.target_size * self.nb_actions, activation='linear')(self.features)
        self.output_u = GatherLayer(self.target_size, self.nb_actions)([self.output, self.u])

        # Models
        self.model = Model(input=[self.input, self.u], output=self.output_u)
        self.encoder = Model(input=self.input, output=self.features)

        # Optimization algorithm
        try:
            '''
            self.optimizer = RMSpropGraves(lr=0.00025,
                                           momentum=0.95,
                                           squared_momentum=0.95,
                                           epsilon=0.01)
            '''
            self.optimizer = Adam()
        except NameError:
            self.optimizer = RMSprop()

        # Load the network from saved model
        if load_path is not None:
            self.load(load_path)

        self.model.compile(optimizer=self.optimizer, loss='mse',
                           metrics=['accuracy'])

    def fit(self, x, u, y):
        """
        Trains the model on a set of batches.
        :param x: f samples on which to train.
        :param y: targets on which to traih.
        :return: the metrics of interest as defined in the model (loss,
            accuracy, etc.)
        """
        x_train = np.asarray(x).astype('float32') / 255  # Normalize in 0-1 range
        u_train = np.asarray(u)
        y_train = np.asarray(y)

        if self.binarize:
            x_train[x_train < 0.1] = 0
            x_train[x_train >= 0.1] = 1
        return self.model.fit([x_train, u_train], y_train, class_weight=self.class_weight,
                              sample_weight=self.sample_weight,
                              nb_epoch=self.nb_epochs, validation_split=0.1)

    def train_on_batch(self, x, u, y):
        """
        Trains the model on a batch.
        :param x: batch of samples on which to train.
        :param y: targets for the batch.
        :return: the metrics of interest as defined in the model (loss,
            accuracy, etc.)
        """
        x_train = np.asarray(x).astype('float32') / 255  # Normalize in 0-1 range
        u_train = np.asarray(u)
        y_train = np.asarray(y)

        if self.binarize:
            x_train[x_train < 0.1] = 0
            x_train[x_train >= 0.1] = 1

        return self.model.train_on_batch([x_train, u_train], y_train, class_weight=self.class_weight,
                                         sample_weight=self.sample_weight)

    def predict(self, x, u):
        """
        Runs the given images through the model and returns the predictions.
        :param x: a batch of samples on which to predict.
        :return: the predictions of the batch.
        """
        # Feed input to the model, return encoded and re-decoded images
        x_test = np.asarray(x).astype('float32') / 255  # Normalize in 0-1 range
        u_test = np.asarray(u)

        return self.model.predict_on_batch([x_test, u_test])

    def test(self, x, y):
        """
        Tests the model on a batch.
        :param x: batch of samples on which to test.
        :return: the metrics of interest as defined in the model (loss,
            accuracy, etc.)
        """
        x_test = np.asarray(x).astype('float32') / 255  # Normalize in 0-1 range
        y_test = np.asarray(y)

        if self.binarize:
            x_test[x_test < 0.1] = 0
            x_test[x_test >= 0.1] = 1

        return self.model.test_on_batch(x_test, y_test)

    def all_features(self, sample):
        """
        Runs the given sample on the model and returns the features of the last
        dense layer in a 1d array.
        :param sample: a single sample to encode.
        :return: the encoded sample.
        """
        # Feed input to the model, return encoded images flattened
        sample = np.asarray(sample).astype('float32') / 255  # Normalize in 0-1 range

        return np.asarray(self.encoder.predict_on_batch(sample)).flatten()

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

    def save(self, filename=None, append=''):
        """
        Saves the model weights to disk (in the run folder if a logger was
        given, otherwise in the current folder)
        :param filename: custom filename for the hdf5 file.
        :param append: the model will be saved as model_append.h5 if a value is
            provided.
        """
        # Save the DQN weights to disk
        f = ('model%s.h5' % append) if filename is None else filename
        if not f.endswith('.h5'):
            f += '.h5'
        a = 'architecture_' + f.lstrip('.h5') + '.json'

        if self.logger is not None:
            self.logger.log('Saving model as %s' % self.logger.path + f)
            self.model.save_weights(self.logger.path + f)
            with open(self.logger.path + a, 'w') as a_file:
                a_file.write(self.model.to_json())
                a_file.close()
        else:
            self.model.save_weights(f)
            with open(a, 'w') as a_file:
                a_file.write(self.model.to_json())
                a_file.close()

    def load(self, path):
        """
        Load the model and its weights from path.
        :param path: path to an hdf5 file that stores weights for the model.
        """
        if self.logger is not None:
            self.logger.log('Loading weights from file...')
        self.model.load_weights(path)
