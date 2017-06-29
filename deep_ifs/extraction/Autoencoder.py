from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np


class Autoencoder:
    def __init__(self, input_shape, encoding_dim=512, nb_epochs=10,
                 dropout_prob=0.5, binarize=False, class_weight=None,
                 sample_weight=None, load_path=None, logger=None,
                 ckpt_file=None):
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.nb_epochs = nb_epochs
        self.dropout_prob = dropout_prob
        self.binarize = binarize
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.logger = logger
        self.decoding_available = False
        self.use_contractive_loss = False
        self.support = None

        if ckpt_file is not None:
            self.ckpt_file = ckpt_file if logger is None else (logger.path + ckpt_file)
        else:
            self.ckpt_file = 'NN.h5' if logger is None else (logger.path + 'NN.h5')

        self.es = EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=5)

        self.mc = ModelCheckpoint(self.ckpt_file, monitor='val_loss',
                                  save_best_only=True, save_weights_only=True,
                                  verbose=0)

        # Build network
        self.input = Input(shape=(4, 108, 84))

        self.encoded = Conv2D(32, (8, 8), padding='valid',
                              activation='relu', strides=(4, 4),
                              data_format='channels_first')(self.input)

        self.encoded = Conv2D(64, (4, 4), padding='valid',
                              activation='relu', strides=(2, 2),
                              data_format='channels_first')(self.encoded)

        self.encoded = Conv2D(64, (3, 3), padding='valid',
                              activation='relu', strides=(1, 1),
                              data_format='channels_first')(self.encoded)

        self.encoded = Conv2D(16, (3, 3), padding='valid',
                              activation='relu', strides=(1, 1),
                              data_format='channels_first')(self.encoded)

        # Features
        self.features = Flatten()(self.encoded)

        # Decoded
        self.decoded = Reshape((16, 8, 5))(self.features)

        self.decoded = Conv2DTranspose(16, (3, 3), padding='valid',
                                       activation='relu', strides=(1, 1),
                                       data_format='channels_first')(self.decoded)

        self.decoded = Conv2DTranspose(64, (3, 3), padding='valid',
                                       activation='relu', strides=(1, 1),
                                       data_format='channels_first')(self.decoded)

        self.decoded = Conv2DTranspose(64, (4, 4), padding='valid',
                                       activation='relu', strides=(2, 2),
                                       data_format='channels_first')(self.decoded)

        self.decoded = Conv2DTranspose(64, (8, 8), padding='valid',
                                       activation='relu', strides=(4, 4),
                                       data_format='channels_first')(self.decoded)

        self.decoded = Conv2DTranspose(4, (1, 1), padding='valid',
                                       activation='sigmoid', strides=(1, 1),
                                       data_format='channels_first')(self.decoded)

        # Models
        self.model = Model(inputs=self.input, outputs=self.decoded)
        self.encoder = Model(inputs=self.input, outputs=self.features)

        # Build decoder model
        if self.decoding_available:
            self.encoded_input = Input(shape=(16 * 8 * 5,))
            self.decoding_intermediate = self.model.layers[-6](self.encoded_input)
            self.decoding_intermediate = self.model.layers[-5](self.decoding_intermediate)
            self.decoding_intermediate = self.model.layers[-4](self.decoding_intermediate)
            self.decoding_intermediate = self.model.layers[-3](self.decoding_intermediate)
            self.decoding_intermediate = self.model.layers[-2](self.decoding_intermediate)
            self.decoding_output = self.model.layers[-1](self.decoding_intermediate)
            self.decoder = Model(input=self.encoded_input, output=self.decoding_output)

        # Optimization algorithm
        self.optimizer = Adam()

        # Load the network from saved model
        if load_path is not None:
            self.load(load_path)

        if self.use_contractive_loss:
            self.loss = self.contractive_loss
        else:
            self.loss = 'binary_crossentropy'

        self.model.compile(optimizer=self.optimizer, loss=self.loss,
                           metrics=['accuracy'])

    def preprocess_state(self, x, binarize=False):
        if not x.shape[1:] == (4, 108, 84):
            x = x[:, :, 2:, :]
            assert x.shape[1:] == (4, 108, 84)
        x = np.asarray(x).astype('float32') / 255.  # To 0-1 range
        if binarize:
            x[x < 0.1] = 0
            x[x >= 0.1] = 1

        return x

    def fit(self, x, y, validation_data=None):
        """
        Trains the model on a set of batches.

        Args
            x: samples on which to train.
            u: actions associated to the samples.
            y: targets on which to train.
            validation_data: tuple (X, Y) to use as validation data
        Returns
            The metrics of interest as defined in the model (loss, accuracy,
                etc.)
        """
        # Preprocess training data
        x_train = self.preprocess_state(np.asarray(x), binarize=self.binarize)
        y_train = self.preprocess_state(np.asarray(y), binarize=self.binarize)

        # Preprocess validation data
        if validation_data is not None:
            val_x = self.preprocess_state(validation_data[0], binarize=self.binarize)
            val_y = self.preprocess_state(validation_data[1], binarize=self.binarize)
            validation_data = (val_x, val_y)

        return self.model.fit(x_train, y_train,
                              class_weight=self.class_weight,
                              sample_weight=self.sample_weight,
                              epochs=self.nb_epochs,
                              validation_data=validation_data,
                              callbacks=[self.es, self.mc])

    def fit_generator(self, generator, steps_per_epoch, nb_epochs,
                      validation_data=None):
        # Preprocess validation data
        if validation_data is not None:
            val_x = self.preprocess_state(validation_data[0], binarize=self.binarize)
            val_y = self.preprocess_state(validation_data[1], binarize=self.binarize)
            validation_data = (val_x, val_y)

        return self.model.fit_generator(generator,
                                        steps_per_epoch,
                                        epochs=nb_epochs,
                                        max_q_size=250,
                                        callbacks=[self.es, self.mc],
                                        validation_data=validation_data)

    def predict(self, x):
        """
        Runs the given images through the model and returns the predictions.

        Args
            x: a batch of samples on which to predict.
            u: actions associated to the samples.
        Returns
            The predictions of the batch.
        """
        # Feed input to the model, return encoded and re-decoded images
        x_test = self.preprocess_state(x, binarize=self.binarize)
        pred = self.model.predict(x_test)

        return pred

    def all_features(self, x):
        """
        Runs the given samples on the model and returns the features of the last
        dense layer in an array.

        Args
            x: samples to encode.
        Returns
            The encoded sample.
        """
        # Feed input to the model, return encoded images flattened
        x = self.preprocess_state(x, binarize=self.binarize)

        if x.shape[0] == 1:
            # x is a singe sample
            return np.asarray(self.encoder.predict_on_batch(x)).flatten()
        else:
            return np.asarray(self.encoder.predict(x))

    def s_features(self, x, support=None):
        """
        Runs the given samples on the model and returns the features of the last
        dense layer filtered by the support mask.

        Args
            x: samples to encode.
            support: a boolean mask with which to filter the output.
        Returns
            The encoded sample.
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

    def save(self, filename=None, append=''):
        """
        Saves the model weights to disk (in the run folder if a logger was
        given, otherwise in the current folder)

        Args
            filename: custom filename for the hdf5 file.
            append: the model will be saved as model_append.h5 if a value is
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

    def save_encoder(self, filepath):
        """
        Save the encoder weights at filepath.

        Args
            filepath: path to an hdf5 file to store weights for the model.
        """
        self.encoder.save(filepath)

    def load(self, path):
        """
        Load the model and its weights from path.

        Args
            path: path to an hdf5 file that stores weights for the model.
        """
        if self.logger is not None:
            self.logger.log('Loading weights from file...')
        self.model.load_weights(path)

    def set_support(self, support):
        self.support = support

    def contractive_loss(self, y_pred, y_true):
        mse = K.mean(K.square(y_true - y_pred), axis=-1)

        W = K.variable(value=self.model.get_layer('features').get_weights()[0])  # N x N_hidden
        W = K.transpose(W)  # N_hidden x N
        h = self.model.get_layer('features').output
        dh = h * (1 - h)  # N_batch x N_hidden

        # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
        contractive = 1e-03 * K.sum(dh ** 2 * K.sum(W ** 2, axis=1), axis=1)

        return mse + contractive
