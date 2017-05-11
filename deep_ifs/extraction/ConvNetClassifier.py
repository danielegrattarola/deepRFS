import numpy as np
from keras.models import Model
from keras.layers import Input, Convolution2D, Flatten, Dense, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1
from sklearn.exceptions import NotFittedError

from deep_ifs.extraction.GatherLayer import GatherLayer


class ConvNetClassifier:
    def __init__(self, input_shape, nb_classes, nb_actions=1, encoding_dim=512,
                 nb_epochs=10, dropout_prob=0.5, l1_alpha=0.01, binarize=False,
                 class_weight=None, sample_weight=None, load_path=None,
                 logger=None, chkpt_file=None):
        self.dim_ordering = 'th'  # (samples, filters, rows, cols)
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.nb_actions = nb_actions
        self.encoding_dim = encoding_dim
        self.nb_epochs = nb_epochs
        self.dropout_prob = dropout_prob
        self.l1_alpha = l1_alpha
        self.binarize = binarize
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.logger = logger

        if chkpt_file is not None:
            self.chkpt_file = chkpt_file if logger is None else (logger.path + chkpt_file)
        else:
            self.chkpt_file = 'NN.h5' if logger is None else (logger.path + 'NN.h5')

        self.es = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5)

        self.mc = ModelCheckpoint(self.chkpt_file, monitor='val_loss',
                                  save_best_only=True, save_weights_only=True,
                                  verbose=1)

        # Build network
        self.input = Input(shape=self.input_shape)
        self.u = Input(shape=(1,), dtype='int32')

        self.hidden = Convolution2D(32, (8, 8), padding='valid',
                                    activation='relu', strides=(4, 4),
                                    data_format='channels_first')(self.input)

        self.hidden = Convolution2D(64, (4, 4), padding='valid',
                                    activation='relu', strides=(2, 2),
                                    data_format='channels_first')(self.hidden)

        self.hidden = Convolution2D(64, (3, 3), padding='valid',
                                    activation='relu', strides=(1, 1),
                                    data_format='channels_first')(self.hidden)

        self.hidden = Flatten()(self.hidden)
        self.features = Dense(self.encoding_dim, activation='relu')(self.hidden)
        self.output = Dense(self.nb_classes * self.nb_actions,
                            activation='softmax',
                            activity_regularizer=l1(self.l1_alpha))(self.features)
        self.output_u = GatherLayer(self.nb_classes, self.nb_actions)([self.output, self.u])

        # Models
        self.model = Model(outputs=[self.output_u], inputs=[self.input, self.u])
        self.encoder = Model(outputs=[self.features], inputs=[self.input])

        # Optimization algorithm
        self.optimizer = Adam()

        # Load the network from saved model
        if load_path is not None:
            self.load(load_path)

        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy',
                           metrics=['accuracy'])

    def fit(self, x, u, y, validation_data=None):
        """
        Trains the model on a set of batches.

        Args
            x: samples on which to train.
            u: actions associated to the samples.
            y: targets on which to train.
            validation_data: tuple ([X, U], Y) to use as validation data
        Returns
            The metrics of interest as defined in the model (loss, accuracy,
                etc.)
        """
        # Preprocess training data
        x_train = np.asarray(x).astype('float32') / 255  # Convert to 0-1 range
        u_train = np.asarray(u)
        y_train = np.asarray(y)
        if self.scaler is not None:
            y_train = self.scaler.fit_transform(y_train)
            if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
                print('WARNING: nan in y_train.')
        if self.binarize:
            x_train[x_train < 0.1] = 0
            x_train[x_train >= 0.1] = 1

        # Preprocess validation data
        if validation_data is not None:
            val_x = np.asarray(validation_data[0][0]).astype('float32') / 255
            val_u = np.asarray(validation_data[0][1])
            val_y = np.asarray(validation_data[1])
            if self.scaler is not None:
                val_y = self.scaler.transform(val_y)
            if np.any(np.isnan(val_y)) or np.any(np.isinf(val_y)):
                print('WARNING: nan in val_y.')
            if self.binarize:
                val_x[val_x < 0.1] = 0
                val_x[val_x >= 0.1] = 1
            validation_data = ([val_x, val_u], val_y)

        return self.model.fit([x_train, u_train], y_train,
                              class_weight=self.class_weight,
                              sample_weight=self.sample_weight,
                              epochs=self.nb_epochs,
                              validation_data=validation_data,
                              callbacks=[self.es, self.mc])

    def fit_generator(self, generator, steps_per_epoch, nb_epochs,
                      validation_data=None, clip=False):
        # Preprocess validation data
        if validation_data is not None:
            val_x = np.asarray(validation_data[0][0]).astype('float32') / 255
            val_u = np.asarray(validation_data[0][1])
            val_y = np.asarray(validation_data[1])
            if self.scaler is not None:
                val_y = self.scaler.transform(val_y)
            if self.binarize:
                val_x[val_x < 0.1] = 0
                val_x[val_x >= 0.1] = 1
            if clip:
                val_y = np.clip(val_y, -1, 1)
            validation_data = ([val_x, val_u], val_y)

        return self.model.fit_generator(generator,
                                        steps_per_epoch,
                                        epochs=nb_epochs,
                                        max_q_size=20,
                                        callbacks=[self.es, self.mc],
                                        validation_data=validation_data)

    def train_on_batch(self, x, u, y):
        """
        Trains the model on a batch.

        Args
            x: batch of samples on which to train.
            u: actions associated to the samples.
            y: targets for the batch.
        Returns
            The metrics of interest as defined in the model (loss, accuracy,
                etc.)
        """
        x_train = np.asarray(x).astype('float32') / 255  # Convert to 0-1 range
        u_train = np.asarray(u)
        y_train = np.asarray(y)
        if self.binarize:
            x_train[x_train < 0.1] = 0
            x_train[x_train >= 0.1] = 1
        return self.model.train_on_batch([x_train, u_train], y_train,
                                         class_weight=self.class_weight,
                                         sample_weight=self.sample_weight)

    def predict(self, x, u):
        """
        Runs the given images through the model and returns the predictions.

        Args
            x: a batch of samples on which to predict.
            u: actions associated to the samples.
        Returns
            The predictions of the batch.
        """
        # Feed input to the model, return encoded and re-decoded images
        x_test = np.asarray(x).astype('float32') / 255  # Convert to 0-1 range
        if self.binarize:
            x_test[x_test < 0.1] = 0
            x_test[x_test >= 0.1] = 1
        u_test = np.asarray(u)
        return self.model.predict([x_test, u_test])

    def test(self, x, y):
        """
        Tests the model on a batch.

        Args
            x: batch of samples on which to test.
            y: real targets for the batch.
        Returns
            The metrics of interest as defined in the model (loss, accuracy,
                etc.)
        """
        x_test = np.asarray(x).astype('float32') / 255  # Convert to 0-1 range
        y_test = np.asarray(y)
        if self.binarize:
            x_test[x_test < 0.1] = 0
            x_test[x_test >= 0.1] = 1
        return self.model.test_on_batch(x_test, y_test)

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
        x = np.asarray(x).astype('float32') / 255  # To 0-1 range
        if self.binarize:
            x[x < 0.1] = 0
            x[x >= 0.1] = 1

        if x.shape[0] == 1:
            # x is a singe sample
            return np.asarray(self.encoder.predict_on_batch(x)).flatten()
        else:
            return np.asarray(self.encoder.predict(x))

    def s_features(self, x, support):
        """
        Runs the given samples on the model and returns the features of the last
        dense layer filtered by the support mask.

        Args
            x: samples to encode.
            support: a boolean mask with which to filter the output.
        Returns
            The encoded sample.
        """
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
