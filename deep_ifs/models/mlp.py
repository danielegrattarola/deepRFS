from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score


class MLP:
    def __init__(self, input_shape, output_size, layers=(512,)):
        self.model = Sequential()
        if isinstance(input_shape, tuple):
            self.input_shape = input_shape
        else:
            self.input_shape = (input_shape, )
        self.model.add(Dense(layers[0], activation='relu', input_shape=self.input_shape))
        if len(layers) > 1:
            for l in layers[1:]:
                self.model.add(Dense(l, activation='relu'))
        self.model.add(Dense(output_size))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X, Y, epochs=10, patience=2):
        es = EarlyStopping(patience=patience)
        return self.model.fit(X, Y, epochs=epochs, callbacks=[es])

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, Y):
        return r2_score(Y, self.model.predict(X))
