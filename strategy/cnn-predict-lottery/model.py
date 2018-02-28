from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
from data_helper import getData

def LotteryCNNModel(input_shape):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(32, (input_shape[1], input_shape[0]), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name="bn0")(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), name="max_pool")(X)

    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # model = Model(input=X_input, output=X, name='LotteryModel')
    # return model


def LotteryLSTMModel(layers, neurons, d):
    model = Sequential()

    model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))

    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def LotteryNNModel():
    result_num = 33

    model = Sequential()
    model.add(Dense(6, activation='relu', input_dim=6))
    model.add(Dense(33, activation='relu'))
    model.add(Dense(33, activation='relu'))
    model.add(Dense(33, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model
