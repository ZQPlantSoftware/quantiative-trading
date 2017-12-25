import quandl
from get_data_quandl import get_stock_data
from split_data import load_data
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
# from keras.models import load_model
import keras

# Model Execution

def build_model(shape, neurons, dropout, decay):
    model = Sequential()

    model.add(LSTM(neurons[0], input_shape=(shape[0], shape[1]), return_sequences=True))
    model.add(Dropout(dropout))

    model.add(LSTM(neurons[1], input_shape=(shape[0], shape[1]), return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))
    model.add(Dense(neurons[3],kernel_initializer="uniform",activation='linear'))
    # model = load_model('my_LSTM_stock_model1000.h5')
    adam = keras.optimizers.Adam(decay=decay)
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

# Prediction vs Real Results

def percentage_difference(model, X_test, y_test):
    percentage_diff = []
    p = model.predict(X_test)

    for u in range(len(y_test)):
        pr = p[u][0]

        percentage_diff.append((pr - y_test[u] / pr) * 100)

    return p

# Input of the model

quandl.ApiConfig.api_key = 'zpFWg7jpwtBPmzA8sT2Z'
seq_len = 22
shape = [seq_len, 9, 1]
neurons = [256, 256, 32, 1]
dropout = 0.3
decay = 0.5 
epochs = 90
stock_name = 'AAPL'

# Pull the data from Quandl first

df = get_stock_data(stock_name, ma=[50, 100, 200])

X_train, y_train, X_test, y_test = load_data(df, seq_len)
model = build_model(shape, neurons, dropout, decay)
model.fit(
X_train,
    y_train,
    batch_size=512,
    epochs=epochs,
    validation_split=0.2,
    verbose=1
)
p = percentage_difference(model, X_test, y_test)