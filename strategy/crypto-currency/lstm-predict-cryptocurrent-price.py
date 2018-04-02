from data_clean import query_and_clean
import numpy as np
import matplotlib.pyplot as plt
import datetime

# lstm

currency_type = ['bt_', 'eth_']

"""
## Addition Attribute

    - close_off_high: The gap between the closing price and price high for the day
        -1 and 1 mean the closing price was equal to the daily low or daily high
    
    - volatility: The difference between high and low price divided by the opening 
        price

## Prediction

    The next day's closing price of a specific coin.

## Input
    10 Days previous days access
"""
market_info = query_and_clean()

for coins in currency_type:
    kwargs = {
        coins + 'close_off_high': lambda x: 2 * (x[coins+'High'] - x[coins + 'Close']) / (x[coins + 'High'] - x[coins + 'Low']) - 1,
        coins + 'volatility': lambda x: (x[coins + 'High'] - x[coins + 'Low']) / (x[coins + 'Open'])}

    market_info = market_info.assign(**kwargs)

model_data = market_info[
    ['Date'] + [coin + metric for coin in currency_type
        for metric in ['Close', 'Volume', 'close_off_high', 'volatility']]]

model_data = model_data.sort_values(by='Date')
model_data.head()

split_date = '2017-06-01'
training_set = model_data[model_data['Date'] < split_date]
test_set = model_data[model_data['Date'] >= split_date]

# Normalize data

training_set = training_set.drop('Date', 1)
test_set = test_set.drop('Date', 1)

window_len = 10
norm_cols = [coin+metric for coin in currency_type for metric in ['Close', 'Volume']]

LSTM_training_inputs = []

for i in range(len(training_set) - window_len):
    temp_set = training_set[i:(i + window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1

    LSTM_training_inputs.append(temp_set)
LSTM_training_outputs = (training_set['eth_Close'][window_len:].values/training_set['eth_Close'][:-window_len].values) - 1

LSTM_test_inputs = []

for i in range(len(test_set) - window_len):
    temp_set = test_set[i:(i+window_len)].copy()
    for col in norm_cols:
        temp_set.loc[:, col] = temp_set[col] / temp_set[col].iloc[0] - 1
    LSTM_test_inputs.append(temp_set)

LSTM_test_outputs = (test_set['eth_Close'][window_len:].values / test_set['eth_Close'][:-window_len].values) - 1

LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)

# LSTM model

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout


"""
Layers has been shaped to fit our inputs (
    n * m tables, n: the number of timepoints/rows w: columns)
    
"""


def build_model(inputs, output_size, neurons, activ_func="linear", dropout=0.25, loss='mae', optimizer='adam'):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


np.random.seed(202)
eth_model = build_model(LSTM_training_inputs, output_size=1, neurons=20)
LSTM_training_outputs = (training_set['eth_Close'][window_len:].values / training_set['eth_Close'][:-window_len].values) - 1
eth_history = eth_model.fit(LSTM_training_inputs, LSTM_training_outputs,
                            epochs=50, batch_size=1, verbose=2, shuffle=True)


def plot_error():
    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(eth_history.epoch, eth_history.history['loss'])
    ax1.set_title('Training Error')

    if eth_model.loss == 'mae':
        ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    # just in case you decided to change the model loss calculation
    else:
        ax1.set_ylabel('Model Loss', fontsize=12)
    ax1.set_xlabel('# Epochs', fontsize=12)
    plt.show()


# plot_error()

# ETH tomorrow price predict

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


def plot_predict_eth_with_actual():
    fig, ax1 = plt.subplot(1, 1)
    turn_datetime = training_set['Date'][window_len:].astype(datetime.datetime)
    eth_close = training_set['eth_Close'][window_len:]
    eth_close_end = training_set['eth_Close'][:-window_len]

    x_ticks = [datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 5, 9]]
    x_ticks_labels = [datetime.date(i, j, 1).strftime('%b %Y') for i in range(2013, 2019) for j in [1, 5, 9]]

    predict = np.transpose(
        eth_model.predict(LSTM_training_inputs)
    ) + 1

    predict_time_eth_close = (predict * eth_close_end)[0]

    mae = np.mean(
        np.abs(predict - \
               eth_close /
               eth_close_end
        )
    )

    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticks_labels)
    ax1.plot(
        turn_datetime,
        eth_close,
        label='Actual')

    ax1.plot(
        turn_datetime,
        predict_time_eth_close,
        label='Predicted'
    )

    ax1.set_title('Training Set: Single Timepoint Prediction')
    ax1.set_ylabel('Ethereum Price ($)', fontsize=12)
    ax1.legend(bbox_to_anchor=(0.15, 1), loc=2, borderaxespad=0., prop={'size': 14})
    ax1.annotate('MAE: %.4f' % mae,
                 xy=(0.75, 0.9),
                 xycoords='axes fraction',
                 xytext=(0.75, 0.9),
                 textcoords='axes fraction')

    # matplotlib zoom from http://akuederle.com/matplotlib-zoomed-up-inset

    axins = zoomed_inset_axes(ax1, 3,35, loc=10)
    axins.set_xticks(x_ticks)
    axins.plot(turn_datetime,
               eth_close,
               label='Actual')

    axins.plot(turn_datetime,
               predict_time_eth_close,
               label='Predicted')

    axins.set_xlim([datetime.date(2017, 3, 1), datetime.date(2017, 5, 1)])
    axins.set_ylim([10, 60])
    axins.set_xticklabels('')
    mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    plt.show()


plot_predict_eth_with_actual()
