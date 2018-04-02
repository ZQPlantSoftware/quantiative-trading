import numpy as np

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


def data_preprocessor(market_info, split_date, window_len, currency_type):
    for coins in currency_type:
        kwargs = {
            coins + 'close_off_high': lambda x: 2 * (x[coins + 'High'] - x[coins + 'Close']) / (
                        x[coins + 'High'] - x[coins + 'Low']) - 1,
            coins + 'volatility': lambda x: (x[coins + 'High'] - x[coins + 'Low']) / (x[coins + 'Open'])}

        market_info = market_info.assign(**kwargs)

    model_data = market_info[
        ['Date'] + [coin + metric for coin in currency_type
                    for metric in ['Close', 'Volume', 'close_off_high', 'volatility']]]

    model_data = model_data.sort_values(by='Date')
    model_data.head()

    origin_training_set = model_data[model_data['Date'] < split_date]
    origin_test_set = model_data[model_data['Date'] >= split_date]

    # Normalize data

    training_set = origin_training_set.drop('Date', 1)
    test_set = origin_test_set.drop('Date', 1)

    norm_cols = [coin + metric for coin in currency_type for metric in ['Close', 'Volume']]

    lstm_training_inputs = []

    for i in range(len(training_set) - window_len):
        temp_set = training_set[i:(i + window_len)].copy()
        for col in norm_cols:
            temp_set.loc[:, col] = temp_set[col] / temp_set[col].iloc[0] - 1

        lstm_training_inputs.append(temp_set)
    lstm_training_outputs = (training_set['eth_Close'][window_len:].values / training_set['eth_Close'][
                                                                             :-window_len].values) - 1

    lstm_test_inputs = []

    for i in range(len(test_set) - window_len):
        temp_set = test_set[i:(i + window_len)].copy()
        for col in norm_cols:
            temp_set.loc[:, col] = temp_set[col] / temp_set[col].iloc[0] - 1
        lstm_test_inputs.append(temp_set)

    lstm_test_outputs = (test_set['eth_Close'][window_len:].values / test_set['eth_Close'][:-window_len].values) - 1

    lstm_training_inputs = [np.array(lstm_training_input) for lstm_training_input in lstm_training_inputs]
    lstm_training_inputs = np.array(lstm_training_inputs)

    lstm_test_inputs = [np.array(lstm_test_inputs) for lstm_test_inputs in lstm_test_inputs]
    lstm_test_inputs = np.array(lstm_test_inputs)

    lstm_training_outputs = (training_set['eth_Close'][window_len:].values / training_set['eth_Close'][
                                                                             :-window_len].values) - 1

    return origin_training_set, origin_test_set, lstm_training_inputs, lstm_training_outputs, lstm_test_inputs, lstm_test_outputs
