
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from tools.data_clean import query_and_clean
from tools.model_persistence import load_model
from data_preprocess import data_preprocessor
from train_model import train_model
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import os

# configuration

split_date = '2017-06-01'
window_len = 10
currency_type = ['bt_', 'eth_']
persistence_path = 'model.json'
base_path = 'work/data/'
market_info_path = 'market_info.h5'


def predict_and_plot(X, y, data_set, x_ticks, x_ticks_labels, model, title, need_zoom):
    turn_datetime = data_set['Date'][window_len:].astype(datetime.datetime)
    close = y.values[window_len:]
    close_end = y.values[:-window_len]
    # close = data_set['eth_Close'].values[window_len:]
    # close_end = data_set['eth_Close'].values[:-window_len]

    predict = np.transpose(
        model.predict(X)
    ) + 1

    print('- Predict success')

    predict_time_close = (predict * close_end)[0]

    mae = np.mean(
        np.abs(predict - \
               close /
               close_end)
    )

    def plot_predict_with_actual():
        fig, ax1 = plt.subplots(1, 1)

        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_ticks_labels)
        ax1.plot(
            turn_datetime,
            close,
            label='Actual')

        ax1.plot(
            turn_datetime,
            predict_time_close,
            label='Predicted'
        )

        ax1.set_title(title)
        ax1.set_ylabel('Ethereum Price ($)', fontsize=12)
        ax1.legend(bbox_to_anchor=(0.15, 1), loc=2, borderaxespad=0., prop={'size': 14})
        ax1.annotate('MAE: %.4f' % mae,
                     xy=(0.75, 0.9),
                     xycoords='axes fraction',
                     xytext=(0.75, 0.9),
                     textcoords='axes fraction')

        if need_zoom:
            axins = zoomed_inset_axes(ax1, 3.35, loc=10)
            axins.set_xticks(x_ticks)
            axins.plot(turn_datetime,
                       close,
                       label='Actual')

            axins.plot(turn_datetime,
                       predict_time_close,
                       label='Predicted')

            axins.set_xlim([datetime.date(2017, 3, 1), datetime.date(2017, 5, 1)])
            axins.set_ylim([10, 60])
            axins.set_xticklabels('')
            mark_inset(ax1, axins, loc1=1, loc2=3, fc="none", ec="0.5")
        plt.show()

    plot_predict_with_actual()


def do_predict(need_train=True, need_save=True):
    # data preprocessor
    data_path = base_path + market_info_path
    if os.path.exists(data_path):
        market_info = pd.read_pickle(data_path)
    else:
        market_info = query_and_clean()
        market_info.to_pickle(data_path)

    training_set, test_set, lstm_training_inputs, lstm_training_outputs, lstm_test_inputs, lstm_test_outputs \
        = data_preprocessor(market_info, split_date, window_len, currency_type=currency_type)

    # train or load model
    if need_train:
        model = train_model(
            lstm_training_inputs, lstm_training_outputs, need_save_model=need_save)
    else:
        model = load_model(persistence_path)

    training_x_ticks = [datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 5, 9]]
    training_x_ticks_labels = [datetime.date(i, j, 1).strftime('%b %Y') for i in range(2013, 2019) for j in [1, 5, 9]]

    test_x_ticks = [datetime.date(2017, i+1, 1) for i in range(12)]
    test_x_ticks_labels = [datetime.date(2017, i+1, 1).strftime('%b %d %Y') for i in range(12)]

    # Eth predict plot
    predict_and_plot(lstm_training_inputs, training_set['eth_Close'], training_set, training_x_ticks, training_x_ticks_labels, model, "[ETH] Training Set: Single Timepoint Prediction", need_zoom=True)
    predict_and_plot(lstm_test_inputs, test_set['eth_Close'], test_set, test_x_ticks, test_x_ticks_labels, model, "[ETH] Test Set: Single Timepoint Prediction", need_zoom=False)

    # Here we should train bit coin data and do the predict

    # Btc predict plot
    # predict_and_plot(lstm_training_inputs, training_set['bt_Close'], training_set, training_x_ticks, training_x_ticks_labels, model, "[BTC] Training Set: Single Timepoint Prediction", need_zoom=True)
    # predict_and_plot(lstm_test_inputs, test_set['bt_Close'], test_set, test_x_ticks, test_x_ticks_labels, model, "[BTC] Test Set: Single Timepoint Prediction", need_zoom=False)
