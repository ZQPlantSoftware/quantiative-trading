
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from tools.data_clean import query_and_clean
from tools.model_persistence import load_model
from data_preprocess import data_preprocessor
from train_model import train_model
import numpy as np
import matplotlib.pyplot as plt
import datetime

# configuration

split_date = '2017-06-01'
window_len = 10
currency_type = ['bt_', 'eth_']
persistence_path = 'model.h5'


def do_predict(need_train=True, need_save=True):
    # data preprocessor
    market_info = query_and_clean()
    training_set, test_set, lstm_training_inputs, lstm_training_outputs, lstm_test_inputs, lstm_test_outputs \
        = data_preprocessor(market_info, split_date, window_len, currency_type=currency_type)

    # train or load model
    if need_train:
        model = train_model(
            lstm_training_inputs, lstm_training_outputs, need_save_model=need_save)
    else:
        model = load_model(persistence_path)

    # ETH tomorrow price predict
    turn_datetime = training_set['Date'][window_len:].astype(datetime.datetime)
    eth_close = training_set['eth_Close'][window_len:]
    eth_close_end = training_set['eth_Close'][:-window_len]

    x_ticks = [datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 5, 9]]
    x_ticks_labels = [datetime.date(i, j, 1).strftime('%b %Y') for i in range(2013, 2019) for j in [1, 5, 9]]

    predict = np.transpose(
        model.predict(lstm_training_inputs)
    ) + 1

    print('- Predict success with result:', predict)

    predict_time_eth_close = (predict * eth_close_end)[0]

    mae = np.mean(
        np.abs(predict - \
               eth_close /
               eth_close_end)
    )

    def plot_predict_with_actual():
        fig, ax1 = plt.subplot(1, 1)

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

    plot_predict_with_actual()
