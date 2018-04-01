import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np

# Bitcoin price sign 20140428
origin_bitcoin_price = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]

print('- Query btc success')

# print('origin_bitcoin_price:', origin_bitcoin_price)

origin_bitcoin_price = origin_bitcoin_price.assign(Date=pd.to_datetime(origin_bitcoin_price['Date']))
origin_bitcoin_price.loc[origin_bitcoin_price['Volume'] == '-', 'Volume'] = 0
origin_bitcoin_price['Volume'] = origin_bitcoin_price['Volume'].astype('int64')

bitcoin_market_info = origin_bitcoin_price
bitcoin_market_info.head()

# Eth
origin_eth_price = pd.read_html("https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]

print('- Query eth success')

origin_eth_price = origin_eth_price.assign(Date=pd.to_datetime(origin_eth_price['Date']))
eth_market_info = origin_eth_price
eth_market_info.head()

# plot bitcoin and eth

def plot_price_and_volume(data, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    ax1.set_ylabel('Closing Price ($)', fontsize=12)
    ax1.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
    ax1.set_xticklabels('')

    ax2.set_ylabel('Volume', fontsize=12)
    ax2.set_yticks([int('%d000000000' % i) for i in range(10)])
    ax2.set_yticklabels(range(10))
    ax2.set_xticks([datetime.date(i, j, 1) for i in range(2013, 2019) for j in [1, 7]])
    ax2.set_xticklabels([datetime.date(i, j, 1).strftime('%b %Y') for i in range(2013, 2019) for j in [1, 7]])

    ax1.plot(data['Date'].astype(datetime.datetime), data['Open'])
    ax2.bar(data['Date'].astype(datetime.datetime).values, data['Volume'].values)

    fig.tight_layout()
    ax1.set_title(title)
    plt.show()


plot_price_and_volume(bitcoin_market_info, title='BitCoin')
plot_price_and_volume(eth_market_info, title='ETH')


# Merge Date

bitcoin_market_info.columns = [bitcoin_market_info.columns[0]] + ['bt_' + i for i in bitcoin_market_info.columns[1:]]
eth_market_info.columns = [eth_market_info.columns[0]] + ['eth_' + i for i in eth_market_info.columns[1:]]

market_info = pd.merge(bitcoin_market_info, eth_market_info, on=['Date'])
market_info = market_info[market_info['Date'] >= '2016-01-01']

for coins in ['bt_', 'eth_']:
    kwargs = {coins+'day_diff': lambda x: (x[coins + 'Close'] - x[coins + 'Open']) / x[coins + 'Open']}
    market_info = market_info.assign(**kwargs)

market_info.head()

# Training, Test & Random Walks

split_date = '2017-06-01'

training_set = market_info[market_info['Date'] < split_date]
test_set = market_info[market_info['Date'] >= split_date]

test_y = market_info[(market_info['Date'] + datetime.timedelta(days=1)) >= split_date]


def plot_as_normal_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(training_set['bt_day_diff'].values, bins=100)
    ax2.hist(training_set['eth_day_diff'].values, bins=100)
    ax1.set_title('Bitcoin Daily Price Changes')
    ax2.set_title('Ethereum Daily Price Changes')
    plt.show()


def random_walk():
    np.random.seed(202)
    bt_r_walk_mean = np.mean(training_set['bt_day_diff'].values)
    bt_r_walk_sd = np.std(training_set['bt_day_diff'].values)

    bt_random_steps = np.random.normal(
        bt_r_walk_mean, bt_r_walk_sd,
        (max(market_info['Date']).to_pydatetime() - datetime.datetime.strptime(split_date, '%Y-%m-%d')).days + 1)

    eth_r_walk_mean = np.mean(training_set['bt_day_diff'].values)
    eth_r_walk_sd = np.std(training_set['eth_day_diff'].values)

    eth_random_steps = np.random.normal(
        eth_r_walk_mean, eth_r_walk_sd,
        (max(market_info['Date']).to_pydatetime() - datetime.datetime.strptime(split_date, '%Y-%m-%d')).days + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_xticks([datetime.date(2017, i + 1, 1) for i in range(12)])
    ax1.set_xticklabels('')
    ax2.set_xticks([datetime.date(2017, i+1, 1) for i in range(12)])
    ax2.set_xticklabels([datetime.date(2017, i + 1, 1).strftime('%b %d %Y') for i in range(12)])

    ax1.plot(
        test_set['Date'].astype(datetime.datetime),
        test_set['bt_Close'].values, label='Actual')

    ax1.plot(test_set['Date'].astype(datetime.datetime),
             test_y['bt_Close'].values[1:] * (1+bt_random_steps),
             label='Predicted')

    ax2.plot(test_set['Date'].astype(datetime.datetime),
             test_set['eth_Close'].values, label='Actual')
    ax2.plot(test_set['Date'].astype(datetime.datetime),
             test_y['eth_Close'].values[1:] * (1+eth_random_steps), label='Predicted')

    ax1.set_title('Single Point Random Walk (Test set)')
    ax1.set_ylabel('Bitcoin Price ($)', fontsize=12)
    ax2.set_ylabel('Ethereum Price ($)', fontsize=12)
    plt.tight_layout()
    ax1.legend(bbox_to_anchor=(0.03, 1), loc=2, borderaxespad=0., prop={'size': 14})
    plt.show()


random_walk()

