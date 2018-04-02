import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np

def query_and_clean():
    # Bitcoin price sign 20140428
    origin_bitcoin_price = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]

    print('- Query btc success')

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

    # Merge Date

    bitcoin_market_info.columns = [bitcoin_market_info.columns[0]] + ['bt_' + i for i in bitcoin_market_info.columns[1:]]
    eth_market_info.columns = [eth_market_info.columns[0]] + ['eth_' + i for i in eth_market_info.columns[1:]]

    market_info = pd.merge(bitcoin_market_info, eth_market_info, on=['Date'])
    market_info = market_info[market_info['Date'] >= '2016-01-01']

    for coins in ['bt_', 'eth_']:
        kwargs = {coins+'day_diff': lambda x: (x[coins + 'Close'] - x[coins + 'Open']) / x[coins + 'Open']}
        market_info = market_info.assign(**kwargs)

    return market_info
