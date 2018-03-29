from cryptory import Cryptory
import pandas as pd

my_cryptory = Cryptory(from_date="2017-01-01")
pd.options.display.max_rows = 6

all_coins_df = my_cryptory.extract_bitinfocharts("btc")

bit_info_coins = ['btc', 'eth', 'xrp', 'bch', 'ltc']

for coin in bit_info_coins[1:]:
    all_coins_df = all_coins_df.merge(my_cryptory.extract_bitinfocharts(coin), on='date', bow='left')

import seaborn as sns
import matplotlib.pyplot as plt
import datetime

fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))

ca = plt.get_cmap('gist_rainbow')

print(my_cryptory.head())
# Help on class Cryptory in module cryptory.cryptory:

my_cryptory.extract_poloniex(coin1="btc", coin2="eth")

