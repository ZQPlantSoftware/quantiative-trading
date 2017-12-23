import quandl
import get_stock_data from get_data_quandl

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
