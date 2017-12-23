import quandl
from sklearn import preprocessing
import pandas as pd

def get_stock_data(stock_name, normalize=True, ma=[]):
    df = quandl.get_table('WIKI/PRICES', ticker = stock_name)
    df.drop(['ticker', 'open', 'high', 'low', 'close', 'ex-dividend', 'volume', 'split_ratio'], 1, inplace=True)
    df.set_index('date', inplace=True)

    # Renaming all the columns so that we can use the old version code
    df.rename(columns={'adj_open': 'Open', 'adj_high': 'High', 'adj_low': 'Low', 'adj_volume': 'Volume', 'adj_close': 'Adj Close'}, inplace=True)

    # Percentage change
    df['Pct'] = df['Adj Close'].pct_change()
    df.dropna(inplace=True)

    if ma != []:
        for moving in ma:
            df['{}ma'.format(moving)] = df['Adj Close'].rolling(window=moving).mean()

        df.dropna(inplace=True)

        if normalize:
            min_max_scaler = preprocessing.MinMaxScaler()
            df['Open'] = min_max_scaler.fit_transform(df.Open.reshape(-1, 1))
            df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1, 1))
            df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1, 1))
            df['Volume'] = min_max_scaler.fit_transform(df.Volume.values.reshape(-1, 1))
            df['Adj Close'] = min_max_scaler.fit_transform(df['Adj Close'].values.reshape(-1, 1))
            df['Pct'] = min_max_scaler.fit_transform(df['Pct'].values.reshape(-1, 1))

            if ma != []:
                for moving in ma:
                    df['{}ma'.format(moving)] = min_max_scaler.fit_transform(df['{}ma'.format(moving)].values.reshape(-1, 1))

        adj_close = df['Adj Close']
        df.drop(labels=['Adj Close'], axis=1, inplace=True)
        df = pd.concat([df, adj_close], axis=1)

        return df