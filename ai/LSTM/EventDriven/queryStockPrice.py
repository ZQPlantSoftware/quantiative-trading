import numpy as np
import csv, json
import pandas as pd

def interpolateDf(df):
    df1 = df
    idx = pd.date_range('12-29-2006', '12-31-2016')
    df1.index = pd.DatatimeIndex(df1.index)
    df1 = df1.reindex(idx, fill_value=np.NaN)
    interpolated_df = df1.interpolate()
    interpolated_df.count()
    return interpolated_df

def queryStockPrice():
    with open(
            '/Users/georgezou/Documents/Coding/ML/Neural-Network-with-Financial-Time-Series-Data/data/DJIA_indices_data.csv',
            'r', encoding="utf-8") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # Converting the csv file reader to a lists
        data_list = list(spamreader)

    # Seoarating header from the data
    header = data_list[0]
    data_list = data_list[1:]
    data_list = np.asarray(data_list)

    print(data_list)

    # Selecting date and close value and adj close for each day
    # Volume will be added in the next model
    selected_data = data_list[:, [0, 4, 6]]

    # Convert it into dataframe
    # index = date
    df = pd.DataFrame(data=selected_data[0:, 1:],
                      index=selected_data[0:, 0],
                      columns=['close', 'adj close'],
                      dtype='float64')

    return interpolateDf(df)