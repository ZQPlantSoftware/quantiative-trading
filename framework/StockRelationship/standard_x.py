from sklearn.preprocessing import StandardScaler

def standard_x(stocks):
    '''
    Change data structure from training set
    :param stocks: Training set
    :return:
    '''
    str_list = []
    for colname, colvalue in stocks.iteritems():
        if type(colvalue[1]) == str:
            str_list.append(colname)

    # Get to the numeric columns by inversion
    num_list = stocks.columns.difference(str_list)
    stocks_num = stocks[num_list]

    # print(num_list)
    # print(stocks_num.head())

    # fill NA/NaN values using the specified methods
    stocks_num = stocks_num.fillna(value=0, axis=1)
    X = stocks_num.values

    # Standardize features by removing the mean and scaling to unit variance
    X_std = StandardScaler().fit_transform(X)

    return X_std, stocks_num
