import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from standard_x import standard_x
from pearson_correlation import pearson_correlation
from explained_variance import explained_variance
from k_means_clustering import k_means_clustering
from pca import plt_pca

np.seterr(divide='ignore', invalid='ignore')

# Quick way to test just a few column features
# stocks = pd.read_csv('supercolumns-elements-nasdaq-nyse-otcbb-general-UPDATE-2017-03-01.csv', usecols=range(1,16))

stocks = pd.read_csv('supercolumns-elements-nasdaq-nyse-otcbb-general-UPDATE-2017-03-01.csv')

X_std, stocks_num = standard_x(stocks)

#################################################################
# Pearson Correlation of Concept Feature

# pearson_correlation(stocks_num)

#################################################################
# Calculting Eigenvectors and eigenvalues of Cov matrix

explained_variance(stocks, X_std)

#################################################################
#

x_9d = plt_pca(X_std)

#################################################################
#

k_means_clustering(x_9d)