import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from standard_x import standard_x
from pearson_correlation import pearson_correlation
from explained_variance import explained_variance

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

#
#
# # pca = PCA(n_components=9)
# # x_9d = pca.fit_transform(X_std)
# #
# # plt.figure(figsize = (9,7))
# # plt.scatter(x_9d[:,0],x_9d[:,1], c='goldenrod',alpha=0.5)
# # plt.ylim(-10,30)
# # plt.show()