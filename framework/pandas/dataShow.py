#! usr/bin/python
# coding=utf-8

import numpy as np
import pandas as pd

from pandas import Series, DataFrame

# Set output screen width
pd.set_option('display.width', 200)

stock_list = ['000001.XSHE', '000002.XSHE', '000568.XSHE', '000625.XSHE', '000768.XSHE', '600028.XSHG', '600030.XSHG', '601111.XSHG', '601390.XSHG', '601998.XSHG']

raw_data = DataAPI.MktEqudGet(secID=stock_list, beginDate='20150101', endDate='20150131', pandas='1')

df = raw_data[['secID', 'tradeDate', 'secShortName', 'openPrice', 'highestPrice', 'lowestPrice', 'closePrice', 'turnoverVol']]

print "Order by column names, descending:"
print df.sort_index(axis=1, ascending=False).head()

print "Order by column value, ascending:"
print df.sort(columns='tradeDate').head()

print "Order by multiple columns value:"
df = df.sort(columns=['tradeDate', 'secID'], ascending=[False, True])
print df.head()

# handle mission data

# Example:

df['openPrice'][df['secID'] == '000001.XSHE'] = np.nan
df['highestPrice'][df['secID'] == '601111.XSHG'] = np.nan
df['lowestPrice'][df['secID'] == '601111.XSHG'] = np.nan
df['closePrice'][df['secID'] == '000002.XSHE'] = np.nan
df['turnoverVol'][df['secID'] == '601111.XSHG'] = np.nan
print df.head(10)

print "Drop all rows that have any NaN values:"
print "Data size after filtering:"
print df.dropna().shape
print df.dropna().head(0)

print "Drop only if all column are NaN:"
print df.dropna(how="all").shape
print df.dropna(how="all").head(10)

print "Drop rows who do not have at least six values that are not NaN"
print df.dropna(thresh=6).shape
print df.dropna(thresh=6).head(10)

print "Drop only if NaN in specific column:"
print "Data size after filtering:"
print df.dropna(subset=['closePrice']).shape
print df.dropna(subset=['closePrice']).head(10)

# 填补缺损数值

print df.fillna(value=20150101).head()

# 4. 数据可视化

dat = df[df['secID'] == '600028.XSHG'].set_index('tradeDate')['closePrice']
dat.plot(title="Close Price of SINOPEC (600028) during Jan, 2015")