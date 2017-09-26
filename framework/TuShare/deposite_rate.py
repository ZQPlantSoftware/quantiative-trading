#!/usr/bin/python
# -*- coding: UTF-8 -*-
# coding:utf-8

import matplotlib.pyplot as plt
import tushare as ts
import pandas as pd
from sqlalchemy import create_engine

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

engine = create_engine('mysql://root:TFMysql02@bj-cdb-52ew1l7v.sql.tencentcdb.com:63992/quant?charset=utf8')

df = ts.get_deposit_rate()

df = df[df.deposit_type == u'定期存款整存整取(一年)']

# df.to_csv('/Users/georgezou/PycharmProjects/quantitative-trading/data/deposit.csv', encoding="utf-8")
# df.to_sql('deposit_rate', engine, if_exists='append')

df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")

df = df[df.rate > 0]

df['rate'] = df.rate.astype(float)

ax = df.plot(x='date', y='rate')
print('never be here:')
print('never be here:')

plt.title(u'1986 - 2017 China One-year Deposit Interest Rate')
plt.show()

print('job done!')