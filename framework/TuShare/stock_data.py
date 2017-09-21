#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tushare as ts
from sqlalchemy import create_engine

df = ts.get_stock_basics()
engine = create_engine('mysql://root:TFMysql02@bj-cdb-52ew1l7v.sql.tencentcdb.com:63992/quant?charset=utf8')

values = df[(df.rev > 2) & (df.pe > 0) & (df.pe < 300) & (df.rev < 50)]
values.loc[:, 'peg'] = values.pe / values.rev

# Save to database
# df.to_sql('stock_fund_peg', engine, if_exists='append')
# Append to exists data table
# df.to_sql('stock_fund_peg', engine, if_exists='append')

values.sort_index(by='peg', ascending=True).to_csv('/Users/georgezou/PycharmProjects/quantitative-trading/data/peg.csv')

print('job done')