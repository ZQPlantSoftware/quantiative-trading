import tushare as ts
import matplotlib.pyplot as plt

df.get_tick_data('600848', data='2017-02-02')
engine = create_engine('mysql://root:TFMysql02@bj-cdb-52ew1l7v.sql.tencentcdb.com/quant?charset=utf8')

# Save to database


print(ts.get_stock_basics().length)