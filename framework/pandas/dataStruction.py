#! usr/bin/python
# coding=utf-8

import numpy as np
import pandas as pd

from pandas import Series, DataFrame

# Set output screen width
pd.set_option('display.width', 200)

# 创建以日期为元素的Series:
dates = pd.date_range('20150101', periods=5)

# 将这个 Series 赋值给 DataFrame:
df = pd.DataFrame(np.random.randn(5, 4), index=dates, columns=list('ABCD'))

print df

# 只要是能转成 Series 的对象, 都可以用于创建 DataFrame

df2 = pd.DataFrame({
    'A': 1.,
    'B': pd.Timestamp('20150214'),
    'C': pd.Series(1.6, index=list(range(4)), dtype='float64'),
    'D': np.array([4] * 4, dtype='int64'),
    'E': 'hello pandas!'})


