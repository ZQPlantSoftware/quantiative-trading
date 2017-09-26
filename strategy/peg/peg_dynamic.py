# 克隆自聚宽文章：https://www.joinquant.com/post/5395
# 标题：【量化课堂】PEG 动态选股策略
# 作者：JoinQuant量化课堂

from jqdata import *
import numpy as np
import talib
# import datetime
from numpy import mean
import traceback
import pandas as pd


def initialize(context):
    set_para()
    settings(context)
    get_all_price(context)


def before_trading_start(context):
    timer(context)
    trends_decision_signal(context)
    peg(context)
    blacklist(context)


def handle_data(context, data):
    if trends_decision_signal(context) == False:
        for security in g.inx:
            order_target_value(security, 0)
        order_target_value('000012.XSHG', context.portfolio.total_value)
    elif trends_decision_signal(context) == True:
        order_target_value('000012.XSHG', 0)
        stop(context)
        peg_ex(context)
    l = context.portfolio.positions.keys()
    log.info(l)


# setting parameters
def set_para():
    set_benchmark('000300.XSHG')
    g.lose = 0.92
    g.gain = 2
    g.i = 14
    g.blocking_days = 0
    g.number = 50
    g.ornum = 10
    g.days = 30
    g.current_price = 0
    g.MA = 0
    g.inx = 0


# 交易开始钱手续费滑点设置函数设置函数
def settings(context):
    # 将滑点和交易手续费设为0
    log.set_level('order', 'error')
    set_slippage(FixedSlippage(0))
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0, close_commission=0, min_commission=0),
                   type='stock')
    g.hp = {}
    g.b_l = {}
    g.list3 = []


# 取所有股票成交价格：
def get_all_price(context):
    index1 = get_index_stocks('000001.XSHG')
    index2 = get_index_stocks('399001.XSHE')
    index = index1 + index2
    log.info(len(index))
    for security in index:
        g.hp[security] = attribute_history(security, 1, '1d', 'close', df=False)['close'][-1]


# timer
def timer(context):
    g.i += 1


# 判断清仓函数
def trends_decision_signal(context):
    close_data1 = attribute_history('000001.XSHG', g.days, '1d', ['close'])
    close_data2 = attribute_history('399001.XSHE', g.days, '1d', ['close'])
    # 取得过n天的平均价格
    g.MA = 0.5 * (close_data1['close'].mean() + close_data2['close'].mean())
    # 取得上一时间点价格
    g.current_price = 0.5 * (close_data1['close'][-1] + close_data2['close'][-1])
    # 取得现有仓位中的所有股票
    g.inx = list(context.portfolio.positions.keys())
    if g.current_price < g.MA:
        g.i = 0
        return False
    elif g.i >= g.blocking_days:
        return True
    else:
        pass

        # 止损策略 如果价格低于三十天平均价 清仓


def decision_ex(context):
    if trends_decision_signal(context) == 0:
        for security in g.inx:
            order_target_value(security, 0)
        order_target_value('000012.XSHG', context.portfolio.total_value)
    if trends_decision_signal(context):
        order_target_value('000012.XSHG', 0)
        stop(context)
        peg_ex(context)


# 止盈止损与黑名单
def stop(context):
    s = context.portfolio.positions.keys()
    list4 = list(s)
    price = history(1, '1m', 'close', security_list=list4)
    for security in list4:
        price_now = price[security][-1]
        price_ji = context.portfolio.positions[security].avg_cost
        if security not in g.hp.keys():
            g.hp[security] = price_now
        elif price_now >= g.hp[security]:
            g.hp[security] = price_now
        elif price_now < g.lose * g.hp[security] or price_now > g.gain * price_ji:
            g.b_l[security] = 0
            order_target_value(security, 0)
        else:
            pass


# 黑名单剔除
def blacklist(context):
    list_q = list(g.b_l.keys())
    for security in list_q:
        g.b_l[security] += 1
        if g.b_l[security] > g.ornum:
            del g.b_l[security]


# 获取备选股票
def peg(context):
    # 取沪深300中的股票为备选股
    list0 = get_index_stocks('000001.XSHG')
    list2 = get_index_stocks('399001.XSHE')
    list1 = list0 + list2
    # 取出个股的净收益 总股数 和市盈率
    q1 = query(indicator.inc_net_profit_to_shareholders_year_on_year,
               income.code
               ).filter(income.code.in_(list1)
                        )
    q2 = query(indicator.inc_revenue_year_on_year,
               valuation.code
               ).filter(income.code.in_(list1)
                        )
    q3 = query(valuation.pe_ratio,
               valuation.code).filter(balance.code.in_(list1))
    a = pro_g = get_fundamentals(q1)
    b = reve_g = get_fundamentals(q2)

    # c = cap_cy = get_fundamentals(q2, date = str(ct_d))
    # d = cap_py = get_fundamentals(q2, statDate = str(py_y))

    c = pe_ratio = get_fundamentals(q3)

    a.index, b.index, c.index = a['code'], b['code'], c['code']
    # 取出所有盈利备选股
    a = a[a['inc_net_profit_to_shareholders_year_on_year'] > 0]
    b = b[b['inc_revenue_year_on_year'] > 0]
    c = c[c['pe_ratio'] > 0]

    # 从备选股中筛选出增长率大于10%的股票
    f = (c['pe_ratio'] / b['inc_revenue_year_on_year'])
    h = (c['pe_ratio'] / a['inc_net_profit_to_shareholders_year_on_year'])
    df1 = pd.DataFrame(f, columns=['peg_re'])
    df2 = pd.DataFrame(h, columns=['peg_pro'])
    df1 = df1.sort(columns=['peg_re'], ascending=True)
    df2 = df2.sort(columns=['peg_pro'], ascending=True)
    list1 = df1.index
    list2 = df2.index
    g.list3 = [x for x in list1[:g.number] if x in list2[:g.number]]


# 将资金分成若干等份并买入
def peg_ex(context):
    a_f = context.portfolio.available_cash
    na = len(g.list3)
    for security in g.list3[:g.ornum]:
        if security not in g.b_l.keys():
            order_value(security, a_f / g.ornum)
        else:
            pass

    a_f = context.portfolio.available_cash
    for security in g.list3:
        if security not in g.b_l.keys():
            if security not in context.portfolio.positions.keys():
                order_value(security, a_f)


