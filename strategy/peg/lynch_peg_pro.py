import pandas as pd
import numpy as np
import scipy as sp
import scipy.optimize
import datetime as dt
from scipy import linalg as sla
from scipy import spatial
from jqdata import gta
from strategy.lib.peg.low_peg as logPegAgo
'''
=====================================================
    Global variable (define by JoinQuant)
=====================================================
'''

g = {}
log = {}
datetime = {}
valuation = {}
indicator = {}
def set_option():
    pass
def set_slip_fee():
    pass
def get_index_stocks():
    pass
def get_current_data():
    pass
def set_commission():
    pass
def set_slippage():
    pass
def query():
    pass
def order_target_value():
    pass
def get_fundamentals():
    pass
def PerTrade():
    pass
def FixedSlippage():
    pass

'''
=====================================================
    回测前准备
=====================================================
'''

# 初始化 - 回测前准备工作
def initialize(context):
    # 用沪深 300 做回报基准
    set_benchmark('000300.XSHG')
    set_slippage(FixedSlippage(0.002))
    set_option('use_real_price', True)

    # 关闭部分log
    log.set_level('order', 'error')
    # 定义策略占用仓位比例
    context.lowPEG_ratio = 1.0

    # for lowPEG algorithms
    # 正态分布概率表，标准差倍数以及置信率
    # 1.96, 95%; 2.06, 96%; 2.18, 97%; 2.34, 98%; 2.58, 99%; 5, 99.9999%
    context.lowPEG_confidencelevel = 1.96
    context.lowPEG_hold_periods, context.lowPEG_hold_cycle = 0, 30
    context.lowPEG_stock_list = []
    context.lowPEG_position_price = {}

    g.quantlib = quantlib()

    run_daily(fun_main, '10:30')


def fun_main(context):
    lowPEG_trade_ratio = lowPEG_algo(context, context.lowPEG_ratio, context.portfolio.portfolio_value)
    # 调仓，执行交易
    g.quantlib.fun_do_trade(context, lowPEG_trade_ratio, context.lowPEG_moneyfund)

