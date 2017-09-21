import pandas as pd

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
    set_params()
    set_variables()
    set_backtest()

# 1
# 设置策略参数
def set_params():
    g.tc = 15               # 调仓天数
    g.num_stocks = 10       # 每次调仓选取的最大股票数量

# 2
# 设置中间变量
def set_variables():
    g.t = 0                 # 记录回测运行的天数
    g.if_trade = False      # 当天是否交易

# 3
# 设置回测条件
def set_backtest():
    set_option('use_real_price', True)      # 用真实价格交易
    log.set_level('order', 'error')         # 设置报错等级


'''
=====================================================
    每天开盘前
=====================================================
'''

# 开盘前准备工作
def before_trading_start(context):
    if g.t% g.tc == 0:
        g.if_trade = True           # 每 g.tc 天，调仓一次
        set_slip_fee(context)       # 设置手续费
        g.stocks = get_index_stocks('000300.XSHG') # 设置沪深300为初始股票池
        g.feasible_stocks = set_feasible_stocks(g.stocks, context) # 设置可行股票池

    g.t += 1

# 4
# 设置可行股票池：过滤当日停牌的股票
# 输入：initial_stocks 为 list 类型，表示初始股票池
# 输出：unsuspened_stocks 为list类型，表示当日未停牌的股票池，即可行股票池
def set_feasible_stocks(initial_stocks, context):
    paused_info = []
    current_data = get_current_data()
    for i in initial_stocks:
        paused_info.append(current_data[i].paused)
    df_paused_info = pd.DataFrame({'paused_info': paused_info}, index = initial_stocks)
    unsupened_stocks = list(df_paused_info.index[df_paused_info.paused_info == False])
    return unsupened_stocks

# 5
# 根据不同的时间段设置滑点与手续费
# 输入：context
# 输出：none
def set_slip_fee(context):
    # 将滑点设置为0
    set_slippage(FixedSlippage(0))
    # 根据不同时间段设置手续费
    dt = context.current_dt
    if dt > datetime.datetime(2013, 1, 1):
        set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))

    elif dt > datetime.datetime(2011, 1, 1):
        set_commission(PerTrade(buy_cost=0.001, sell_cost=0.002, min_cost=5))

    elif dt > datetime.datetime(2009, 1, 1):
        set_commission(PerTrade(buy_cost=0.002, sell_cost=0.003, min_cost=5))
    else:
        set_commission(PerTrade(buy_cost=0.003, sell_cost=0.004, min_cost=5))


'''
=====================================================
    每天交易时
=====================================================
'''

# 每天回测做的事
def handle_data(context, data):
    if g.if_trade == True:
        # 待买入的 g.num_stocks 支股票，list类型
        list_to_buy = stocks_to_buy(context)
        # 待卖出的股票，list类型
        list_to_sell = stocks_to_sell(context, list_to_buy)
        # 卖出操作
        sell_operation(list_to_sell)
        # 买入操作
        buy_operation(context, list_to_buy)

    g.if_trade = False

# 6
# 计算股票的PEG值
# 输入：context；stock_list 为 list类型，表示股票池
# 输出：df_PEG 为 dataframe：index为股票代码， data 为相应的 PEG 值
def get_PEG(context, stock_list):
    # 查询过池里股票的市盈率，收益增长率
    q_PE_G = query(valuation.code, valuation.pe_ratio, indicator.inc_net_profit_year_on_year).filter(valuation.code.in_(stock_list))
    # 得到一个 DataFrame：包含股票代码 市盈率PE 收益增长率G
    # 默认 date = context.current_dt 的前一天，使用默认值，避免未来函数
    df_PE_G = get_fundamentals(q_PE_G)
    # 筛选出成长股：删除市盈率或收益增长率为负值的股票
    df_Growth_PE_G = df_PE_G[(df_PE_G.pe_ratio > 0) & (df_PE_G.inc_net_profit_year_on_year > 0)]
    # 去除 PE 为负数或者非数字
    df_Growth_PE_G.dropna()
    # 得到一个 series：存放股票的市盈率 TTM 即 PE 值
    Series_PE = df_Growth_PE_G.ix[:, 'pe_ratio']
    # 得到一个Series：存放股票的收益增长率，即 G 值
    Series_G = df_Growth_PE_G.ix[:, 'inc_net_profit_year_on_year']
    # 得到一个Series： 存放股票的PEG值
    Series_PEG = Series_PE / Series_G
    # 将股票的 PEG 值与其股票对应
    Series_PEG.index = df_Growth_PE_G.ix[:, 0]
    # 将 Series 类型转换成 DataFrame 类型
    df_PEG = pd.DataFrame(Series_PEG)
    return df_PEG

# 7
# 获得买入信号
# 输入：context
# 输出：list_to_buy 为 list 类型，表示待买入的 g.num_stocks 支股票
def stocks_to_buy(context):
    list_to_buy = []
    # 得到一个 DataFrame：index为股票代码，data为相应的PEG值
    df_PEG = get_PEG(context, g.feasible_stocks)
    # 将股票按PEG升序排列，返回 DataFrame 类型
    df_sort_PEG = df_PEG.sort(columns=[0], ascending=[1])
    # 将储存有股票代码 index 转化成 list 并取前 g.num_stocks 个为待买入股票，返回list
    for i in range(g.num_stocks):
        if df_sort_PEG.ix[i, 0] < 0.5:
            list_to_buy.append(df_sort_PEG.index[i])
    return list_to_buy

# 8
# 获得卖出信号
# 输入：context；list_to_buy 为 list 类型，代办待买入股票
# 输出：list_to_sell 为 list 类型，表示待卖出股票
def stocks_to_sell(context, list_to_buy):
    list_to_sell = []
    # 对于不需要持仓的股票，全仓卖出
    for stock_sell in context.portfolio.positions:
        if stock_sell not in list_to_buy:
            list_to_sell.append(stock_sell)
    return list_to_sell

# 9
# 执行卖出操作
# 输入：list_to_sell 为 list 类型，表示待卖出的股票
# 输出：none
def sell_operation(list_to_sell):
    for stock_sell in list_to_sell:
        order_target_value(stock_sell, 0)

# 10
# 执行买入操作
# 输入：context；list_to_buy 为 list 类型，表示待买入的股票
# 输出：none
def buy_operation(context, list_to_buy):
    for stock_sell in list_to_buy:
        # 为买个持仓分配资金
        g.capital_unit = context.portfolio.portfolio_value / len(list_to_buy)
        # 买入在 "待买股票列表" 中的股票
        for stock_buy in list_to_buy:
            order_target_value(stock_buy, g.capital_unit)

'''
================================================================================
每天收盘后
================================================================================
'''
# 每天收盘后做的事情
# 进行长运算（本策略中不需要）
def after_trading_end(context):
    return