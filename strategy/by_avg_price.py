def initialize(context):
    # 写最开始要做什么的地方
    g.security = '002043.XSHE'  # 存入兔宝宝的股票代码


def handle_data(context, data):
    # 每天循环要来做什么
    last_price = data[g.security].close  # 获取最近日收盘价，命名为last_price
    # 获取近二十日股票收盘价的平均价格
    average_price = data[g.security].mavg(20, 'close')
    cash = context.portfolio.cash

    if last_price > average_price:
        order_value(g.security, cash)  # 用当前所有的前买入股票
    elif last_price < average_price:
        order_target(g.security, 0)  # 将股票仓位调整到0，即全部卖出
