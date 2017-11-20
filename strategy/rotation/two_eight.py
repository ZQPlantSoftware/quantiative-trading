g = {}

# ######################### Helper Function ##########################

def attribute_history(stock, intervalm, unit='1d', fields=('close'), skip_paused=True):
    # TODO Get the history data
    print 1

def order_target():
    print 1

def order_value():
    print 1

# ####################### Helper Function End ########################


def initialize(context):
    g.cntLost = 0
    g.cntProfit = 0
    g.lost = 0
    g.profit = 0

def getStockPrice(stock, interval):
    h = attribute_history(stock, interval, unit='1d', fields=('close'), skip_paused=True)
    return h['close'].values[0]

def calPosition(context):
    return context.portfolio.cash

def calProfitLost(context, stock, price):
    if (context.portfolio.positions[stock].avg_cost < price):
        g.cntProfit = g.cntProfit + 1
        g.profit = g.profit + (price - context.portfolio.positions[stock].avg_cost) * context.portfolio.positions[stock].amount
    else:
        g.cntLost = g.cntLost + 1
        g.profit = g.profit + (price - context.portfolio.positions[stock]/avg_cost) * context.portfolio.positions[stock].amount

def handle_data(context, data):
    interval = 21
    zs2 = ''
    zs8 = ''
    etf2 = ''
    etf8 = ''
    hs = getStockPrice(zs2, interval)
    zz = getStockPrice(zs8, interval)

    cp300 = data[zs2].close
    cp500 = data[zs8].close

    hsIncrease = (cp300 - hs) / hs
    zzIncrease = (cp500 - zz) / zz

    positions = context.portfolio.positions

    if hsIncrease > 0 or zzIncrease > 0:
        if hsIncrease > zzIncrease:
            if positions[etf8].amount > 0:
                calProfitLost(context, etf8, data[etf8].close)
                order_target(etf8, 0)
            if positions[etf2].amount == 0:
                order_value(etf2, calPosition(context))

        else:
            if positions[etf2].amount > 0:
                calProfitLost(context, etf2, data[etf2].close)
                order_target(etf2, 0)

            if positions[etf8].amount == 0:
                order_value(etf8, calPosition(context))
        return

    if positions[etf8].amount > 0:
        calProfitLost(context, etf8, data[etf8].close)
        order_target(etf8, 0)

    if positions[etf2].amount > 0:
        calProfitLost(context, etf2, data[etf2].close)
        order_target(etf2, 0)