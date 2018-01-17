from dataHelper import queryNewsFromNYTimes, queryStockPrice, mergingData

# Process with News and Stock Price Data

# queryNewsFromNYTimes()
interpolateDf = queryStockPrice()
mergingData(interpolateDf)

# Build Model
