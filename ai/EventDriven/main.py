# from dataHelper import queryNewsFromNYTimes, queryStockPrice, mergingData
from model import run_model, sentimentIntensity, processWithData, normalize_data, generate_hyperparameters, split_train_and_test_set
import nltk


# ### Process Data###

# queryNewsFromNYTimes()
# interpolateDf = queryStockPrice()
# mergingData(interpolateDf)

# ### IF FIRST TIME USE NLTK ###
# nltk.download()

# ### Build Model ###

print('### Build Model ###')
df, df_stocks = processWithData()
df, df_stocks = sentimentIntensity(df, df_stocks)
print('### read data and sentiment intensity success:', df.head())

datasetNorm = normalize_data(df)
print('### normalize data success:', datasetNorm.head())

hp = generate_hyperparameters(len(datasetNorm.index))
xTrain, yTrain, xTest, yTest = split_train_and_test_set(datasetNorm, hp)
# print('### split train and test data success:', xTrain.head(), yTrain.head(), xTest.head(), yTest.head())

# ### Run model ###

run_model(
    {xTrain: xTrain, yTrain: yTrain, xTest: xTest, yTest: yTest}, hp)
