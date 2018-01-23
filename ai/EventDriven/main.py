# from preprocess import query_news_from_nyt, query_stock_price, merging_data
from model import run_model, sentimentIntensity, processWithData, normalize_data, generate_hyperparameters, split_train_and_test_set
import nltk


# ### Process Data###

# query_news_from_nyt()
# interpolate_df = query_stock_price()
# merging_data(interpolate_df)

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
print('### split train and test data success:')

# ### Run model ###

run_model(
    {'xTrain': xTrain, 'yTrain': yTrain, 'xTest': xTest, 'yTest': yTest}, hp)
