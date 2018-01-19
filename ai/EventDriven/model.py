import pandas as pd
import numpy as np
import tensorflow as tf
import unicodedata
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def processWithData():
    df_stocks = pd.read_pickle('/home/quantiative-trading/ai/EventDriven/data/pickled_ten_year_filtered_lead_para.pkl')

    df_stocks['prices'] = df_stocks['adj close'].apply(np.int64)

    # Selecting the prices and articles
    df_stocks = df_stocks[['prices', 'articles']]
    df_stocks.head()

    # Removing . or - from the begining
    df_stocks['articles'] = df_stocks['articles'].map(lambda x: x.lstrip('.-'))
    df_stocks.head()

    df = df_stocks[['prices']].copy()
    df.head()

    # Adding new columns to the data frame for text weight
    df['compound'] = ''
    df['neg'] = ''
    df['neu'] = ''
    df['pos'] = ''

    df.head()

    df_stocks.T

    return df, df_stocks

# unicodedata.normalize = Return the normal form for the Unicode string unistr.
def sentimentIntensity(df, df_stocks):
    sid = SentimentIntensityAnalyzer()
    for date, row in df_stocks.T.iteritems():
        try:
            sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles'])
            ss = sid.polarity_scores(sentence)
            df.set_value(date, 'compound', ss['compound'])
            df.set_value(date, 'neg', ss['neg'])
            df.set_value(date, 'neu', ss['neu'])
            df.set_value(date, 'pos', ss['pos'])

        except TypeError:
            print(df_stocks.loc[date, 'articles'])
            print(date)

    df.head()
    return df, df_stocks

# Gonna use numpy normalize to replace this formula soon
def normalize_data(df):
    datasetNorm = (df - df.mean()) / (df.max() - df.min())
    datasetNorm.reset_index(inplace=True)
    del datasetNorm['index']
    datasetNorm['next_prices'] = datasetNorm['prices'].shift(-1)
    datasetNorm.head(5)

    return datasetNorm

def split_train_and_test_set(datasetNorm, hp):
    datasetTrain = datasetNorm[datasetNorm.index < hp.num_batches * hp.batch_size * hp.truncated_backprop_length]

    for i in range(hp.min_test_size, len(datasetNorm.index)):
        if(i % hp.truncated_backprop_length * hp.batch_size == 0):
            test_first_idx = len(datasetNorm.index) - i
            break

    datasetTest = datasetNorm[datasetNorm.index >= test_first_idx]

    xTrain = datasetTrain[['prices', 'neu', 'neg', 'pos']].as_matrix()
    yTrain = datasetTrain['next_prices'].as_matrix()

    xTest = datasetTest[['prices', 'neu', 'neg', 'pos']].as_matrix()
    yTest = datasetTest['next_prices'].as_matrix()

    return xTrain, yTrain, xTest, yTest

def generate_hyperparameters(total_length):
    # hyperparameters
    batch_size = 1
    total_series_length = total_length
    truncated_backprop_length = 3  # The size of the sequence
    state_size = 12  # The number of neurons

    num_epochs = 1000
    num_features = 4
    num_classes = 1
    num_batches = total_series_length / batch_size / truncated_backprop_length

    min_test_size = 100

    print('The total series length is: %d' % total_series_length)
    print('The current configuration gives us %d batches of %d observations each one looking %d steps in the past'
          % (num_batches, batch_size, truncated_backprop_length))

    return {
        batch_size,
        total_series_length,
        truncated_backprop_length,
        state_size,
        num_epochs,
        num_features,
        num_classes,
        num_batches,
        min_test_size
    }

def generate_placeholder(hp):
    batchX_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, hp.truncated_backprop_length, hp.num_features], name='data_ph')
    batchY_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, hp.truncated_backprop_length, hp.num_classes], name='target_ph')

    # Weights and biases Because its 3 layer net INPUT: Hidden Recurrent Layer OUTPUT: We need 3 pair of W and b
    W2 = tf.Variable(initial_value=np.random.rand(hp.state_size, hp.num_classes), dtype=tf.float32)
    b2 = tf.Variable(initial_value=np.random.rand(1, hp.num_classes), dtype = tf.float32)

    return batchX_placeholder,  batchY_placeholder, W2, b2

def generate_model(hp):
    tf.reset_default_graph()

    batchX_placeholder, batchY_placeholder, W2, b2 = generate_placeholder()

    # Unpack
    labels_series = tf.unstack(batchY_placeholder, axis=1)

    # Forward pass - unroll the cell
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hp.state_size)
    states_series, current_state = tf.nn.dynamic_rnn(cell=cell, inputs=batchX_placeholder, dtype=tf.float32)
    state_series = tf.transpose(states_series, [1, 0, 2])

    # Backward pass - Output
    last_state = tf.gather(params=states_series, indices=states_series.get_shape()[0] - 1)
    last_label = tf.gather(params=labels_series, indices=len(labels_series) - 1)

    weight = tf.Variable(tf.truncated_normal([hp.state_size, hp.num_classes]))
    bias = tf.Variable(tf.constant(0.1, shape=[hp.num_classes]))

    prediction = tf.matmul(last_state, weight) + bias

    loss = tf.reduce_mean(tf.squared_difference(last_label, prediction))
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    return {
        loss,
        train_step,
        prediction,
        last_label,
        last_state,
        prediction,
        batchX_placeholder,
        batchY_placeholder
    }

def run_model(parameter, hp):
    loss_list = []
    test_pred_list = []

    xTrain, yTrain, xTest, yTest = parameter
    parameters = generate_model(hp)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for epoch_idx in range(hp.num_epochs):
            print('Epoch %d' % epoch_idx)

            for batch_idx in range(hp.num_batches):
                start_idx = batch_idx * hp.truncated_backprop_length
                end_idx = start_idx + hp.truncated_backprop_length

                batchX = xTrain[start_idx:end_idx, :].reshape(hp.batch_size, hp.truncated_backprop_length, hp.num_features)
                batchY = yTrain[start_idx:end_idx].reshape(hp.batch_size, hp.truncated_backprop_length, 1)

                feed = {parameters.batchX_placeholder: batchX, parameters.batchY_placeholder: batchY}

                # Train
                _loss, _train_step, _pred, _last_label, _prediction = sess.run(
                    fetches=[parameters.loss, parameters.train_step, parameters.prediction, parameters.last_label, parameters.prediction],
                    feed_dict=feed
                )

                loss_list.append(_loss)

                if(batch_idx % 50 == 0):
                    print('Step %d - Loss: %.6f' % (batch_idx, _loss))

        for test_idx in range(len(xTest) - hp.truncated_backprop_length):
            testBatchX = xTest[test_idx:test_idx + hp.truncated_backprop_length, :].reshape((1, hp.truncated_backprop_length, hp.num_features))
            testBatchY = yTest[test_idx:test_idx + hp.truncated_backprop_length].reshape((1, parameters.truncated_backprop_length, 1))

            feed = {
                parameters.batchX_placeholder: testBatchX,
                parameters.batchY_placeholder: testBatchY
            }

            _last_state, _last_label, test_pred = sess.run([parameters.last_state, parameters.last_label, parameters.prediction], feed_dict=feed)
            test_pred_list.append(test_pred[-1][0])