from model import LotteryLSTMModel, LotteryNNModel
from data_helper import randomOpenCode
import keras
import numpy as np
# lottery_model = LotteryModel()
# random.random(size=(2,4))

seq_len = 6
d = 0.2
shape = [6, seq_len, 6]
neurons = [128, 128, 32, 6]
epochs = 10

X_train = randomOpenCode(100)
y_train = keras.utils.to_categorical(randomOpenCode(100), num_classes=33)
y_train = y_train[:100]

X_test = randomOpenCode(20)
y_test = keras.utils.to_categorical(randomOpenCode(20), num_classes=33)
y_test = y_test[:20]

# model = LotteryLSTMModel(shape, neurons, d)

model = LotteryNNModel()
model.fit(X_train, y_train, epochs=epochs, batch_size=33)

score = model.evaluate(X_test, y_test, batch_size=33)

print('score:', score)

val = np.array([[16, 0, 1, 30, 4, 32]])

print('val shape:', val.shape)

print('predict result:', model.predict(val))

