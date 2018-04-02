from predict import do_predict
import time
from sys import stdin

print("<<<< DuoCloud Crypto Currency Market Predict >>>>")
print("* Use LSTM Neural Network to Predict BTC and ETH Tomorrow Price *")
print("Hello, and Welcome Use!")
print('First, did you want train model or load weights?(y or anything)')
need_train = stdin.readline() == 'y'
need_save = True

if need_train:
    print("Training Begin date (yyyy-MM-dd):")
    begin_date = stdin.readline() or '20150101'

    print("Training End date (default now):")
    end_date = stdin.readline() or time.strftime("%Y%m%d")

    print("Did you want save parameters and weight?")
    need_save = stdin.readline() == 'y'

print("===== Preparing to Predict =====")

do_predict(need_train, need_save)
