# -*- coding: utf-8 -*-
"""
Created on Tue May 21 00:05:40 2024

@author: User
"""
from dateutil.relativedelta import relativedelta
from datetime import datetime
import numpy as np
import requests
import json
import time
import csv
import pandas as pd
from pandas import ExcelWriter
import xlsxwriter
from pandas_datareader import data as pdr
import yfinance as yf
from pandas_datareader._utils import RemoteDataError
from numpy import median#要引入這個才能跑中位數
from dateutil.relativedelta import relativedelta
from datetime import datetime
from pandas import ExcelWriter
import xlsxwriter
from pandas_datareader import data as pdr
from pandas_datareader._utils import RemoteDataError
from numpy import median#
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from tensorflow import keras
from tensorflow.keras import layers
from keras_self_attention import SeqSelfAttention
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
import keras
from keras_self_attention import SeqSelfAttention
import matplotlib.pyplot as plt
import tqdm #使用進度條
import tensorflow as tf
from random import sample
import os

##載入NP檔
x_bigdata=np.load(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/x_bigdata.npy')
y_bigdata=np.load(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/bigfinal_data/y_bigdata.npy')
##跑模型##
indexs=np.random.permutation(len(x_bigdata)) #隨機排序 49005以下的數字
train_indexs=indexs[:int(len(x_bigdata)*0.6)]
val_indexs=indexs[int(len(x_bigdata)*0.6):int(len(x_bigdata)*0.8)]
test_indexs=indexs[int(len(x_bigdata)*0.8):]
#x部分
x_bigdata_array=np.array(x_bigdata)
x_train=x_bigdata_array[train_indexs]
x_val=x_bigdata_array[val_indexs]
x_test=x_bigdata_array[test_indexs]
#y部分
y_scaler = MinMaxScaler(feature_range = (0, 1))
y_bigdata_array=np.array(y_bigdata)
y_train=y_bigdata_array[train_indexs]
y_val=y_bigdata_array[val_indexs]
y_test=y_bigdata_array[test_indexs]

##建立並訓練網路模型#############
from tensorflow import keras
from tensorflow.keras import layers
from keras_self_attention import SeqSelfAttention
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
import keras
from keras_self_attention import SeqSelfAttention
import matplotlib.pyplot as plt
import tqdm 
import tensorflow as tf
from keras.layers import Concatenate
from keras.layers import Conv1D , MaxPool2D , Flatten , Dropout,Conv2D
from keras.layers import GRU  #載入GRU

n = 10
n_steps = n 
n_features = 21
model = keras.Sequential(name='model-8')
model.add(LSTM(50,activation='tanh', return_sequences=True, input_shape = (n_steps, n_features)))
model.add(LSTM(50,activation='tanh'))
model.add(Dense(1))
model.summary()



model.compile(keras.optimizers.Adam(0.001),
              loss=keras.losses.MeanAbsoluteError(), 
              metrics=[keras.metrics.MeanAbsoluteError()])


model_dir = r'D:/2021 4月開始的找回程式之旅/lab2-logs/model8/'
#os.makedirs(model_dir)
log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/lab2-logs', 'model8')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')


#訓練網路模型：
history = model.fit(x_train, y_train,  # 傳入訓練數據
               batch_size=64,  # 批次大小設為64
               epochs=20,  # 整個dataset訓練300遍
               validation_data=(x_val, y_val),  # 驗證數據
               callbacks=[model_cbk, model_mckp])  # Tensorboard回調函數紀錄訓練過程，ModelCheckpoint回調函數儲存最好的模型


#訓練結果
history.history.keys()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('Mean square error')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
y_pred = model.predict(x_test)
# 顯示誤差到小數點第二位 #0.05
meanmae_error=np.mean(abs(y_test- np.array(y_pred)))

print(" 平均mae誤差: {:.2f}".format(meanmae_error))
    
    
    
    