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
from numpy import median
from dateutil.relativedelta import relativedelta
from datetime import datetime
from pandas import ExcelWriter
import xlsxwriter
from pandas_datareader import data as pdr
from pandas_datareader._utils import RemoteDataError
from numpy import median
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
import tqdm
import tensorflow as tf
from random import sample
import os




##載入NP檔
x_bigdata=np.load(r'C:\Users\2507\Desktop\遠端資料\data\x_bigdata.npy')
y_bigdata=np.load(r'C:\Users\2507\Desktop\遠端資料\data\y_bigdata.npy')


##設定步驟##
indexs=np.random.permutation(len(x_bigdata)) #隨機排序
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
#新
y_train=y_scaler.fit_transform(pd.DataFrame(y_train))
y_val=y_bigdata_array[val_indexs]
#新
y_val=y_scaler.fit_transform(pd.DataFrame(y_val))
y_test=y_bigdata_array[test_indexs]
#新
y_test_orign=y_test
y_test=y_scaler.fit_transform(pd.DataFrame(y_test))

##開始訓練##
#########建立並訓練網路模型#############
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
#import xgboost as xgb
from keras.layers import Concatenate
from keras.layers import Conv1D , MaxPool2D , Flatten , Dropout,Conv2D
from keras.layers import GRU  #載入GRU

###############套件
n = 5
n_steps = n 
n_features = 21
model = keras.Sequential(name='model-9')
model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape = (n_steps,n_features)))
model.add(keras.layers.Bidirectional(LSTM(10,activation='relu', return_sequences=True)))
#model.add(LSTM(10,activation='relu', return_sequences=True, input_shape = (n_steps, n_features)))  
model.add(SeqSelfAttention(attention_activation='tanh'))
model.add(Flatten())
#model.add(LSTM(50,activation='relu'))
model.add(Dense(1))


#自訂損失函數ustom_mean_squared_error
def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))
#設定訓練使用的優化器、損失函數和指標函數：
model.compile(keras.optimizers.Adam(0.001),
              loss=keras.losses.MeanAbsoluteError(),
              metrics=[keras.metrics.MeanAbsoluteError()])


#創建模型儲存目錄：
#在C:/Users/User/lab2-logs/models/建立模型目錄
model_dir = r'D:/2021 4月開始的找回程式之旅/lab2-logs/model9/'



log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/lab2-logs', 'model9')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')



history = model.fit(x_train, y_train, 
               batch_size=64, 
               epochs=30, 
               validation_data=(x_val, y_val),  
               callbacks=[model_cbk, model_mckp]) 


#訓練結果
history.history.keys()  # 查看history儲存的資訊有哪些
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.ylim(0.001, 0.006)
plt.title('Mean square error')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')


# 預測測試數據
y_pred = model.predict(x_test)
y_pred = y_scaler.inverse_transform(y_pred)

# 顯示誤差到小數點第二位 #0.05
meanmae_error=np.mean(abs(y_test- np.array(y_pred)))
print(" 平均mae誤差: {:.2f}".format(meanmae_error))
    
    
    
    