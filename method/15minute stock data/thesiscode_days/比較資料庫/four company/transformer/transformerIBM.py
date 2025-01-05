# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 23:36:51 2024

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
#重要要引進RemoteDataError才能跑
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
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from keras import initializers
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from keras import initializers
import tensorflow as tf
#layer
from keras import layers
from keras import activations





IBM= yf.download("IBM", start="1980-01-01", end="2024-07-31")


#IBM= pd.read_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/0506比較方法/新資料比較方法/我的模型套用不同資料集ablilationstudy/IBM資料集/data/IBM.csv', encoding='utf_8_sig')

#final_data_real=[]
#final_data_real.append(IBM)
#stock_id=['IBM']
#stock_mae=[] 
#stock=[] 
#final_data_real_copy=final_data_real

x_scaler = MinMaxScaler(feature_range = (0, 1))
y_scaler = MinMaxScaler(feature_range = (0, 1))


#開始跑資料
tsla_data =IBM
tsla_data.columns
#tsla_data.set_index(['Date'], inplace=True)
tsla_close = tsla_data['Close'].values
tsla_data=tsla_data.drop('Adj Close', axis=1)
n = 10
train =tsla_data[:int(len(tsla_data) *0.6)]
val =tsla_data[:int(len(tsla_data) *0.8)]
test =tsla_data[int(len(tsla_data) *0.8):]
y_testc=test['Close'][n:]
feature_names = list(train.drop('Close', axis=1).columns)
x_train = []
y_train = []
train_indexes = []
norm_data_xtrain = train[feature_names]
for i in tqdm.tqdm_notebook(range(0,len(train)-n)):    
    x_trainadd=norm_data_xtrain.iloc[i:i+n]. values
    x_trainaddscalar=x_scaler.fit_transform(x_trainadd)
    x_train.append(np.transpose(x_trainaddscalar))  
    y_train.append(train['Close'].iloc[i+n]) 
    train_indexes.append(train.index[i+n]) 
print(x_train[0])
    
x_train=np.array(x_train)
y_train_dataframe=pd.DataFrame(y_train).iloc[:len(y_train)]
y_train_tran=y_scaler.fit_transform(y_train_dataframe)
y_train=np.array(y_train_tran).reshape(-1)
#y_train=np.array(y_train_tran)
print(x_train.shape)
    
x_val = []
y_val = []
val_indexes = []
norm_data_xval = val[feature_names]
for i in tqdm.tqdm_notebook(range(0,len(val)-n)):      
    x_valadd=norm_data_xval.iloc[i:i+n]. values
    x_valaddscalar=x_scaler.fit_transform(x_valadd)
    x_val.append(np.transpose(x_valaddscalar))  
    y_val.append(val['Close'].iloc[i+n]) 
    val_indexes.append(val.index[i+n]) 
print(x_val[0])
    
x_val=np.array(x_val)
y_val_dataframe=pd.DataFrame(y_val).iloc[:len(y_val)]
y_val_val=y_scaler.fit_transform(y_val_dataframe)
y_val=np.array(y_val_val).reshape(-1)
#y_val=np.array(y_val_val)
print(x_val.shape)    
      
x_test = []
y_test = []
test_indexes = []
    
norm_data_xtest = test[feature_names]
for i in tqdm.tqdm_notebook(range(0,len(test)-n)): 
    x_testadd=norm_data_xtest.iloc[i:i+n]. values
    x_testaddscalar=x_scaler.fit_transform(x_testadd)
    x_test.append(np.transpose(x_testaddscalar)) 
    y_test.append(test['Close'].iloc[i+n]) 
    test_indexes.append(test.index[i+n]) 

x_test1=x_test
y_test1=y_test
x_test=np.array(x_test)
y_test_dataframe=pd.DataFrame(y_test).iloc[:len(y_test)]
#y_test備份
y_test_orign=np.array(y_test_dataframe)
y_test_tran=y_scaler.fit_transform(y_test_dataframe)
y_test=np.array(y_test_tran).reshape(-1) 
#y_test=np.array(y_test_tran)

#開始跑模型
from tensorflow import keras
from tensorflow.keras import layers

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    #multihead多頭
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    
    #多加的selfattention
    #x= SeqSelfAttention(attention_activation='sigmoid')(x) #sigmoid
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


#建立模型顯示出要用幾次
def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks, #執行三次encoder block
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    #執行3次transformerblock
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    #x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    #接3次resnet decoder部分
    #這邊省略
    #將它變成一維
    x=layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    outputs = layers.Dense(1, activation="linear")(x)  
    return keras.Model(inputs, outputs)
    #resnet結束
    



#設定超參數
input_shape = x_train.shape[1:]
# epoch_number=2
epoch_number=10
# epoch_number = 200
# batch_size=4
batch_size=64
# batch_size=64



model = build_model(
    input_shape,
    head_size=256,
    num_heads=1, #單一頭
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=range(3),#接多少次resnet   [3]
    mlp_dropout=0.4,
    dropout=0.25,
)


#設定訓練使用的優化器、損失函數和指標函數：
model.compile(keras.optimizers.Adam(0.001),
              loss=keras.losses.MeanSquaredError(),  #loss=keras.losses.MeanSquaredError()
              #loss=custom_loss, 
              metrics=[keras.metrics.MeanAbsoluteError()])



model.summary()


#設定callback
model_dir = r'D:/2021 4月開始的找回程式之旅/parameter/save_best'

log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/parameter/lab2-logs', 'model10')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
# ModelCheckpoint回調函數幫忙儲存網路模型，可以設定只儲存最好的模型，「monitor」表示被監測的數據，「mode」min則代表監測數據越小越好。
#將模型儲存在C:/Users/User/lab2-logs/models/
model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')



history = model.fit(x_train, y_train,  # 傳入訓練數據
               batch_size=64,  # 批次大小設為64
               epochs=30,  # 整個dataset訓練100遍
               validation_data=(x_val, y_val),  # 驗證數據
               callbacks=[model_cbk, model_mckp])  # Tensorboard回調函數紀錄訓練過程，ModelCheckpoint回調函數儲存最好的模型
               #callbacks=[model_cbk, model_mckp,lrdecay])
      
               

##預測結果
#訓練結果
history.history.keys()  # 查看history儲存的資訊有哪些

#在model.compile已經將損失函數設為均方誤差(Mean Square Error)
#所以history紀錄的loss和val_loss為Mean Squraed Error損失函數計算的損失值
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
#plt.ylim(0.003, 0.006)
plt.title('loss function')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
##MAE評估指標
plt.plot(history.history['mean_absolute_error'], label='train')
plt.plot(history.history['val_mean_absolute_error'], label='validation')
plt.title('mae')
plt.ylabel('mae')
plt.xlabel('epochs')
plt.legend(loc='upper right')
#plt.savefig(path+'loss.png')
#清空圖片
#plt.cla()



# 預測測試數據
y_pred = model.predict(x_test)
# 顯示誤差到小數點第二位 #0.05
#epooch20次0.04297471135761948
y_pred = y_scaler.inverse_transform(y_pred)
meanmae_error=np.mean(abs(y_pred- np.array(y_test_orign)))
print(" 平均mae誤差: {:.2f}".format(meanmae_error))
#寫入txt檔
#pathtxt =path+ 'output.txt'
#f = open(pathtxt,'w')
#print(" 平均mae誤差:"+str(round(meanmae_error,2)), file=f)
#big_lstm_data.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/0506比較方法/五個股票/MINMAXSCALAR 3060five_outputdata_lstm.csv', encoding='utf_8_sig')

