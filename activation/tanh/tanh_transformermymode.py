# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:30:11 2024

@author: 2507
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
#重要要引進RemoteDataError才能跑
from pandas_datareader._utils import RemoteDataError
from numpy import median#要引入這個才能跑中位數
from dateutil.relativedelta import relativedelta
from datetime import datetime

from pandas import ExcelWriter
import xlsxwriter
from pandas_datareader import data as pdr
#重要要引進RemoteDataError才能跑
from pandas_datareader._utils import RemoteDataError
from numpy import median#要引入這個才能跑中位數

############################跑lstm
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

###########神經元層
from tensorflow import keras
from tensorflow.keras import layers
from keras_self_attention import SeqSelfAttention

# Adding the LSTM layer
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
from tensorflow.keras.utils import plot_model
from IPython.display import Image
from keras import initializers
#attention
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from keras import initializers
import tensorflow as tf


#x_bigdata=np.load(r'C:\Users\2507\Desktop\遠端資料\data\x_bigdata.npy')
#y_bigdata=np.load(r'C:\Users\2507\Desktop\遠端資料\data\y_bigdata.npy')

x_bigdata=np.load(r'D:\2021 4月開始的找回程式之旅\0409論文想做的題目\bigfinal_data\x_bigdata.npy')
y_bigdata=np.load(r'D:\2021 4月開始的找回程式之旅\0409論文想做的題目\bigfinal_data\y_bigdata.npy')


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


##開始建構transformer模型
#Build the model
from tensorflow import keras
from tensorflow.keras import layers


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    
    # Normalization and Attention
    # "EMBEDDING LAYER"
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    
    # "ATTENTION LAYER"
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    
    # FEED FORWARD Part - you can stick anything here or just delete the whole section - it will still work. 
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    
    ##這邊做兩個分支
    #x  = layers.Conv1D(filters=4,kernel_size=1,padding='causal', dilation_rate=2,activation='relu')(x)
    x_1 = layers.Conv1D(filters=4,kernel_size=1,padding='causal', dilation_rate=2,activation='tanh')(x)
    x_2 = layers.Conv1D(filters=4,kernel_size=1,padding='causal', dilation_rate=4,activation='tanh')(x)
    #兩個分支結合
    x = layers.Concatenate()([x_1, x_2])
    
    #兩個分支結束
    
    
    #x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation = "relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res




def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    for _ in range(num_transformer_blocks):  # This is what stacks our transformer blocks
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    x_encoder1=x
    for dim in mlp_units:
        x_encoder=x
        #x =layers.Conv1D(filters=4,kernel_size=1,padding='causal', dilation_rate=2,activation='relu')(x)
        #x= layers.BatchNormalization()(x)
        #x = layers.Dropout(mlp_dropout)(x)
        #x = layers.Dense(dim, activation="elu")(x)
        
        #新增加
        x = layers.Dense(5, activation="tanh")(x)
        x = layers.Dropout(mlp_dropout)(x)
        x = layers.Dense(5, activation="tanh")(x)
        x = layers.Dropout(mlp_dropout)(x)
        x=x_encoder+x
        
    ##加入resnet
    #x=x_encoder+x
    x=layers.Concatenate()([x_encoder1, x])
    
    
    #新加入的casual  
    #x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    #x = layers.Dense(10, activation="elu")(x)
    #x = layers.Dropout(mlp_dropout)(x)
    #新加入的casual 結束 
    
    outputs = layers.Dense(1, activation="tanh")(x) #this is a pass-through
    return keras.Model(inputs, outputs)
    


#設定超參數
input_shape = x_train.shape[1:]

print(input_shape)
# epoch_number=2
epoch_number=20
# epoch_number = 200
# batch_size=4
batch_size=64
# batch_size=64


model = build_model(
    input_shape,
    head_size=256, #256
    num_heads=4,  #4
    ff_dim=4,
    num_transformer_blocks=9,  #IBM用2 validation會比較變動  4
    #mlp_units=[128],
    mlp_units=range(0,3), #mlp_units=range(0,7),
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
#model_dir = r'C:\Users\2507\Desktop\遠端資料\save_best'
model_dir = r'D:/2021 4月開始的找回程式之旅/parameter/save_best'

log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/lab2-logs', 'model10')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
# ModelCheckpoint回調函數幫忙儲存網路模型，可以設定只儲存最好的模型，「monitor」表示被監測的數據，「mode」min則代表監測數據越小越好。
#將模型儲存在C:/Users/User/lab2-logs/models/
model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')



def scheduler(epoch, lr):
    if epoch < 0:
        return lr
    else:
        return lr * tf.exp(-0.1, name ='exp')



history = model.fit(x_train, y_train,  # 傳入訓練數據
               batch_size=32,  # 批次大小設為64
               epochs=30,  # 整個dataset訓練100遍
               validation_data=(x_val, y_val),  # 驗證數據
               callbacks=[model_cbk, model_mckp,keras.callbacks.LearningRateScheduler(scheduler)])
               #callbacks = [
                   #keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                   #keras.callbacks.LearningRateScheduler(scheduler)
                   #]
               #)  # Tensorboard回調函數紀錄訓練過程，ModelCheckpoint回調函數儲存最好的模型
               

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