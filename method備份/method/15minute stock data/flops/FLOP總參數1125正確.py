# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:10:50 2024

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

#y_bigdata位置標瑪test_indexs_new



##設定步驟##
indexs=np.random.permutation(len(x_bigdata)) #隨機排序
#indexs=np.arange(len(y_bigdata))
train_indexs=indexs[:int(len(x_bigdata)*0.6)]
val_indexs=indexs[int(len(x_bigdata)*0.6):int(len(x_bigdata)*0.8)]
test_indexs=indexs[int(len(x_bigdata)*0.8):]
#新的排序
test_indexs_new=test_indexs+1

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
    x_1 = layers.Conv1D(filters=4,kernel_size=1,padding='causal', dilation_rate=2,activation='relu')(x)
    x_2 = layers.Conv1D(filters=4,kernel_size=1,padding='causal', dilation_rate=4,activation='relu')(x)
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
        x = layers.Dense(10, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        x = layers.Dense(10, activation="relu")(x)
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
    
    outputs = layers.Dense(1)(x) #this is a pass-through
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
    num_transformer_blocks=2,  #IBM用2 validation會比較變動  4
    #mlp_units=[128],
    mlp_units=range(0,2), #mlp_units=range(0,7),
    mlp_dropout=0.25,
    dropout=0.25,
)




#設定訓練使用的優化器、損失函數和指標函數：
model.compile(keras.optimizers.Adam(0.001),
              loss=keras.losses.MeanSquaredError(),  #loss=keras.losses.MeanSquaredError()
              #loss=custom_loss, 
              metrics=[keras.metrics.MeanAbsoluteError()])


model.summary()



def get_flops(model):
  tf.compat.v1.disable_eager_execution()  #關閉eager狀態
  sess = tf.compat.v1.Session()#自動轉換腳本

  run_meta = tf.compat.v1.RunMetadata()
  profiler = tf.compat.v1.profiler
  #opts = tf.profiler.ProfileOptionBuilder.float_operation()
  opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
  # We use the Keras session graph in the call to the profiler.
  flops = profiler.profile(graph=sess.graph, 
                           run_meta=run_meta, cmd='op', options=opts)

  return flops.total_float_ops  # Prints the "flops" of the model


get_flops(model)  #1455234115