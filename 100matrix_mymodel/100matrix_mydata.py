# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:30:11 2024

@author: 2507
"""

import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
import os
import keras
from keras.layers import Flatten
import yfinance as yf
from tensorflow.keras import layers
from keras_self_attention import SeqSelfAttention
from tensorflow.python.framework import ops
from sklearn.metrics import mean_absolute_error



#load data
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


##開始建構transformer模型
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    
    # "ATTENTION LAYER"
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    
    # Normalization    
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    #Branch
    x_1 = layers.Conv1D(filters=4,kernel_size=2,padding='causal', dilation_rate=2,activation='relu')(x)
    x_2 = layers.Conv1D(filters=4,kernel_size=2,padding='causal', dilation_rate=4,activation='relu')(x)
    x = layers.Concatenate()([x_1, x_2])
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=2,padding='same')(x)
    return x + res





def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_encoderblocks,
    num_transformer_decoderblocks,
    #mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    for _ in range(num_transformer_encoderblocks):  # This is what stacks our transformer blocks
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x) 
    x_encoder1=x 
    #decoder
    for dim in num_transformer_decoderblocks:
        x_encoder=x
        x = layers.Dense(10, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        x = layers.Dense(10, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        x=x_encoder+x
    x=layers.Concatenate()([x_encoder1, x])
    outputs = layers.Dense(1)(x) 
    return keras.Model(inputs, outputs)
    

#Set hyperparameters
input_shape = x_train.shape[1:]
print(input_shape)
epoch_number=30
batch_size=32



#make matrix
bigmae=np.zeros(shape=(10,10))


def scheduler(epoch, lr):
    if epoch < 0:
        return lr
    else:
        return lr * tf.exp(-0.1, name ='exp')


model_dir = r'C:\Users\2507\Desktop\遠端資料\save_best'
log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/lab2-logs', 'model10')

for i in range(0,10):
    for j in range(0,10):
        print("****i為第"+str(i)+"筆資料***")
        print("****j為第"+str(j)+"筆資料***")
        
        model = build_model(          
            input_shape,
            head_size=64, #256
            num_heads=2,  #4
            ff_dim=4,
            num_transformer_encoderblocks=i,  #IBM用2 validation會比較變動  4
            num_transformer_decoderblocks=range(0,j), #mlp_units=range(0,5),
            mlp_dropout=0.25,
            dropout=0.25)
        
    
        model.compile(keras.optimizers.Adam(0.001),
                     loss=keras.losses.MeanSquaredError(),  #loss=keras.losses.MeanSquaredError()
                     metrics=[keras.metrics.MeanAbsoluteError()])


        model.summary()

        model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
        model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5',                                                   
                                                    monitor='val_mean_absolute_error', 
                                                    save_best_only=True, 
                                                    mode='min')



        history = model.fit(x_train, y_train,         
                            batch_size=32, 
                            epochs=10, 
                            validation_data=(x_val, y_val),
                            callbacks=[model_cbk, model_mckp,keras.callbacks.LearningRateScheduler(scheduler)])

        #predict             
        test_predict = model.predict(x_test)
        test_predict = y_scaler.inverse_transform(test_predict)
        meanmae_error=np.mean(abs(test_predict- np.array(y_test_orign)))
        print(" 平均mae誤差: {:.2f}".format(meanmae_error)) 
        bigmae[i,j]=round(meanmae_error,2)
        

bigmae=pd.DataFrame(bigmae)
bigmae.to_csv(r'D:/pytorch範例/transformer_tensorflow/1002transformer/100MAE/mydataMAE.csv')













