# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 23:36:51 2024

@author: User
"""

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


# Load  dataset
df = yf.download("IBM", start="1980-01-01", end="2024-07-31")
data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)


def create_dataset(dataset, time_step=100):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# Parameters
time_step = 10
training_size = int(len(data_scaled) * 0.6)
validat_size=int(len(data_scaled) * 0.8)
test_size = len(data_scaled) - validat_size
train_data,val_data,test_data = data_scaled[0:training_size,:], data_scaled[training_size:validat_size,:], data_scaled[validat_size:len(data_scaled),:]



X_train, y_train = create_dataset(train_data, time_step)
X_val, y_val = create_dataset(val_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


# Reshape input for the model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
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
    #執行3次transformerblock
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    for dim in mlp_units:
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x=layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    outputs = layers.Dense(1)(x)  
    return keras.Model(inputs, outputs)
    
#Set hyperparameters
input_shape = X_train.shape[1:]
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
            head_size=64, 
            num_heads=2,  
            ff_dim=4,
            num_transformer_blocks=i, 
            mlp_units=range(0,j),
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



        history = model.fit(X_train, y_train,                          
                           batch_size=32, 
                           epochs=30, 
                           validation_data=(X_val, y_val),  
                           callbacks=[model_cbk, model_mckp,keras.callbacks.LearningRateScheduler(scheduler)])

        # Make predictions
        train_predict = model.predict(X_train)
        val_predict = model.predict(X_val)
        test_predict = model.predict(X_test)

        # Inverse transform predictions
        train_predict = scaler.inverse_transform(train_predict)
        val_predict = scaler.inverse_transform(val_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_testorign=scaler.inverse_transform(data_scaled)[(validat_size+time_step+1):len(data_scaled),:]
        meanmae_error=np.mean(abs(test_predict- np.array(y_testorign)))
        test_rmse = math.sqrt(mean_squared_error(test_predict, np.array(y_testorign)))
        print(" 平均mae誤差: {:.2f}".format(meanmae_error)) 
        print(" RMSE誤差: {:.2f}".format(test_rmse)) 
        bigmae[i,j]=round(meanmae_error,2)
        
#output_csv
bigmae=pd.DataFrame(bigmae)
bigmae.to_csv(r'D:/pytorch範例/transformer_tensorflow/1002transformer/100MAE/TRANSFORMER_IBM.csv')


