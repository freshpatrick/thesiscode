# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:30:11 2024

@author: 2507
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


#load data
x_bigdata=np.load(r'../../../../data/StockQM/x_bigdata.npy')
y_bigdata=np.load(r'../../../../data/StockQM/y_bigdata.npy')


indexs=np.random.permutation(len(x_bigdata)) 
train_indexs=indexs[:int(len(x_bigdata)*0.6)]
val_indexs=indexs[int(len(x_bigdata)*0.6):int(len(x_bigdata)*0.8)]
test_indexs=indexs[int(len(x_bigdata)*0.8):]

#x
x_bigdata_array=np.array(x_bigdata)
x_train=x_bigdata_array[train_indexs]
x_val=x_bigdata_array[val_indexs]
x_test=x_bigdata_array[test_indexs]

#y
y_scaler = MinMaxScaler(feature_range = (0, 1))
y_bigdata_array=np.array(y_bigdata)
y_train=y_bigdata_array[train_indexs]

y_train=y_scaler.fit_transform(pd.DataFrame(y_train))
y_val=y_bigdata_array[val_indexs]

y_val=y_scaler.fit_transform(pd.DataFrame(y_val))
y_test=y_bigdata_array[test_indexs]

y_test_orign=y_test
y_test=y_scaler.fit_transform(pd.DataFrame(y_test))







def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    
    # "ATTENTION LAYER"
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs 
    #Normalization
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    #Branch
    x_1 = layers.Conv1D(filters=4,kernel_size=1,padding='same',activation='sigmoid')(x)
    x_2 = layers.Conv1D(filters=4,kernel_size=3,padding='same',activation='sigmoid')(x)
    x = layers.Concatenate()([x_1, x_2])
       
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=2, kernel_size=2,padding='same')(x)
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
        x = layers.Dense(31, activation="sigmoid")(x)
        x = layers.Dropout(mlp_dropout)(x)
        x = layers.Dense(31, activation="sigmoid")(x)
        x = layers.Dropout(mlp_dropout)(x)
        x=x_encoder+x
    x=layers.Concatenate()([x_encoder1, x])
    outputs = layers.Dense(1)(x) 
    return keras.Model(inputs, outputs)
    


#Set hyperparameters
input_shape = x_train.shape[1:]
print(input_shape)



model = build_model(
    input_shape,
    head_size=64,
    num_heads=2,  
    ff_dim=4,
    num_transformer_encoderblocks=8, 
    num_transformer_decoderblocks=range(0,6), 
    mlp_dropout=0.2,
    dropout=0.2,
)


model.compile(keras.optimizers.Adam(0.001),
              loss=keras.losses.MeanSquaredError(),  #loss=keras.losses.MeanSquaredError()
              metrics=[keras.metrics.MeanAbsoluteError()])


model.summary()


#設定callback
model_dir = r'C:\Users\2507\Desktop\遠端資料\save_best'

log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/lab2-logs', 'model10')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5', 
                                        monitor='val_mean_absolute_error', 
                                        save_best_only=True, 
                                        mode='min')



def scheduler(epoch, lr):
    if epoch < 0:
        return lr
    else:
        return lr * tf.exp(-0.1, name ='exp')



history = model.fit(x_train, y_train, 
               batch_size=32,  
               epochs=30, 
               validation_data=(x_val, y_val),  
               callbacks=[model_cbk, model_mckp,keras.callbacks.LearningRateScheduler(scheduler)])

history.history.keys() 

y_pred = model.predict(x_test)
y_pred = y_scaler.inverse_transform(y_pred)
meanmae_error=np.mean(abs(y_pred- np.array(y_test_orign)))
print(" 平均mae誤差: {:.2f}".format(meanmae_error))



#plot loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('loss function')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')