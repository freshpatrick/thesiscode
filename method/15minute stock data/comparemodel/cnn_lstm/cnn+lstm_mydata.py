# -*- coding: utf-8 -*-
"""
Created on Tue May 21 00:05:40 2024

@author: User
"""

import numpy as np
import pandas as pd
import keras
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from keras_self_attention import SeqSelfAttention
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, GlobalAveragePooling1D,Conv1D,Flatten



#load data
x_bigdata=np.load(r'C:\Users\2507\Desktop\遠端資料\data\x_bigdata.npy')
y_bigdata=np.load(r'C:\Users\2507\Desktop\遠端資料\data\y_bigdata.npy')


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

#load model
n = 1
n_steps = n 
n_features = 21
model = keras.Sequential(name='model-9')
model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape = (n_steps,n_features)))
model.add(keras.layers.Bidirectional(LSTM(10,activation='relu', return_sequences=True)))
model.add(SeqSelfAttention(attention_activation='tanh'))
model.add(Flatten())
model.add(Dense(1))




model.compile(keras.optimizers.Adam(0.001),
              loss=keras.losses.MeanAbsoluteError(),
              metrics=[keras.metrics.MeanAbsoluteError()])



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


history.history.keys()  # 查看history儲存的資訊有哪些
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.ylim(0.001, 0.006)
plt.title('Mean square error')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')


#pred 
y_pred = model.predict(x_test)
y_pred = y_scaler.inverse_transform(y_pred)

meanmae_error=np.mean(abs(y_test- np.array(y_pred)))
print(" 平均mae誤差: {:.2f}".format(meanmae_error))
    
    
    
    