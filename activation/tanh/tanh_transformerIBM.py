# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:20:19 2024

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




df = yf.download("IBM", start="1980-01-01", end="2024-07-31")
#df = yf.download("TSLA", start="1980-01-01", end="2024-07-31")

data = df[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
#資料
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
#test_size = len(data_scaled) - training_size
train_data,val_data,test_data = data_scaled[0:training_size,:], data_scaled[training_size:validat_size,:], data_scaled[validat_size:len(data_scaled),:]



X_train, y_train = create_dataset(train_data, time_step)
X_val, y_val = create_dataset(val_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)



# Reshape input for the model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

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
        x = layers.Dense(10, activation="tanh")(x)
        x = layers.Dropout(mlp_dropout)(x)
        x = layers.Dense(10, activation="tanh")(x)
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
input_shape = X_train.shape[1:]

print(input_shape)
# epoch_number=2
epoch_number=20
# epoch_number = 200
# batch_size=4
batch_size=64
# batch_size=64


model = build_model(
    input_shape,
    head_size=64, #256
    num_heads=2,  #4
    ff_dim=4,
    num_transformer_blocks=8,  #IBM用2 validation會比較變動  4
    #mlp_units=[128],
    mlp_units=range(0,3), #mlp_units=range(0,5),
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
model_dir = r'C:\Users\2507\Desktop\遠端資料\save_best'

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



history = model.fit(X_train, y_train,  # 傳入訓練數據
               batch_size=32,  # 批次大小設為64
               epochs=30,  # 整個dataset訓練100遍
               validation_data=(X_val, y_val),  # 驗證數據
               callbacks=[model_cbk, model_mckp,keras.callbacks.LearningRateScheduler(scheduler)])
               #callbacks = [
                   #keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                   #keras.callbacks.LearningRateScheduler(scheduler)
                   #]
               #)  # Tensorboard回調函數紀錄訓練過程，ModelCheckpoint回調函數儲存最好的模型
               







# Make predictions
train_predict = model.predict(X_train)
val_predict = model.predict(X_val)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
val_predict = scaler.inverse_transform(val_predict)
test_predict = scaler.inverse_transform(test_predict)

# Evaluate the model (Optional: Calculate RMSE or other metrics)
#train_rmse = math.sqrt(mean_squared_error(y_train, scaler.inverse_transform(train_predict.reshape(-1, 1))))
#test_rmse = math.sqrt(mean_squared_error(y_test, scaler.inverse_transform(test_predict.reshape(-1, 1))))

#print(f"Train RMSE: {train_rmse}")
#print(f"Test RMSE: {test_rmse}")

from sklearn.metrics import mean_absolute_error
#y_testorign=scaler.inverse_transform(y_test.reshape(-1, 1))

y_testorign=scaler.inverse_transform(data_scaled)[(validat_size+time_step+1):len(data_scaled),:]

meanmae_error=np.mean(abs(test_predict- np.array(y_testorign)))
test_rmse = math.sqrt(mean_squared_error(test_predict, np.array(y_testorign)))

print(" 平均mae誤差: {:.2f}".format(meanmae_error))  #12.73
print(" RMSE誤差: {:.2f}".format(test_rmse))  #13.18



# Plotting the results
# Adjust the time_step offset for plotting
trainPredictPlot = np.empty_like(data_scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_step:len(train_predict)+time_step, :] = train_predict

#val_plot
valPredictPlot = np.empty_like(data_scaled)
valPredictPlot[:, :] = np.nan
valPredictPlot[(len(train_predict)+(time_step*2)+1):(validat_size-1), :] = val_predict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(data_scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[validat_size+(time_step):len(data_scaled)-1, :] = test_predict

# Plot baseline and predictions
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(data_scaled), label='Actual Stock Price')
plt.plot(trainPredictPlot, label='Train Predict')
plt.plot(valPredictPlot, label='val Predict')
plt.plot(testPredictPlot, label='Test Predict')
plt.title('Stock Price Prediction using Transformer')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


##畫LOSS圖
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
plt.figure(figsize=(12, 6))
plt.plot(history.history['mean_absolute_error'], label='train')
plt.plot(history.history['val_mean_absolute_error'], label='validation')
plt.title('mae')
plt.ylabel('mae')
plt.xlabel('epochs')
plt.legend(loc='upper right')
