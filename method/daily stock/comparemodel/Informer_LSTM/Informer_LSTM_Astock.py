# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 17:34:46 2025

@author: 2507
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import yfinance as yf
import os
import keras


#load data
x_train=np.load('C:/Users/2507/Desktop/遠端資料/data/Astockdata/Astockx_train.npy')
x_val=np.load('C:/Users/2507/Desktop/遠端資料/data/Astockdata/Astockx_val.npy')
x_test=np.load('C:/Users/2507/Desktop/遠端資料/data/Astockdata/Astockx_test.npy')
y_train=np.load('C:/Users/2507/Desktop/遠端資料/data/Astockdata/Astocky_train.npy')
y_val=np.load('C:/Users/2507/Desktop/遠端資料/data/Astockdata/Astocky_val.npy')
y_test=np.load('C:/Users/2507/Desktop/遠端資料/data/Astockdata/Astocky_test.npy')

y_test_orign=y_test
y_scaler = MinMaxScaler(feature_range = (0, 1))
y_train=tf.one_hot(y_train,3)
y_val=tf.one_hot(y_val,3)
y_test=tf.one_hot(y_test,3)


# Prepare data for LSTM
look_back = 25 # Number of previous time steps to use as input variables
n_features = 2


#機率稀疏自註意力：
class ProbSparseSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(ProbSparseSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        # Assert that d_model is divisible by num_heads
        assert self.d_model % self.num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.depth = d_model // self.num_heads

        # Defining the dense layers for Query, Key and Value
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Fixing matrix multiplication
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        d_k = tf.cast(self.depth, tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))

        return self.dense(concat_attention)
    


#建立模型
class InformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, conv_filters, **kwargs):
        super(InformerEncoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads

        # Assert that d_model is divisible by num_heads
        assert self.d_model % self.num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.self_attention = ProbSparseSelfAttention(d_model=d_model, num_heads=num_heads)

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # This dense layer will transform the input 'x' to have the dimensionality 'd_model'
        self.dense_transform = tf.keras.layers.Dense(d_model)

        self.conv1 = tf.keras.layers.Conv1D(conv_filters, 3, padding='same')
        self.conv2 = tf.keras.layers.Conv1D(d_model, 3, padding='same')
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.dense = tf.keras.layers.Dense(d_model)

    def call(self, x):
        attn_output = self.self_attention(x, x, x)

        # Transform 'x' to have the desired dimensionality
        x_transformed = self.dense_transform(x)
        attn_output = self.norm1(attn_output + x_transformed)

        conv_output = self.conv1(attn_output)
        conv_output = tf.nn.relu(conv_output)
        conv_output = self.conv2(conv_output)

        encoded_output = self.norm2(conv_output + attn_output)

        pooled_output = self.global_avg_pooling(encoded_output)
        return self.dense(pooled_output)[:, -4:]



input_layer = tf.keras.layers.Input(shape=(look_back, n_features))

# Encoder
encoder_output = InformerEncoder(d_model=360, num_heads=8, conv_filters=64)(input_layer)

# Decoder (with attention)
decoder_lstm = tf.keras.layers



input_layer = tf.keras.layers.Input(shape=(look_back, n_features))

# Encoder
encoder_output = InformerEncoder(d_model=360, num_heads=8, conv_filters=64)(input_layer)

# Decoder (with attention)
decoder_lstm = tf.keras.layers



from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def InformerModel(input_shape, d_model=64, num_heads=2, conv_filters=256, learning_rate= 1e-3):
    # Input
    input_layer = Input(shape=input_shape)

    # Encoder
    encoder_output = InformerEncoder(d_model=d_model, num_heads=num_heads, conv_filters=conv_filters)(input_layer)

    # Decoder
    repeated_output = RepeatVector(4)(encoder_output)  # Repeating encoder's output
    decoder_lstm = LSTM(312, return_sequences=True)(repeated_output)
    decoder_output = Dense(3,activation="softmax")(decoder_lstm[:, -1, :])  # Use the last sequence output to predict the next value

    # Model
    model = Model(inputs=input_layer, outputs=decoder_output)
    # Compile the model with the specified learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

model = InformerModel(input_shape=(look_back, n_features))
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


#callback
model_dir = r'C:\Users\2507\Desktop\遠端資料\save_best'

log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/lab2-logs', 'model10')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5', 
                                        monitor='val_categorical_accuracy', 
                                        save_best_only=True, 
                                        mode='max')



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



#accuracy
y_prelabel=[]
for j in range(0,len(y_pred)):
    y_label=np.where(y_pred[j] ==max(y_pred[j])) 
    y_prelabel.append(y_label[0][0])
    
y_prelabel=np.array(y_prelabel)    
accuracy=(y_prelabel==y_test_orign).mean()
print("準確率為"+str(accuracy*100)+"%")



#plot loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.title('loss function')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')