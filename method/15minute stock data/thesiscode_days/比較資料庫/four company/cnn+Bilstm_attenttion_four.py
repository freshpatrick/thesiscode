# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:53:25 2024

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
from keras.layers import Conv1D , MaxPool2D , Flatten , Dropout


#SeqSelfAttention
class SeqSelfAttention(keras.layers.Layer):
    np.random.seed(16)
    tf.random.set_seed(16)
    
    #a=tf.constant(0.01)

    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 history_only=False,
                 #kernel_initializer='glorot_normal',
                 #改這邊
                 kernel_initializer=initializers.glorot_normal(seed=0),
                 #原本
                 #kernel_initializer=initializers.glorot_uniform(seed=0),
                 #kernel_initializer=initializers.glorot_normal(seed=0.05),
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 attention_activation=None,
                 attention_regularizer_weight=0.0,
                 **kwargs):
        """Layer initialization.

        For additive attention, see: https://arxiv.org/pdf/1806.01264.pdf

        :param units: The dimension of the vectors that used to calculate the attention weights.
        :param attention_width: The width of local attention.
        :param attention_type: 'additive' or 'multiplicative'.
        :param return_attention: Whether to return the attention weights for visualization.
        :param history_only: Only use historical pieces of data.
        :param kernel_initializer: The initializer for weight matrices.
        :param bias_initializer: The initializer for biases.
        :param kernel_regularizer: The regularization for weight matrices.
        :param bias_regularizer: The regularization for biases.
        :param kernel_constraint: The constraint for weight matrices.
        :param bias_constraint: The constraint for biases.
        :param use_additive_bias: Whether to use bias while calculating the relevance of inputs features
                                  in additive mode.
        :param use_attention_bias: Whether to use bias while calculating the weights of attention.
        :param attention_activation: The activation used for calculating the weights of attention.
        :param attention_regularizer_weight: The weights of attention regularizer.
        :param kwargs: Parameters for parent class.
        """
        super(SeqSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.attention_activation = keras.activations.get(attention_activation)
        self.attention_regularizer_weight = attention_regularizer_weight
        self._backend = keras.backend.backend()

        if attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

    def get_config(self):
        config = {
            'units': self.units,
            'attention_width': self.attention_width,
            'attention_type': self.attention_type,
            'return_attention': self.return_attention,
            'history_only': self.history_only,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'attention_activation': keras.activations.serialize(self.attention_activation),
            'attention_regularizer_weight': self.attention_regularizer_weight,
        }
        base_config = super(SeqSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self._build_additive_attention(input_shape)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self._build_multiplicative_attention(input_shape)
        super(SeqSelfAttention, self).build(input_shape)

    def _build_additive_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_additive_bias:
            self.bh = self.add_weight(shape=(self.units,),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Add_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def _build_multiplicative_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wa = self.add_weight(shape=(feature_dim, feature_dim),
                                  name='{}_Mul_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Mul_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def call(self, inputs, mask=None, **kwargs):
        input_len = K.shape(inputs)[1]

        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        if self.attention_width is not None:
            if self.history_only:
                lower = K.arange(0, input_len) - (self.attention_width - 1)
            else:
                lower = K.arange(0, input_len) - self.attention_width // 2
            lower = K.expand_dims(lower, axis=-1)
            upper = lower + self.attention_width
            indices = K.expand_dims(K.arange(0, input_len), axis=0)
            e -= 10000.0 * (1.0 - K.cast(lower <= indices, K.floatx()) * K.cast(indices < upper, K.floatx()))
        if mask is not None:
            mask = K.expand_dims(K.cast(mask, K.floatx()), axis=-1)
            e -= 10000.0 * ((1.0 - mask) * (1.0 - K.permute_dimensions(mask, (0, 2, 1))))

        # a_{t} = \text{softmax}(e_t)
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        a = e / K.sum(e, axis=-1, keepdims=True)

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        v = K.batch_dot(a, inputs)
        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(a))

        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)
        if self.use_additive_bias:
            h = K.tanh(q + k + self.bh)
        else:
            h = K.tanh(q + k)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        if self.use_attention_bias:
            e = K.reshape(K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = K.reshape(K.dot(h, self.Wa), (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs):
        # e_{t, t'} = x_t^T W_a x_{t'} + b_a
        e = K.batch_dot(K.dot(inputs, self.Wa), K.permute_dimensions(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            e += self.ba[0]
        return e

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_regularizer(self, attention):
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        indices = K.expand_dims(K.arange(0, input_len), axis=0)
        diagonal = K.expand_dims(K.arange(0, input_len), axis=-1)
        eye = K.cast(K.equal(indices, diagonal), K.floatx())
        return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(
            attention,
            K.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}


AAPL= yf.download("AAPL", start="1980-01-01", end="2024-07-31")
TSLA= yf.download("TSLA", start="1980-01-01", end="2024-07-31")
MSFT= yf.download("MSFT", start="1980-01-01", end="2024-07-31")
IBM = yf.download("IBM ", start="1980-01-01", end="2024-07-31")


final_data_real=[]
final_data_real.append(AAPL)
final_data_real.append(TSLA)
final_data_real.append(MSFT)
final_data_real.append(IBM)

#####################合併大資料集 final_data_real##############################
final_data_real=[]
final_data_real.append(AAPL)
final_data_real.append(TSLA)
final_data_real.append(MSFT)
final_data_real.append(IBM)

stock_id=['AAPL','TSLA','MSFT','IBM']

stock_mae=[] #股票MSE
stock=[] #股票名稱

###複製表格
final_data_real_copy=final_data_real


###使用minmaxscalar
x_scaler = MinMaxScaler(feature_range = (0, 1))
y_scaler = MinMaxScaler(feature_range = (0, 1))

for k in range(0,4):  #len(stock_id)  5
    print("第"+str(k)+"支股票")
    tsla_data =final_data_real[k]
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
    y_test_tran=y_scaler.fit_transform(y_test_dataframe)
    y_test=np.array(y_test_tran).reshape(-1) 
    #開始跑模型
    n = 10
    n_steps = n 
    n_features = 4
    model = keras.models.Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape = (n_features,n_steps)))
    model.add(keras.layers.Bidirectional(LSTM(10,activation='relu', return_sequences=True)))
    #model.add(LSTM(10,activation='relu', return_sequences=True, input_shape = (n_steps, n_features)))  
    model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Flatten())
    #model.add(LSTM(50,activation='relu'))
    model.add(Dense(1))
    # 顯示網路模型架構
    model.summary()
    model.compile(keras.optimizers.Adam(0.001),
    loss=keras.losses.MeanSquaredError(),  #loss=keras.losses.MeanSquaredError()  loss=custom_mean_squared_error
    metrics=[keras.metrics.MeanAbsoluteError()])
    #設定回調函數
    model_dir = r'D:/2021 4月開始的找回程式之旅/lab2-logs/fivestock/model8/'
    log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/lab2-logs/fivestock', 'model8')
    model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
    model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h5', 
                                                 monitor='val_mean_absolute_error', 
                                                 save_best_only=True, 
                                                 mode='min')
    

    history = model.fit(x_train, y_train,  
               batch_size=32,  
               epochs=30,  
               validation_data=(x_val, y_val), 
               callbacks=[model_cbk, model_mckp])  

    predictions = model.predict(x_test)
    predictions1=predictions.reshape(-1)
    predictions_orign = y_scaler.inverse_transform(predictions)
    meanmae_error=np.mean(abs(predictions_orign- np.array(y_testc)))

    stock_mae.append(meanmae_error) 
    stock.append(stock_id[k])
    

##合併大資料 ############
big_lstm_data=pd.concat([pd.DataFrame(stock),pd.DataFrame(stock_mae)], axis=1)
#big_lstm_data.mean()  

#big_lstm_data.to_csv(r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/0506比較方法/五個股票/MINMAXSCALAR 3060five_outputdata_lstm.csv', encoding='utf_8_sig')

