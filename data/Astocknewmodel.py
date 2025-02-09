# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 22:10:49 2025

@author: User
"""
import pandas as pd
#import tensorflow as tf
from keras.datasets import imdb
#from keras.preprocessing import sequence
from keras_preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import numpy as np
np.random.seed(10)
import keras
import os
import tensorflow as tf
import matplotlib.pyplot as plt

# Load  dataset
#path= r'C:/Users/2507/Desktop/遠端資料/data/Astockdata/'
path=r'D:/2021 4月開始的找回程式之旅/0409論文想做的題目/0506比較方法/比較資料資料庫/Astock A New Dataset and Automated Stock Trading based on Stock-specific News Analyzing Model/data/'

df_train = pd.read_csv(path+'train.csv',sep='\t')
df_test = pd.read_csv(path+'test.csv',sep='\t')
df_val = pd.read_csv(path+'val.csv',sep='\t')


#*********************train**********************
#載入文字欄位
train_text1=df_train['text_a']
#先讀取所有文章建立字典，限制字典的數量為nb_words=2000
token = Tokenizer(num_words=3800)
token.fit_on_texts(train_text1)

##將文字轉為數字序列
x_train_seq1 = token.texts_to_sequences(train_text1)
#截長補短，讓所有影評所產生的數字序列長度一樣
x_train1 = sequence.pad_sequences(x_train_seq1, maxlen=100)
y_train=df_train['label']

#*********************test**********************
#載入文字欄位
test_text1=df_test['text_a']


##將文字轉為數字序列
x_test_seq1 = token.texts_to_sequences(test_text1)
#截長補短，讓所有影評所產生的數字序列長度一樣
x_test1 = sequence.pad_sequences(x_test_seq1, maxlen=100)
y_test=df_test['label']

#*********************val**********************
#載入文字欄位
val_text1=df_val['text_a']

##將文字轉為數字序列
x_val_seq1 = token.texts_to_sequences(val_text1)
#截長補短，讓所有影評所產生的數字序列長度一樣
x_val1 = sequence.pad_sequences(x_val_seq1, maxlen=100)
y_val=df_val['label']

##轉成ONEHOT
y_test_orign=y_test
y_train=tf.one_hot(y_train,3)
y_val=tf.one_hot(y_val,3)
y_test=tf.one_hot(y_test,3)





##建立模型
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
import keras
from keras.layers import Embedding
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(output_dim=32,
                    input_dim=3800, 
                    input_length=100))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(units=256,
                activation='relu' ))
model.add(Dropout(0.2))

model.add(Dense(3,activation="softmax"))

model.summary()




#訓練模型
model.compile(keras.optimizers.Adam(0.001),
             loss=keras.losses.CategoricalCrossentropy(),  #loss=keras.losses.MeanSquaredError()
            metrics=[keras.metrics.CategoricalAccuracy()])



#set callback
model_dir =r'D:/2021 4月開始的找回程式之旅/save_best'
log_dir = os.path.join(r'D:/2021 4月開始的找回程式之旅/lab2-logs', 'model10')
model_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
model_mckp = keras.callbacks.ModelCheckpoint(model_dir + '/Best-model-1.h6', 
                                        monitor='val_categorical_accuracy', 
                                        save_best_only=True, 
                                        mode='max')

def scheduler(epoch, lr):
    if epoch < 0:
        return lr
    else:
        return lr * tf.exp(-0.1, name ='exp')

history = model.fit(x_train1, y_train,  
               batch_size=32,  
               epochs=2,  
               validation_data=(x_val1, y_val),  
               callbacks=[model_cbk, model_mckp,keras.callbacks.LearningRateScheduler(scheduler)])

history.history.keys() 



y_pred = model.predict(x_test1)



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
