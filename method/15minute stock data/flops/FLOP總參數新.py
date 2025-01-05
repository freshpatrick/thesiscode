# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 20:45:38 2024

@author: 2507
"""

import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, MaxPool2D
from tensorflow.keras.layers import InputLayer
from tensorflow import keras


model = tf.keras.models.Sequential([
    
      InputLayer((32, 32, 1)),
    #nputLayer((32, 32, 1)),
     #layers.Dense(20,input_shape=(1000,)),
     #InputLayer((32, 32, 1)),
      Conv2D(8, 5, padding='same', activation='relu'),
      MaxPool2D(2),
      Conv2D(16, 5, padding='same', activation='relu'),
      MaxPool2D(2),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(64, activation='relu'),
      Dense(10, activation='softmax')
  ])




model =  tf.keras.Sequential()
model.add(keras.Input(shape=(4,)))
model.add(keras.layers.Dense(2, activation="relu"))

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


get_flops(model)


get_flops(123) 





model = keras.applications.EfficientNetV2B0(weights=None, input_tensor=tf.compat.v1.placeholder('float32', shape=(1, 224, 224, 3)))






--------------------2.1-------------------
#這個版本
https://github.com/tensorflow/tensorflow/issues/32809


def get_flops(model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
        

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
        
            return flops.total_float_ops


--------------------------另一種方法---------------------
#tf.reset_default_graph()

model = tf.keras.models.Sequential([
        InputLayer((32, 32, 1)),
        # Conv2D(1, 5, padding='same'),
        # Flatten(),
        # Dense(1, activation='softmax')
    ])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')



#開始跑資料
#opts = tf.profiler.ProfileOptionBuilder.float_operation()
opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

#profile = tf.profiler.profile(tf.get_default_graph(), tf.RunMetadata(), cmd='op', options=opts)
sess = tf.compat.v1.Session()
run_meta = tf.compat.v1.RunMetadata()
profiler = tf.compat.v1.profiler
flops = profiler.profile(graph=sess.graph, run_meta=run_meta, cmd='op', options=opts)

flops.total_float_ops




----------------網路正確--------------------
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model

 #tf.reset_default_graph()
model = tf.keras.models.Sequential([
        InputLayer((32, 32, 1)),
        # Conv2D(1, 5, padding='same'),
        Flatten(),
        Dense(1, activation='softmax')
    ])



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
#opts = tf.profiler.ProfileOptionBuilder.float_operation()
sess = tf.compat.v1.Session()
run_meta = tf.compat.v1.RunMetadata()
profiler = tf.compat.v1.profiler
flops = profiler.profile(graph=sess.graph, run_meta=run_meta, cmd='op', options=opts)
flops.total_float_ops




-----------------------------------
import tensorflow as tf

def load_pb(pb_model):
    with tf.gfile.GFile(pb_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

def estimate_flops(pb_model):
    graph = load_pb(pb_model)
    with graph.as_default():
        # placeholder input would result in incomplete shape. So replace it with constant during model frozen.
        flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        print('Model {} needs {} FLOPS after freezing'.format(pb_model, flops.total_float_ops))

model = "frozen_inference_graph.pb"
estimate_flops(model)







**************************************************************************
import tensorflow as tf
import numpy as np

def get_flops(model, model_inputs) -> float:
        """
        Calculate FLOPS [GFLOPs] for a tf.keras.Model or tf.keras.Sequential model
        in inference mode. It uses tf.compat.v1.profiler under the hood.
        """
        # if not hasattr(model, "model"):
        #     raise wandb.Error("self.model must be set before using this method.")

        if not isinstance(
            model, (tf.keras.models.Sequential, tf.keras.models.Model)
        ):
            raise ValueError(
                "Calculating FLOPS is only supported for "
                "`tf.keras.Model` and `tf.keras.Sequential` instances."
            )

        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2_as_graph,
        )

        # Compute FLOPs for one sample
        batch_size = 1
        inputs = [
            tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype)
            for inp in model_inputs
        ]

        # convert tf.keras model into frozen graph to count FLOPs about operations used at inference
        real_model = tf.function(model).get_concrete_function(inputs)
        frozen_func, _ = convert_variables_to_constants_v2_as_graph(real_model)

        # Calculate FLOPs with tf.profiler
        run_meta = tf.compat.v1.RunMetadata()
        opts = (
            tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder().float_operation()
            )
            .with_empty_output()
            .build()
        )

        flops = tf.compat.v1.profiler.profile(
            graph=frozen_func.graph, run_meta=run_meta, cmd="scope", options=opts
        )

        tf.compat.v1.reset_default_graph()

        # convert to GFLOPs
        return (flops.total_float_ops / 1e9)/2
    
    
    
#Usage

image_model = tf.keras.applications.EfficientNetB0(include_top=False, weights=None)
x = tf.constant(np.random.randn(1,256,256,3))
print(get_flops(image_model, [x]))




#-------------------------------------成功#####################

model = tf.keras.models.Sequential([
    
      InputLayer((32, 32, 1)),
    #nputLayer((32, 32, 1)),
     #layers.Dense(20,input_shape=(1000,)),
     #InputLayer((32, 32, 1)),
      Conv2D(8, 5, padding='same', activation='relu'),
      MaxPool2D(2),
      Conv2D(16, 5, padding='same', activation='relu'),
      MaxPool2D(2),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(64, activation='relu'),
      Dense(10, activation='softmax')
  ])


def flops():
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            #model = keras.applications.EfficientNetV2B0(weights=None, input_tensor=tf.compat.v1.placeholder('float32', shape=(1, 224, 224, 3)))

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # Optional: save printed results to file
            # flops_log_path = os.path.join(tempfile.gettempdir(), 'tf_flops_log.txt')
            # opts['output'] = 'file:outfile={}'.format(flops_log_path)

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)

    tf.compat.v1.reset_default_graph()

    return flops.total_float_ops


flops()