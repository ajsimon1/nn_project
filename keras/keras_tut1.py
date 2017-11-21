# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 20:03:26 2017

@author: Adam
"""

# keras sequential tutorial, trying to get output values
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import layer_utils
import numpy as np
import keras.backend as K

# dummy data
data = np.random.random((1000,100))
labels = np.random.randint(2, size=(1000,1))

model = Sequential()

model.add(Dense(units=32, activation='relu', input_dim=100))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer='rmsprop', 
              loss='mse', 
              metrics=['accuracy'])

history = model.fit(data, labels, epochs=10, batch_size=32)

print(model.layers[1].get_config())
print('\n')
print(model.layers[1].get_output_at(node_index=0))
print('\n')
print(model.layers[1].output)
print('\n')
print(model.layers[1].output_mask)
print('\n')
print(model.layers[1].output_shape)
print('\n')
print(history.history.keys())
print('\n')
print(layer_utils.print_layer_summary_with_connections(model.layers[1]))