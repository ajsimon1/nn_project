"""
attempt at research project with keras library
started 3/22/2017
adam simon
coninuted 4/21/2017, hours to date = approx. 10

the network is a model of the cotrical / cerebellur network
there are 15 inputs nodes that received an input vector, with 5 nodes defined
as the conditioned stimulus (CS) and 10 nodes deinfed as the environmental
context.  the input network is initialized with random values of 1 & 0
the lower layer weights between the input and hidden layers are fixed and
only updated by the hippocampal hidden layer.  when the HHL is missing the
weights are fixed.  as such the input vector is passed directly to the cortical
hidden layer, where there is an activation function.
"""
import numpy as np
import keras

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import initializers
from keras import optimizers
from keras import losses
from keras import utils

from keras_helper import create_input, create_cs, create_context

# define constants
NUM_CS = 5
NUM_CONTEXT = 10
BIAS = 1


# other variables
us = 1

cs_input = create_cs(NUM_CS)
context_input = create_context(NUM_CONTEXT)

# create input
cs_context_input = create_input(cs_input, context_input)

# create optimizers
sgd = optimizers.SGD()

# create losses
msqe = losses.mean_squared_error

# create the training set as a different context on the same cs
context_input_train = create_context(NUM_CONTEXT)
cs_context_input_train = create_input(cs_input, context_input_train)

# y set for fit function, set cateogries?
y_1 = range(100)
y_final = utils.to_categorical(y_1)

# create model skeleton
model = Sequential()
# create layers
model.add(Dense(units=40,  # neurons in layer
                name='input_layer',
                input_shape=(,15),
                kernel_initializer=initializers.random_uniform(minval=-3.0,maxval=3.0),
                ))
model.add(Activation(activation='sigmoid'))
model.add(Dense(units=1,
                name='hidden_layer'))

# need to comple the model before training
model.compile(optimizer=sgd, loss=msqe,
              metrics=['accuracy'])

# getting a consistent error on input shape
# ValueError: Error when checking input: expected input_layer_input to have 3 dimensions, but got array with shape (15, 1)
model.fit(cs_context_input_train, y_final)
