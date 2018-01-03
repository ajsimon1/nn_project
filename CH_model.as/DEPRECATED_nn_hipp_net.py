"""
created 9/20/17

author: adam

hour vested to date (11/18/17) --> ~ 55

first attempt at creating the hippocampal network
this network contains 3 layers, 15 input layer, 8 hidden layer, 16 output layer
the output layer produces a mirror of the input layer with a prediction
the hidden layer produces training data for the cortical network to look at
"""

import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.nonlinearities import sigmoid, rectify
from DEPRECATED_nn_proj_utils import create_input_vector, create_targets
import matplotlib.pyplot as plt
import pprint
from theano.compile.nanguardmode import NanGuardMode

X_data = T.matrix('X_data') # data
Y_targets = T.matrix('Y_targets') # targets

NUM_OF_CS = 5 # the variable conditioned stimuls coupled with the context
NUM_OF_CONTEXT = 10 # the immutable context that reflects the 'environment'
# CS & CONTEXT are combined to equal an 'input vector'
NUM_OF_TRIALS = 1 # trial is a collection of input vector
NUM_OF_BATCHES = 100 # batch is a collection of trials
LEARNING_RATE = 0.1 # learning rate to update the network after error backprop

data = create_input_vector(NUM_OF_TRIALS, NUM_OF_CS, NUM_OF_CONTEXT)

targets = create_targets(data)

input_layer = lasagne.layers.InputLayer(shape=(None, 15), input_var=X_data)

hidden_layer = lasagne.layers.DenseLayer(input_layer, num_units=8, nonlinearity=rectify)

output_layer = lasagne.layers.DenseLayer(hidden_layer, num_units=15, nonlinearity=sigmoid)


hidden_layer_output = lasagne.layers.get_output(hidden_layer)
network_output = lasagne.layers.get_output(output_layer)

params = lasagne.layers.get_all_params(output_layer, trainable=True)


loss = lasagne.objectives.squared_error(input_layer.input_var, network_output).mean()
# grads = theano.grad(loss, wrt=params)
updates = lasagne.updates.adadelta(loss, params, rho=0.5, learning_rate = LEARNING_RATE)

updates = lasagne.updates.apply_momentum(updates, params=params)

train = theano.function([X_data], network_output, updates=updates)

predict = theano.function([X_data], network_output, allow_input_downcast=True)

encode = theano.function([X_data], hidden_layer_output, allow_input_downcast=True)

net_output_list = []
encoded_output_list = []

# is it a brand new network everytime?

for epoch in range(NUM_OF_BATCHES):
    predict(data)
    net_output = train(data)
    net_output_list.append(net_output)

zip_input_putput = list(zip(data, net_output_list))
pp = pprint.PrettyPrinter()

pp.pprint(net_output_list)
print(data)
# pp.pprint(encoded_output_list)
# pp.pprint(zip_input_putput)
# print(zip_input_putput)
