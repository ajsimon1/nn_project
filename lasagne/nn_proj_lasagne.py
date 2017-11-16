import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.objectives import binary_crossentropy, aggregate
from nn_helper_funcs import create_input_vector, create_targets

# define variables
# data = input to the network
# targets = what the expected output is per data/input
# remeber the .zip() function if you need to compare the iterables
X_data = T.matrix('X_data') # data
Y_targets = T.matrix('Y_targets') # targets
# CONSTANTS
NUM_OF_CS = 5 # the variable conditioned stimuls coupled with the context
NUM_OF_CONTEXT = 10 # the immutable context that reflects the 'environment'
# CS & CONTEXT are combined to equal an 'input vector'
NUM_OF_TRIALS = 25 # trial is a collection of input vector
NUM_OF_BATCHES = 5 # batch is a collection of trials
LEARNING_RATE = 0.01 # learning rate to update the network after error backprop

# both imported functions below available in the nn_helper_funcs file
# create input vector based on CONSTANTS provided by user
data = create_input_vector(NUM_OF_BATCHES, NUM_OF_CS, NUM_OF_CONTEXT)
# data = np.random.rand(5, 15) # sample dummy data
print(data)
# TODO make the data EXACTLY like the random data from the website
# create targets based on length of input vector
# match 'US' target with input vector containing '1' as first element
targets = create_targets(data)
print(targets)
# TODO split into train & test data?

# defining the model, a 3 layer model with 1 input, 1 hidden, 1 output
# all initial paramters including weights matrix and biases are set by
# lasagne neural network framework defaults, changes can be made if needed
# the input layer expects a 15 element vector, batch size is inconsequential
#  input vector equals the defined tensor variable above, a matrix like dtype
l1 = lasagne.layers.InputLayer(shape=(None, 15), input_var=X_data)
# hidden layer has 40 units and rectifier activation function equivalent to
# f(x) = max(0, x) where x is the input to the neuron
l2 = lasagne.layers.DenseLayer(l1, num_units=40, nonlinearity=rectify)
# output layer with 1 unit and sigmoid activation function to limit output
# to values between 1 and 0, sigmoid funcion is defined as
# f(x) = 1 / 1 + e**-x
l3 = lasagne.layers.DenseLayer(l2, num_units=1, nonlinearity=sigmoid)

# function that defines how data show move through the network
predictions = lasagne.layers.get_output(l3)
# formula for above:
# 'sigmoid((((TensorConstant{0.5} * (((input \\dot W) + b) +
# |((input \\dot W) + b)|)) \\dot W) + b))'
# executing the function that pushes input data (x) through the system using
# equation defined by Y, Y then becomes the actual output
# last variable allows lasagne to automatically downcast any higher bit dtype
# to a lower dtype, a float64 to a float32 for example
func_feed_data_through_nn = theano.function([X_data],
                                            predictions,
                                            allow_input_downcast=True)
# TODO having issues with the shape of 'data'
# TODO test function was fn(np.random.randn(3, 784)) where 3 is batches
# and 784 is first layer input
# this comes out as 'array[# of batches[# of units]]'
func_feed_data_through_nn(data)

# begin to gather paramters to update with loss/error function
# get the parameters, trainable=True only returns parameters that can be trained
# only calling the output layer is necessary, all the trainable paramters
# going backwards to input will be gathered also
params = lasagne.layers.get_all_params(l3, trainable=True)

# predictions = actual output of the network, i.e. what output is
# targets is the supervised output, i.e. what output should be
loss = binary_crossentropy(predictions, targets)
loss = aggregate(loss, mode='mean') # mean of loss across all NUM_OF_BATCHES

# get the gradient of a loss function with respect to these parameters
grads = theano.grad(loss, wrt=params)

# define an updat dict that applies the learning rule to the parameters
learning_rate = LEARNING_RATE
# using built in stochasitc gradient descent 'sgd'
# in below function, this will 'update' the network
updates = lasagne.updates.sgd(grads, params, learning_rate)

# function for error backpropagation and updating the network paramters
# there are 2 outputs:
# loss = the mean of loss average for each trial over the entire batch
# the predictions of every trial in the batch
# ie every feed forward gets a backpropagation
func_update_network = theano.function([X_data],
                                    [loss, predictions],
                                    updates=updates,
                                    allow_input_downcast=True)

print(func_update_network(data))

# TODO need to collect the predictions into a graphable dtype
# TODO need a pyplot for output predictions data
