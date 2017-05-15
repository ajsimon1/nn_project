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
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras_helper import create_input, create_cs, create_context

# define constants
NUM_CS = 5
NUM_CONTEXT = 10
BIAS = 1


# other variables
us = 1

cs_input = create_cs(NUM_CS)
context_input = create_context(NUM_CONTEXT)

# set weights
start_weights = [np.random.uniform(low=-3.0, high=3.0, size=(40, 15)), np.ones(15)]

# create input
cs_context_input = create_input(cs_input, context_input)

# create model skeleton
model = Sequential()
# create layers
model.add(Dense(output_dim=40,  # neurons in layer
                name='input_layer',
                input_dim=15,))
model.add(Activation(activation='sigmoid'))
model.add(Dense(output_dim=1,
                name='hidden_layer'))
