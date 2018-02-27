"""
created 9/20/17

author: adam

hour vested to date (11/18/17) --> ~ 80

modeling expperiment replicatng the classical eye blink experiment in code
add additional text here
"""
import lasagne
import numpy as np
import pandas as pd
import pprint
import theano
import theano.tensor as T

from DEPRECATED_nn_proj_utils import create_input_vector, create_targets
from lasagne.objectives import binary_crossentropy, aggregate
from lasagne.init import Constant

# set pretty printing options for results checking
# TODO remove this in final project
np.set_printoptions(threshold=np.nan)
pp = pprint.PrettyPrinter()

# define constants
NUM_OF_CS = 5 # the variable conditioned stimuls coupled with the context
NUM_OF_CONTEXT = 10 # the immutable context that reflects the 'environment'
# CS & CONTEXT are combined to equal an 'input vector'
NUM_OF_TRIALS = 25 # trial is a collection of input vector
NUM_OF_BATCHES = 500 # batch is a collection of trials
LEARN_RATE = 0.01 # learning rate to update the network after error backprop

# define theano variables for net inputs
X_data = T.matrix('X_data') # data
Y_targets = T.matrix('Y_targets') # targets

# define list variables to proces net outputs
hipp_net_raw_output_list = []
hipp_net_raw_hidden_list = []
cort_net_raw_output_list = []
cort_net_raw_hidden_list = []
cort_net_us_present_output_list = []
cort_net_us_absent_output_list = []
cort_net_us_present_hidden_layer_activations_list = []
cort_net_us_absent_hidden_layer_activations_list = []
c_dist_list = []

# #############################################################################
# ############################ create data/input ##############################
# #############################################################################
# gather user input to define network to run
def gather_input_from_user():
    """
    Function to accept user input for the network.  Currently user input
    only defines which network to run.
    Options:
        'i' = Intact --> Run both the bippocampal net and the cortical net with
                         the hippocampal hidden layers training the cortical
                         layers
        'l' = Lesion --> Run the cortical net in isolation of the hippocampal;
                         the cortical network will only be trained on the top
                         level weights as the hippocampal hidden layer weights
                         would otherwise train the cortical net lower layer
                         weights
        'p' = Physostigmine --> increase the learning rate to mirror the effects
                         of administiring the Physostigmine drug which affects
                         brain functions, see link below for more information
                         https://en.wikipedia.org/wiki/Physostigmine
        's' = Scopolamine --> decrease the learning rate to mirror the effects
                         of administering Scopolamine drug which affects brain
                         function, see link below for more information
                         https://en.wikipedia.org/wiki/Hyoscine
    """
    user_input = input('NET type (i=intact, l=lesion, s=scop, p=phys)? '.lower())
    return user_input

# create the input vector based off CONSTANTS defined by user
def create_input_vector(num_of_batches, num_elem_in_cs, num_elem_in_context):
    context_vector = create_context(num_elem_in_context)
    cs_vector = create_cs(num_of_batches, num_elem_in_cs)
    input_vector = []
    for array_item in cs_vector:
        input_vector.append(array_item + context_vector)
    return np.asarray(input_vector)

# create targets based on 1st element in input vector
# vector with '1' in 1st elemnt spot should get a '1' in the targets vector
# '0' in the input vector gets '0' in the targets vector
def create_targets(input_vector):
    targets = []
    for item in input_vector:
        if np.any(item[0] == 1.0):
            targets.append([1.0])
        else:
            targets.append([0.0])
    return np.asarray(targets)

# #############################################################################
# ##################### build cortical networks ###############################
# #############################################################################
cort_l1 = lasagne.layers.InputLayer(shape=(None, 15),
                               input_var=X_data,
                               W=Constant(0.0))
cort_l2 = lasagne.layers.DenseLayer(cort_l1, num_units=40,
                               nonlinearity=lasagne.nonlinearities.rectify)
cort_l3 = lasagne.layers.DenseLayer(cort_l2, num_units=1,
                               nonlinearity=lasagne.nonlinearities.sigmoid)
cort_hid_layer_act, cort_out_layer_act = lasagne.layers.get_output([cort_l2,
                                                                    cort_l3])

# executing the function that pushes input data (x) through the system using
# equation defined by Y, Y then becomes the actual output, ie output layer
# activations.  also grabbing hidden layer activations for hamming distance
# last variable allows lasagne to automatically downcast any higher bit dtype
# to a lower dtype, a float64 to a float32 for example
func_feed_forward_cort_net = theano.function([X_data],
                                            [cort_out_layer_act,
                                             cort_hid_layer_act],
                                            allow_input_downcast=True)
# remove trainable tag so that lower layer weights are not trained
cort_l2.params[cort_l2.W].remove('trainable')
cort_l2.params[cort_l2.b].remove('trainable')
# get the parameters, trainable=True only returns parameters that can be trained
cort_l3_params = lasagne.layers.get_all_params(cort_l3, trainable=True)
# output_layer_activation = actual output of the network, i.e. what output is
# targets is the supervised output, i.e. what output should be
cort_upper_layer_loss = binary_crossentropy(cort_out_layer_act, targets)
# TODO cort_loss = binary_crossentropy(cort_hid_layer_act, hipp_hid_layer_act)
# TODO convert hipp_hid_layer_act to same shape as cort_hid_layer_act
cort_upper_layer_loss = aggregate(cort_upper_layer_loss, mode='mean') # mean loss across all batches
# get the gradient of a loss function with respect to these parameters
cort_grads = theano.grad(cort_upper_layer_loss, wrt=cort_l3_params)
# using built in stochasitc gradient descent 'sgd'
# in below function, this will 'update' the network
cort_updates = lasagne.updates.adam(cort_loss, cort_params, LEARN_RATE)
# function for error backpropagation and updating the network paramters
func_update_cort_net = theano.function([X_data],
                                    [cort_out_layer_act,
                                     cort_hid_layer_act],
                                    updates=cort_updates,
                                    allow_input_downcast=True)


if __name__ == __main__:
    network_type = gather_input_from_user()
    input_vector = create_input_vector(NUM_OF_BATCHES,
                                        NUM_OF_CS,
                                        NUM_OF_CONTEXT)
    output_targets = create_targets(input_vector)
