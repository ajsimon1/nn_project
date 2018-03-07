"""
project started 1/13/17
switched to lasagne package 8/13/17 # TODO check on this

author: adam

hour vested to date (2/27/17) --> ~ 80

modeling expperiment replicatng the classical eye blink experiment in code
add additional text here
"""
import lasagne
import numpy as np
import pandas as pd
import pprint
import theano
import theano.tensor as T

from collections import namedtuple
from lasagne.objectives import binary_crossentropy, aggregate
from lasagne.init import Constant

# TODO remove this in final project
np.set_printoptions(threshold=np.nan)
pp = pprint.PrettyPrinter()

# define constants
NUM_OF_CS = 5
NUM_OF_CONTEXT = 10
NUM_OF_EPOCHS = 250
NUM_OF_BATCHES = 25
LEARNING_RATE = 0.1

X_data = T.matrix('X_data')
Y_targets = T.matrix('Y_targets') # TODO, you dont use this anywhere

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

def create_context(num_elem_in_context):
    """
    create the context that represents elements 5-14 in the 15 element input
    vector the elements are randomly created 1s or 0s using numpys randint
    function
    arguments:
        num_elem_in_context --> # of elements in the context
    return --> x element vector equal to num_elem_in_context
    argument [x, x, x, ...]
    """
    context_vector = []
    for i in range(num_elem_in_context):
        context_vector.append(float(np.random.randint(0, high=2)))
    return context_vector

def create_cs(num_of_batches, num_elem_in_cs):
    """
    create the cs to chain to context, cs is a 5 element vector with 1 vector
    having '1' to inditace that a 'US' is expected
    """
    cs = [[0 for i in range(num_elem_in_cs)] for j in range(num_of_batches)]
    rand_number = np.random.randint(0, high=len(cs))
    cs[rand_number][0] = 1.0
    return cs

def create_input_vector(num_of_batches, num_elem_in_cs, num_elem_in_context):
    context_vector = create_context(num_elem_in_context)
    cs_vector = create_cs(num_of_batches, num_elem_in_cs)
    input_vector = []
    for array_item in cs_vector:
        input_vector.append(array_item + context_vector)
    return np.asarray(input_vector)

def create_targets(input_vector):
    targets = []
    for item in input_vector:
        if np.any(item[0] == 1.0):
            targets.append([1.0])
        else:
            targets.append([0.0])
    return np.asarray(targets)

# #############################################################################
# #################### build hippocampal networks #############################
# #############################################################################
def build_hipp_network(X_data):
    hipp_l1 = lasagne.layers.InputLayer(shape=(None, 15),
                                        input_var=X_data,
                                        W=Constant(0.0),
                                        b=Constant(0.0))
    hipp_l2 = lasagne.layers.DenseLayer(hipp_l1,
                                            num_units=8,
                                            nonlinearity=lasagne.nonlinearities.rectify)

    hipp_l3 = lasagne.layers.DenseLayer(hipp_l2,
                                            num_units=15,
                                            nonlinearity=lasagne.nonlinearities.sigmoid)

    hipp_hid_layer_formula, hipp_out_layer_formula = lasagne.layers.get_output([hipp_l2,
                                                                                hipp_l3])
    HippNetDetails = namedtuple('HippNetDetails', [
                                            'input_layer',
                                            'hidden_layer',
                                            'output_layer',
                                            'hidden_layer_formula',
                                            'output_layer_formula',
                                            ])
    return HippNetDetails(hipp_l1,
                          hipp_l2,
                          hipp_l3,
                          hipp_hid_layer_formula,
                          hipp_out_layer_formula)

def build_updates_hipp_network(hipp_l1, hipp_l3, hipp_out_layer_formula, cs_index, learning_rate=LEARNING_RATE):
    hipp_params = lasagne.layers.get_all_params(hipp_l3, trainable=True)
    hipp_loss = lasagne.objectives.squared_error(hipp_out_layer_formula,
                                                hipp_l1.input_var[cs_index]).mean()
    hipp_updates = lasagne.updates.adadelta(hipp_loss,
                                            hipp_params,
                                            rho=0.75,
                                            learning_rate = learning_rate)
    hipp_updates = lasagne.updates.apply_momentum(hipp_updates,
                                                    params=hipp_params)
    return hipp_updates

def build_functions_hipp_network(X_data, hipp_out_layer_formula, hipp_hid_layer_formula, hipp_updates):
    func_feed_forward_hipp_net = theano.function([X_data],
                                                hipp_out_layer_formula,
                                                allow_input_downcast=True)

    func_update_hipp_net = theano.function([X_data],
                                            [hipp_out_layer_formula,
                                            hipp_hid_layer_formula],
                                            updates=hipp_updates,
                                            allow_input_downcast=True)
    return func_feed_forward_hipp_net, func_update_hipp_net

# #############################################################################
# ##################### build cortical networks ###############################
# #############################################################################
def build_cort_network(X_data):
    cort_l1 = lasagne.layers.InputLayer(shape=(None, 15),
                                       input_var=X_data,
                                       W=Constant(0.0),
                                       b=Constant(0.0))
    cort_l2 = lasagne.layers.DenseLayer(cort_l1, num_units=40,
                                   nonlinearity=lasagne.nonlinearities.rectify)
    cort_l3 = lasagne.layers.DenseLayer(cort_l2, num_units=1,
                                   nonlinearity=lasagne.nonlinearities.sigmoid)
    (cort_hid_layer_formula,
     cort_out_layer_formula) = lasagne.layers.get_output([cort_l2,
                                                          cort_l3])
    CortNetDetails = namedtuple('CortNetDetails', [
                                                    'input_layer',
                                                    'hidden_layer',
                                                    'output_layer',
                                                    'hidden_layer_formula',
                                                    'output_layer_formula',
                                                    ])
    return CortNetDetails(cort_l1,
                          cort_l2,
                          cort_l3,
                          cort_hid_layer_formula,
                          cort_out_layer_formula)

def build_updates_cort_network(cort_l2, cort_l3, cort_out_layer_formula, targets):
    cort_l2.params[cort_l2.W].remove('trainable')
    cort_l2.params[cort_l2.b].remove('trainable')
    cort_params = lasagne.layers.get_all_params([cort_l3], trainable=True)
    cort_loss = binary_crossentropy(cort_out_layer_formula, targets)
    cort_loss = aggregate(cort_loss, mode='mean')
    cort_grads = theano.grad(cort_loss, wrt=cort_params)
    cort_updates = lasagne.updates.adam(cort_loss,
                                        cort_params,
                                        learning_rate = LEARNING_RATE)
    return cort_updates

def build_functions_cort_network(X_data, cort_out_layer_formula, cort_hid_layer_formula, cort_updates):
    func_feed_forward_cort_net = theano.function([X_data],
                                                 cort_out_layer_formula,
                                                 allow_input_downcast=True)
    func_update_cort_net = theano.function([X_data],
                                                 cort_out_layer_formula,
                                                 updates=cort_updates,
                                                 allow_input_downcast=True)
    return func_feed_forward_cort_net, func_update_cort_net

# #############################################################################
# ##################### build funcs to run networks ###########################
# #############################################################################
def run_hipp_network(num_of_epochs, feed_forward_func, backprop_func, input_vector):
    for epoch in range(num_of_epochs):
        feed_forward_func(input_vector)
        (hipp_raw_batch_out_act,
         hipp_raw_batch_hid_act) = backprop_func(input_vector)
        hipp_net_raw_output_list.append(hipp_raw_batch_out_act)
        hipp_net_raw_hidden_list.append(list(hipp_raw_batch_hid_act))
    return hipp_net_raw_output_list, hipp_net_raw_hidden_list

def run_intact_cort_network(num_of_epochs, feed_forward_func, lower_backprop_func, upper_backprop_func, input_vector):
    for epoch in range(num_of_epochs):
        feed_forward_func(input_vector)
        cort_raw_batch_hidden_activation = lower_backprop_func(input_vector)
        # cort_raw_batch_output_activation = upper_backprop_func(input_vector)
        # cort_net_raw_output_list.append(cort_raw_batch_output_activation)
        cort_net_raw_hidden_list.append(list(cort_raw_batch_hidden_activation))
    return cort_net_raw_hidden_list

def run_cort_network(num_of_epochs, feed_forward_func, upper_backprop_func, input_vector):
    for epoch in range(num_of_epochs):
        feed_forward_func(input_vector)
        cort_raw_batch_output_activation = upper_backprop_func(input_vector)
        cort_net_raw_output_list.append(cort_raw_batch_output_activation)
        cort_net_raw_hidden_list.append(list(cort_raw_batch_hidden_activation))
    return cort_net_raw_output_list, cort_net_raw_hidden_list

if __name__ == '__main__':
    network_type = gather_input_from_user()
    input_vector = create_input_vector(NUM_OF_BATCHES,
                                        NUM_OF_CS,
                                        NUM_OF_CONTEXT)
    output_targets = create_targets(input_vector)
    cs_index = np.where(output_targets==1.)

    hipp_net_details = build_hipp_network(X_data)
    hipp_net_updates = build_updates_hipp_network(hipp_net_details.input_layer,
                                                  hipp_net_details.output_layer,
                                                  hipp_net_details.output_layer_formula,
                                                  int(cs_index[0]),
                                                  learning_rate=LEARNING_RATE)
    (func_feed_forward_hipp_net,
     func_update_hipp_net) = build_functions_hipp_network(X_data,
                                                          hipp_net_details.output_layer_formula,
                                                          hipp_net_details.hidden_layer_formula,
                                                          hipp_net_updates)
    cort_net_details = build_cort_network(X_data)
    cort_updates = build_updates_cort_network(cort_net_details.hidden_layer,
                                             cort_net_details.output_layer,
                                             cort_net_details.output_layer_formula,
                                             input_vector)
    (func_feed_forward_cort_net,
     func_update_cort_net) = build_functions_cort_network(X_data,
                                                        cort_net_details.output_layer_formula,
                                                        cort_net_details.hidden_layer_formula,
                                                        cort_updates)
    cort_net_raw_output_list = run_cort_network(NUM_OF_EPOCHS,
                                                 func_feed_forward_cort_net,
                                                 func_update_cort_net,
                                                 input_vector)


    # TODO hey man, the run_hipp net works, do the same for CORTICAL!!!!!!
    # pp.pprint(cort_net_raw_output_list)
    # pp.pprint(cort_net_raw_hidden_list)
    # pp.pprint(hipp_net_raw_output_list)
    # print(input_vector)
    # print(output_targets)
    # print(cs_index[0])
