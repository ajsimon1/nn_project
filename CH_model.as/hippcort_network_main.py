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
NUM_OF_TRIALS = 250 # note this does not match previous nn_proj_main
NUM_OF_BATCHES = 20 # note this does not match previous nn_proj_main
LEARN_RATE = 0.1

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

    hipp_hid_layer_act, hipp_out_layer_act = lasagne.layers.get_output([hipp_l2,
                                                                        hipp_l3])
    HippNetDetails = namedtuple('HippNetDetails', [
                                            'input_layer',
                                            'hidden_layer',
                                            'output_layer',
                                            'hidden_layer_activation',
                                            'output_layer_activation',
                                            ])
    return HippNetDetails(hipp_l1,
                          hipp_l2,
                          hipp_l3,
                          hipp_hid_layer_act,
                          hipp_out_layer_act)

def build_updates_hipp_network(hipp_l1, hipp_l3, hipp_out_layer_act, cs_index):
    hipp_params = lasagne.layers.get_all_params(hipp_l3, trainable=True)
    hipp_loss = lasagne.objectives.squared_error(hipp_l1.input_var[cs_index],
                                                hipp_out_layer_act).mean()
    hipp_updates = lasagne.updates.adadelta(hipp_loss,
                                            hipp_params,
                                            rho=0.75,
                                            learning_rate = LEARN_RATE)
    hipp_updates = lasagne.updates.apply_momentum(hipp_updates,
                                                    params=hipp_params)
    return hipp_updates

def build_functions_hipp_network(X_data, hipp_out_layer_act, hipp_hid_layer_act, hipp_updates):
    func_feed_forward_hipp_net = theano.function([X_data],
                                                hipp_out_layer_act,
                                                allow_input_downcast=True)

    func_update_hipp_net = theano.function([X_data],
                                            [hipp_out_layer_act,
                                            hipp_hid_layer_act],
                                            updates=hipp_updates,
                                            allow_input_downcast=True)
    return func_feed_forward_hipp_net, func_update_hipp_net

# #############################################################################
# ##################### build cortical networks ###############################
# #############################################################################
def build_cort_network(X_data, net_type='i'):
    cort_l1 = lasagne.layers.InputLayer(shape=(None, 15),
                                       input_var=X_data,
                                       W=Constant(0.0),
                                       b=Constant(0.0))
    cort_l2 = lasagne.layers.DenseLayer(cort_l1, num_units=40,
                                   nonlinearity=lasagne.nonlinearities.rectify)
    cort_l3 = lasagne.layers.DenseLayer(cort_l2, num_units=1,
                                   nonlinearity=lasagne.nonlinearities.sigmoid)
    cort_hid_layer_act, cort_out_layer_act = lasagne.layers.get_output([cort_l2,
                                                                        cort_l3])
    return cort_hid_layer_act, cort_out_layer_act

def train_cort_network():
    # TODO add a parameter for 'intact' or 'lesion' to drive a logic statement
    # remove trainable tag so that lower layer weights are not trained
    cort_l2.params[cort_l2.W].remove('trainable')
    cort_l2.params[cort_l2.b].remove('trainable')
    # get the parameters, trainable=True only returns parameters that can be trained
    cort_l3_params = lasagne.layers.get_all_params(cort_l3, trainable=True)
    # output_layer_activation = actual output of the network, i.e. what output is
    # targets is the supervised output, i.e. what output should be
    cort_upper_layer_loss = binary_crossentropy(cort_out_layer_act, targets)
    cort_lower_layer_loss = binary_crossentropy(cort_hid_layer_act,
                                                hipp_hid_layer_act)
    # TODO convert hipp_hid_layer_act to same shape as cort_hid_layer_act
    cort_upper_layer_loss = aggregate(cort_upper_layer_loss, mode='mean')
    cort_grads = theano.grad(cort_upper_layer_loss, wrt=cort_l3_params)
    cort_updates = lasagne.updates.adam(cort_loss, cort_params, LEARN_RATE)
    return cort_updates

def run_cort_network():
    func_feed_forward_cort_net = theano.function([X_data],
                                                [cort_out_layer_act,
                                                 cort_hid_layer_act],
                                                allow_input_downcast=True)
    func_update_cort_net = theano.function([X_data],
                                        [cort_out_layer_act,
                                         cort_hid_layer_act],
                                        updates=cort_updates,
                                        allow_input_downcast=True)

# #############################################################################
# ##################### build funcs to run networks ###########################
# #############################################################################



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
                                                  hipp_net_details.output_layer_activation,
                                                  int(cs_index[0]))
    func_feed_forward_hipp_net, func_update_hipp_net = build_functions_hipp_network(X_data,
                                                        hipp_net_details.output_layer_activation,
                                                        hipp_net_details.hidden_layer_activation,
                                                        hipp_net_updates)
    for epoch in range(NUM_OF_TRIALS):
        func_feed_forward_hipp_net(input_vector)
        hipp_raw_batch_out_act, hipp_raw_batch_hid_act = func_update_hipp_net(input_vector)
        hipp_net_raw_output_list.append(hipp_raw_batch_out_act)
        hipp_net_raw_hidden_list.append(list(hipp_raw_batch_hid_act))

    # TODO run individual functions for everything
    # TODO build hipp network, build cort network, then train both,
    # TODO then run both
    pp.pprint(hipp_net_raw_output_list)
    print(input_vector)
    print(output_targets)
    print(cs_index[0])
