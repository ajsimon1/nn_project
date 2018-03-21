"""
created 9/20/17

author: adam

hour vested to date (11/18/17) --> ~ 70
"""
import theano
import theano.tensor as T
import numpy as np
import lasagne
import matplotlib.pyplot as plt
from lasagne.objectives import binary_crossentropy, aggregate
from lasagne.init import Constant
from DEPRECATED_nn_proj_utils import create_input_vector, create_targets
import pprint
import pandas as pd

np.set_printoptions(threshold=np.nan)
pp = pprint.PrettyPrinter()

# define variables
# X_data = T.col('X_data') # data
X_data = T.matrix('X_data') # data
Y_targets = T.matrix('Y_targets') # targets
# CONSTANTS
NUM_OF_CS = 5 # the variable conditioned stimuls coupled with the context
NUM_OF_CONTEXT = 10 # the immutable context that reflects the 'environment'
# CS & CONTEXT are combined to equal an 'input vector'
NUM_OF_TRIALS = 25 # trial is a collection of input vector
NUM_OF_BATCHES = 250 # batch is a collection of trials
LEARN_RATE = 0.01 # learning rate to update the network after error backprop
NUM_OF_SIMS = 20

##########################*** Build out networks ***###########################
# create input vector based on CONSTANTS provided by user
data = create_input_vector(NUM_OF_TRIALS, NUM_OF_CS, NUM_OF_CONTEXT)
# create targets based on length of input vector
# match 'US' target with input vector containing '1' as first element
targets = create_targets(data)
# identify where in the array the us was present, looking for '1.'
cs_index = np.where(targets==1.)
# #############################################################################
# ##################### build cort network ####################################
# #############################################################################
cort_l1 = lasagne.layers.InputLayer(shape=(None,15),input_var=X_data)
# hidden layer has 40 units and rectifier activation function equivalent to
# f(x) = max(0, x) where x is the input to the neuron
cort_l2 = lasagne.layers.DenseLayer(cort_l1, num_units=40,
                               nonlinearity=lasagne.nonlinearities.rectify)
# output layer with 1 unit and sigmoid activation function to limit output
# to values between 1 and 0, sigmoid funcion is defined as
# f(x) = 1 / 1 + e**-x
cort_l3 = lasagne.layers.DenseLayer(cort_l2, num_units=1,
                               nonlinearity=lasagne.nonlinearities.sigmoid)

# function that defines how data show move through the network
cort_hid_layer_act, cort_out_layer_act = lasagne.layers.get_output([cort_l2,
                                                                    cort_l3])
# formula for above:
# 'sigmoid((((TensorConstant{0.5} * (((input \\dot W) + b) +
# |((input \\dot W) + b)|)) \\dot W) + b))'
# executing the function that pushes input data (x) through the system using
# equation defined by Y, Y then becomes the actual output, ie output layer
# activations.  also grabbing hidden layer activations for hamming distance
# last variable allows lasagne to automatically downcast any higher bit dtype
# to a lower dtype, a float64 to a float32 for example
func_feed_forward_cort_net = theano.function([X_data],
                                            [cort_out_layer_act,
                                             cort_hid_layer_act],
                                            allow_input_downcast=True)
# get the parameters, trainable=True only returns parameters that can be trained
# TODO break this up to get_all_params from [cort_l3] & [cort_l2]
# cort_l2.params[cort_l2.W].remove('trainable')
# cort_l2.params[cort_l2.b].remove('trainable')
cort_params = lasagne.layers.get_all_params([cort_l3], trainable=True)
# output_layer_activation = actual output of the network, i.e. what output is
# targets is the supervised output, i.e. what output should be
cort_loss = binary_crossentropy(cort_out_layer_act, targets[cs_index[0]])
# TODO cort_loss = binary_crossentropy(cort_hid_layer_act, hipp_hid_layer_act)
# TODO convert hipp_hid_layer_act to same shape as cort_hid_layer_act
cort_loss = aggregate(cort_loss, mode='mean') # mean loss across all batches
# get the gradient of a loss function with respect to these parameters
cort_grads = theano.grad(cort_loss, wrt=cort_params)
# using built in stochasitc gradient descent 'sgd'
# in below function, this will 'update' the network
cort_updates = lasagne.updates.adam(cort_loss, cort_params, learning_rate=0.5)
# function for error backpropagation and updating the network paramters
func_update_cort_net = theano.function([X_data],
                                    [cort_out_layer_act,
                                     cort_hid_layer_act],
                                    updates=cort_updates,
                                    allow_input_downcast=True)
# #############################################################################
# ##################### build hipp network ####################################
# #############################################################################
hipp_l1 = lasagne.layers.InputLayer(shape=(None, 15), input_var=X_data)

hipp_l2 = lasagne.layers.DenseLayer(hipp_l1,
                                        num_units=8,
                                        nonlinearity=lasagne.nonlinearities.rectify)

hipp_l3 = lasagne.layers.DenseLayer(hipp_l2,
                                        num_units=15,
                                        nonlinearity=lasagne.nonlinearities.sigmoid)

hipp_hid_layer_act, hipp_out_layer_act = lasagne.layers.get_output([hipp_l2,
                                                                    hipp_l3])

hipp_params = lasagne.layers.get_all_params(hipp_l3, trainable=True)

hipp_loss = lasagne.objectives.squared_error(hipp_l1.input_var,
                                            hipp_out_layer_act).mean()
hipp_updates = lasagne.updates.momentum(hipp_loss,
                                        hipp_params,
                                        learning_rate=0.05,
                                        momentum=0.9)

func_feed_forward_hipp_net = theano.function([X_data],
                            hipp_out_layer_act,
                            allow_input_downcast=True)

func_update_hipp_net = theano.function([X_data],
                        [hipp_out_layer_act,
                        hipp_hid_layer_act],
                        updates=hipp_updates,
                        allow_input_downcast=True)

# ############################################################################
# ######################### run hipp network #################################
# ############################################################################
hipp_net_raw_output_list = []
hipp_net_raw_hidden_list = []
hipp_net_sims_output_list = []
hipp_net_sims_hidden_list = []
for sim in range(NUM_OF_SIMS):
    for epoch in range(NUM_OF_BATCHES):
        func_feed_forward_hipp_net(data)
        hipp_raw_batch_out_act, hipp_raw_batch_hid_act = func_update_hipp_net(data)
        hipp_net_raw_output_list.append(hipp_raw_batch_out_act)
        hipp_net_raw_hidden_list.append(list(hipp_raw_batch_hid_act))
    hipp_net_sims_hidden_list.append(hipp_net_raw_hidden_list)
    hipp_net_sims_output_list.append(hipp_net_raw_output_list)
# ############################################################################
# ######################### run cort network #################################
# ############################################################################
cort_net_batch_output_list = []
cort_net_batch_hidden_list = []
cort_net_raw_output_list = []
cort_net_raw_hidden_list = []
cort_net_sims_hidden_list = []
cort_net_sims_output_list = []
for sim in range(NUM_OF_SIMS):
    for epoch in range(NUM_OF_BATCHES):
        func_feed_forward_cort_net(data)
        cort_raw_output_activation, cort_raw_hidden_activation = func_update_cort_net(data)
        cort_net_raw_output_list.append(cort_raw_output_activation)
        cort_net_raw_hidden_list.append(list(cort_raw_hidden_activation))
    cort_net_sims_hidden_list.append(cort_net_raw_hidden_list)
    cort_net_sims_output_list.append(cort_net_raw_output_list)

cort_net_us_present_output_list = []
cort_net_us_absent_output_list = []
hipp_net_us_present_output_list = []
hipp_net_us_absent_output_list = []
# pull out us present actication and us absent activation
for item in cort_net_raw_output_list:
    cort_net_us_present_output_list.append(float(item[cs_index[0]]))
    try:
        cort_net_us_absent_output_list.append(float(item[cs_index[0] + 1]))
    except IndexError:
        cort_net_us_absent_output_list.append(float(item[cs_index[0] - 1]))
# grab hidden layer activations
cort_net_us_present_hidden_layer_activations_list = []
cort_net_us_absent_hidden_layer_activations_list = []
hipp_net_us_present_hidden_layer_activations_list = []
hipp_net_us_absent_hidden_layer_activations_list = []

for activ_list in cort_net_raw_hidden_list:
    cort_net_us_present_hidden_layer_activations_list.append(list(map(lambda x: abs(x), activ_list[cs_index[0]])))
    try:
        cort_net_us_absent_hidden_layer_activations_list.append(list(map(lambda x: abs(x), activ_list[cs_index[0] + 1])))
    except IndexError:
        cort_net_us_absent_hidden_layer_activations_list.append(list(map(lambda x: abs(x), activ_list[cs_index[0] - 1])))
for activ_list in hipp_net_raw_hidden_list:
    hipp_net_us_present_hidden_layer_activations_list.append(list(map(lambda x: abs(x), activ_list[cs_index[0]])))
    try:
        hipp_net_us_absent_hidden_layer_activations_list.append(list(map(lambda x: abs(x), activ_list[cs_index[0] + 1])))
    except IndexError:
        hipp_net_us_absent_hidden_layer_activations_list.append(list(map(lambda x: abs(x), activ_list[cs_index[0] - 1])))
# build out c_dist
c_dist_list = []
h_dist_list = []
# add a loop to go through all
for i in range(len(cort_net_us_present_hidden_layer_activations_list)):
    c_dist = np.subtract(np.asarray(cort_net_us_absent_hidden_layer_activations_list[i]),
                   np.asarray(cort_net_us_present_hidden_layer_activations_list[i]))
    c_dist_list.append(np.sum(c_dist))
for i in range(len(hipp_net_us_present_hidden_layer_activations_list)):
    h_dist = np.subtract(np.asarray(hipp_net_us_absent_hidden_layer_activations_list[i]),
                   np.asarray(hipp_net_us_present_hidden_layer_activations_list[i]))
    h_dist_list.append(np.sum(h_dist))

# ############################################################################
# ############################### final data #################################
# ############################################################################

final_data = {'X':cort_net_us_absent_output_list,
              'XA':cort_net_us_present_output_list,
              'C-Dist':c_dist_list,
              'H-Dist': h_dist_list}
# dump output into dataframe
network_output_df = pd.DataFrame(final_data,columns=final_data.keys()).round(3)
# set index to start at 1 instead of 0
network_output_df.index = np.arange(1, len(network_output_df) + 1)
def find_criterion(df, column, threshold):
    try:
        criterion = df.loc[df[str(column)]>=threshold].index.values[0]
    except IndexError:
        return 'Criterion not reached'
    return criterion
'''
for intact, lesion
    hipp net was 0.05 learning rate with 0.9 momentum and cortical was 0.5 on
    upper weights and 0.1 on lower weights (same on lesioned)
    decreased the learning rate tenfold for US absent runs
for scolp / phys
    only made changes to hipp network;
    for scolp --> decreased to .005 for us present and .0005 for us absent
    for phys --> increased to  1.0 on us present and 0.1 on us absent

'''

# print('Criterion reached at block: {}'.format(find_criterion(network_output_df, 'XA', .91)))
print(network_output_df)
# pp.pprint(hipp_net_raw_output_list)
# pp.pprint(cort_net_raw_output_list)
# print(data)
# pp.pprint(hipp_net_raw_hidden_list[0])
#network_output_df.loc[:,['X','XA']].plot()
# plt.show()
# print(cort_out_layer_act)
# print(targets[cs_index[0]])
# print(cort_hid_layer_act)
