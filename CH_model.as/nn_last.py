import lasagne
import pprint
import theano
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theano.tensor as T

from lasagne.layers import DenseLayer, InputLayer
from lasagne.objectives import binary_crossentropy, squared_error

# ######################## Define Constants #################################
N_CS = 5
N_CONTEXT = 10
N_SAMPLES = 25

# ######################### Create Datasets #################################
# create dataset with a conditioned stimulus (CS) and context.  the CS has
# 5 elements and the context has 10 creating an input vector of 15 total
# for the purposes of this model the number of samples will be 25 creating
# a total input shape of (25,15)

def build_dataset(n_cs=N_CS, n_context=N_CONTEXT, n_samples=N_SAMPLES):
    # build out cs portion of input var
    cs = [[0 for i in range(n_cs)] for j in range(n_samples)]
    rand_num = np.random.randint(0, high=len(cs))
    cs[rand_num][0] = 1.0

    # build out context portion of input var
    context = [float(np.random.randint(0, high=2)) for i in range(n_context)]

    # build input var
    input_var = []
    for array_item in cs:
        input_var.append(array_item + context)
    return np.asarray(input_var)

def build_targets(input_var):
    targets = []
    for item in input_var:      
        if np.any(item[0] == 1.0):
            targets.append(1.0)
        else:
            targets.append(0.0)
    cs_index = targets.index( 1.)
    return np.asarray(targets), cs_index

# ################## Build Cortical Network #################################
def build_cort_net(input_var=None):
    l_input = lasagne.layers.InputLayer(shape=(None, 15),
                                        input_var=input_var)
    l_hidden = lasagne.layers.DenseLayer(
                l_input,
                num_units=40,
                nonlinearity=lasagne.nonlinearities.rectify)
    l_output = lasagne.layers.DenseLayer(
                l_hidden,
                num_units=1,
                nonlinearity=lasagne.nonlinearities.sigmoid)
    return l_hidden, l_output         

# #################### Build Hippocampal Network ############################
def build_hipp_net(input_var=None):
    l_input = lasagne.layers.InputLayer(shape=(None, 15),
                                        input_var=input_var)
    l_hidden = lasagne.layers.DenseLayer(
            l_input,
            num_units=8,
            nonlinearity=lasagne.nonlinearities.rectify)                                        
    l_output = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=15,
            nonlinearity=lasagne.nonlinearities.sigmoid)
    return l_hidden, l_output            

def iter_net(num_batches, forward_func, update_func, data):
    # instatiate empty lists that are needed
    raw_output_list = []
    raw_hidden_list = []
    for batch in range(num_batches):
        forward_func(data)
        raw_hid_value, raw_out_value = update_func(np.asarray(data))
        raw_hidden_list.append(raw_hid_value)
        raw_output_list.append(raw_out_value)
    return raw_hidden_list, raw_output_list

def run_nets(model='i', **kwargs):
    # define theano shared variables for both networks
    X_data_cort = T.matrix('X_data_cort')
    y_cort = T.vector('y_cort')
    X_data_hipp = T.matrix('X_data_hipp')
    y_hipp = T.vector('y_hipp')
    # create nn models
    print('Building networks based on {} model type...'.format(model))
    cort_hid_layer, cort_out_layer = build_cort_net(input_var=X_data_cort)
    hipp_hid_layer, hipp_out_layer = build_hipp_net(input_var=X_data_hipp)
    cort_hid_formula, cort_out_formula = lasagne.layers.get_output([cort_hid_layer, cort_out_layer])
    hipp_hid_formula, hipp_out_formula = lasagne.layers.get_output([hipp_hid_layer, hipp_out_layer])
    cort_loss = lasagne.objectives.binary_crossentropy(cort_out_formula, kwargs['targets']).mean()
    hipp_loss = lasagne.objectives.squared_error(hipp_out_formula, kwargs['input_var']).mean()
    # branching point for different models based on model type
    if model == 'i':
        cort_params = lasagne.layers.get_all_params([cort_out_layer], trainable=True)
        cort_updates = lasagne.updates.adam(cort_loss, cort_params, learning_rate=0.1)
        hipp_params = lasagne.layers.get_all_params([hipp_out_layer], trainable=True)
        hipp_updates = lasagne.updates.momentum(hipp_loss, hipp_params, learning_rate=0.05, momentum=0.9)
        feed_forward_cort = theano.function([X_data_cort], [cort_hid_formula, cort_out_formula], allow_input_downcast=True)
        back_update_cort = theano.function([X_data_cort], [cort_hid_formula, cort_out_formula], updates=cort_updates, allow_input_downcast=True)
        feed_forward_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], allow_input_downcast=True)
        back_update_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], updates=hipp_updates, allow_input_downcast=True)
        cort_hidd_list, cort_out_list = iter_net(N_SAMPLES, feed_forward_cort, back_update_cort, kwargs['input_var'])
        hipp_hidd_list, hipp_out_list = iter_net(N_SAMPLES, feed_forward_hipp, back_update_hipp, kwargs['input_var'])
    return hipp_hidd_list, hipp_out_list, cort_hidd_list, cort_out_list
if __name__ == '__main__':
    model_dict = {
        'i': 'intact',
        'l': 'lesion',
        's': 'scopolamine',
        'p': 'physostigmine',
        }
    user_response = input('Select model type: intact(i), lesion(l), phystogimine(p), scopolomine(s): ')
    print('Building datasets...')
    input_var = build_dataset()
    print('Dataset built as :')
    print(input_var)
    print('Building targets based on dataset...')
    targets, cs_index = build_targets(input_var)
    print('Targets built as: ')
    print(targets)
    print('The CS has built into the input vector at {} element'.format(cs_index + 1))
    print('Building nets based on {} model type'.format(model_dict[str(user_response)]))
    print(run_nets(model=user_response, targets=targets, input_var=input_var))
