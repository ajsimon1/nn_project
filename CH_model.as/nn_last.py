import lasagne
import pprint
import theano
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theano.tensor as T

from lasagne.layers import DenseLayer, InputLayer
from lasagne.objectives import binary_crossentropy, aggregate

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
    l_input = lasagne.layers.InputLayer(shape=(input_var.shape),
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
    l_input = lasagne.layers.InputLayer(shape=(input_var.shape),
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

def run_nets(model='i', **kwargs):
    # define theano shared variables for both networks
    X_data_cort = T.matrix('X_data_cort')
    X_data_hipp = T.matrix('X_data_hipp')
  
    # create nn models
    print('Building Cortical & Hippocampal networks...')
    cort_hidden_formula, cort_out_formula = build_cort_net(X_data_cort)
    hipp_hidden_formula, hipp_out_formula = build_hipp_net(X_data_hipp)

if __name__ == '__main__':
    
    print('Building datasets...')
    input_var = build_dataset()
    print('Dataset built )s :')
    print(input_var)
    print('Building targets based on dataset...')
    targets, cs_index = build_targets(input_var)
    print('Targets built as: ')
    print(targets)
    print('The CS has built into the input vector at {} element'.format(cs_index + 1))
    print('Cortical net hidden layer formula is {}'.format(cort_hidden_formula))
    print('Cortical net output layer formula is {}'.format(cort_out_formula))
    print('Hippocampal net hidden layer formula is {}'.format(hipp_hidden_formula))
    print('Hippocampal net hidden layer formula is {}'.format(hipp_out_formula))
