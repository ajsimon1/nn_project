"""
created 9/20/17

author: adam

file to manage output of network
need to create pyplot as well as excel output
"""

import theano
import theano.tensor as T
import numpy as np
import lasagne
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.objectives import binary_crossentropy, aggregate
from nn_proj_utils import create_input_vector, create_targets
import pprint
import matplotlib.pyplot as plt

# TODO write function to create pyplot
# TODO write function to create other plot
# TODO write function to create excel file
