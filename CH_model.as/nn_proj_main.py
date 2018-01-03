"""
created 9/20/17

author: adam

hour vested to date (11/18/17) --> ~ 45

main file to run the network TODO consider changing name to run.py
this file calls in the other modular files to take in use input and
produce output
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
