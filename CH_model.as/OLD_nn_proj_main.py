"""
created 9/20/17

author: adam

hour vested to date (1/3/18) --> ~ 55

main file to run the network TODO consider changing name to run.py
this file calls in the other modular files to take in use input and
produce output

note for author, lasagne is the correct environment
"""
from nn_proj_input import (accept_user_input, create_input_vector, 
                           create_targets, create_init_vector)
from collections import namedtuple
from sklearn.model_selection import train_test_split
import theano.tensor as T
import pprint
from nn_proj_network_details import (build_cort_network, define_network_updates,
                                     run_network)
from nn_proj_output import create_dataframe, create_series, create_pyplot_line
import numpy

numpy.set_printoptions(threshold=numpy.nan)

# instantiating pprint for testing
pp = pprint.PrettyPrinter()

# variable declaration
constants_named_tuple = namedtuple('Constants', ['NUM_OF_CS',
                                           'NUM_OF_CONTEXT',
                                           'NUM_OF_BATCHES',
                                           'NUM_OF_TRIALS',
                                           'LEARNING_RATE',],) 
# theano variables
X_data = T.matrix('X_data') # data
Y_targets = T.matrix('Y_targets') # targets

# gather user input
constants = accept_user_input()
named_constants = constants_named_tuple._make(constants)

# build initialization data to zero out network
data_init = create_init_vector(named_constants.NUM_OF_BATCHES,
                                    named_constants.NUM_OF_CS,
                                    named_constants.NUM_OF_CONTEXT)
targets_init = create_targets(data_init)

# build full set of data to split into train/test data
data = create_input_vector(named_constants.NUM_OF_BATCHES,
                                    named_constants.NUM_OF_CS,
                                    named_constants.NUM_OF_CONTEXT)
targets = create_targets(data)

pp.pprint(data)
pp.pprint(targets)

'''
# split input vector into train/test, 75% will be held for training
data_train, data_test, targets_train, targets_test = train_test_split(
        data, targets, test_size=0.50)
'''

# TODO consider intializing the data in large function called create_input
# add this to the input file to generate everything needed

# build the network, input variable is just for shape and will not be run
# until run_network function is called
cort_net_output, cort_net_parameters = build_cort_network(data)

# run training data
cort_net_updates, cort_net_loss = define_network_updates(targets_init,
                                                         cort_net_output,
                                                         cort_net_parameters)

train_live_output, train_live_loss = run_network(X_data,
                                                 cort_net_output,
                                                 cort_net_updates,
                                                 data_init,
                                                 cort_net_loss,
                                                 batches=named_constants.NUM_OF_TRIALS)

# run test data
'''
cort_net_updates, cort_net_loss = define_network_updates(targets,
                                                         cort_net_output,
                                                         cort_net_parameters)
'''
test_live_output, test_live_loss = run_network(X_data,
                                               cort_net_output,
                                               cort_net_updates,
                                               data,
                                               cort_net_loss,
                                               batches=named_constants.NUM_OF_TRIALS)

# cast train data live output into dataframe
df_train_live_output = create_dataframe(train_live_output)
# cast train data live loss into series
df_train_live_loss = create_series(train_live_loss, name='loss')
# cast test data live output into dataframe
df_test_live_output = create_dataframe(test_live_output)
# cast test data live loss into series
df_test_live_loss = create_series(test_live_loss, name='loss')
# TODO the current output is a collection of TRIAL, do you want the mean of 
# that? or do you need to plot every item on 1 graph? can you use pyplot
# to create a new line for each trial item? maybe pick a random trial of 
# the batch and put that in a list to average or something?
print(df_test_live_output.tail())

print(create_pyplot_line(df_test_live_output, 
                         x_label='batch',
                         y_label='final activation'))

# TODO dude, all the data is off, have to go through all the vectors 
# functions from the beginning
