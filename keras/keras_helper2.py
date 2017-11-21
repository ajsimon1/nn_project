"""
created 9/21/2017

author: adam

description: 2nd attempt at helper functions for cortical hippocampal neural
network.  the input should consist of at least 20 vectors per trial and least
200 trials per block.  each input vector will be 15 elements with the first
5 elements referring to the conditioned stimulus and the last 10 elements
referring to the "environmental context". the environmental context is generated
once per block and maintained for the entiritey of the block.  the cs is either
'10000' correlating to 'US present' or '00000' correltating to 'US absent'.
During training the model will be given a confirmative '1' for CS '10000' and
negative '0' for CS '00000'.  The model is expected to accurately predict
the presence '1' of a US when the CS is '10000'
"""
import numpy as np
import random
import itertools

# function to create context
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
        context_vector.append(np.random.randint(0, high=2))
        # 2 indicates values in input vector can only be 1 or 0
    return context_vector

# function to create cs
def create_cs(batches_for_run, num_elem_in_cs):
    """
    create the cs to chain to context, cs is a 5 element vector with 1 vector
    having '1' to inditace that a 'US' is expected
    """
    return np.zeroes((batches_for_run, num_elem_in_cs))

# function to create trials
def create_trial(num_of_vectors_in_trial, num_elem_in_context, num_elem_in_cs):
    """
    create the trials that will repeat through the block, need to assign
    create_context and create_cs to variables then set those variable for each
    vector in the block
    """
    trial_dataset = []
    context = create_context(num_elem_in_context)
    cs = create_cs(num_elem_in_cs)
    for i in range(num_of_vectors_in_trial):
        trial_dataset.append(list(itertools.chain(cs, context)))

    return trial_dataset

# function for median odd numbered set
def get_median(num_set):
    """quick helper to middle index of dataset"""
    return int(np.ceil(num_set) / 2 + .5)

# function to prepare data to input
def prepare_data(dataset):
    median = get_median(len(dataset))
    dataset[median][0] = 1 # set median array value 1st element to 1
    # iterate through data and assign '1' for 'US present' and '0'
    # for 'US absent' as separate list at the end
    for vector in dataset:
        if vector[0] == 1:
            vector.append("US Present")
            print("just appended 1 to vector at index {} of the dataset".format(dataset.index(vector)))
        else:
            vector.append("No US")
    return np.asarray(dataset)
"""
data set complete, final array is 25 x 16 with last item in each vector
the US present or not indicating whether the CS should predict the US
"""
