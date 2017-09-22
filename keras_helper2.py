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
def create_cs(num_elem_in_cs):
    """
    create cs that represents elements 0-4 in 15 element input vector, the
    elements are set to '0' initially and a 'US present' CS will be set with
    another function
    arguments:
        num_elem_in_cs --> number of elements in the CS
    return --> x element vector equal to num_elem_in_cs argument [x, x, x,...]
    """
    cs_vector = []
    for i in range(num_elem_in_cs):
        cs_vector.append(0)
        # 2 indicates values in input vector can only be 1 or 0
    return cs_vector

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

    return np.asarray(trial_dataset)
