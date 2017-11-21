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

# create context of data using CONSTANT NUM_OF_CONTEXT
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
        # 2 indicates values in input vector can only be 1 or 0
    return context_vector

# create cs for input using CONSTANTS NUM_OF_BATCHES and NUM_OF_CS
def create_cs(num_of_batches, num_elem_in_cs):
    """
    create the cs to chain to context, cs is a 5 element vector with 1 vector
    having '1' to inditace that a 'US' is expected
    """
    cs = [[0 for i in range(num_elem_in_cs)] for j in range(num_of_batches)]
    rand_number = np.random.randint(0, high=len(cs))
    cs[rand_number][0] = 1.0
    return cs

# create the input vector based off CONSTANTS defined by user
def create_input_vector(num_of_batches, num_elem_in_cs, num_elem_in_context):
    context_vector = create_context(num_elem_in_context)
    cs_vector = create_cs(num_of_batches, num_elem_in_cs)
    input_vector = []
    for array_item in cs_vector:
        input_vector.append(array_item + context_vector)
    return np.asarray(input_vector)

# create targets based on 1st element in input vector
# vector with '1' in 1st elemnt spot should get a '1' in the targets vector
# '0' in the input vector gets '0' in the targets vector
def create_targets(input_vector):
    targets = []
    for item in input_vector:
        if np.any(item[0] == 1.0):
            targets.append([1.0])
        else:
            targets.append([0.0])
    return np.asarray(targets)
