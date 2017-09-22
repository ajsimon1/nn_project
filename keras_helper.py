# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:26:27 2017

@author: adam

helper functions for keras nn
"""
import numpy as np
import itertools
import random

# create the input vector
def create_context(num_context):
    """
    create a dataset of 1 and 0 to feed into network, the context element
    is a 10 element vector that represents environmental context while training
    optimally it is held contanst for the duration of an epoch
    the context is coupled with the cs to create a 15 element vector at run time
        arguments
        num_context: number of context elements in each input vector
        batch_size: number of total input vectors for the network
    """
    context_data_set = []
    for i in range(num_context):
        context_data_set.append(np.random.randint(0, high=2))
        # 2 indicates values in input vector can only be 1 or 0
    return context_data_set

def create_dataset(context, int_len_dataset):
    """creating the multidimensional array of 200 x 15"""
    dataset = []
    for i in range(int_len_dataset + 1): # creates the 200 part of loop
        temp_list = []
        for cs_int in range(5): # creates the cs
            temp_list.append(np.random.randint(0, high=2))
            # appends cs to context creating single vector
        temp_list2 = list(itertools.chain(temp_list, context))
        # above adds single vectors to multi D array
        dataset.append(temp_list2)
    return dataset

"""set random item in dataset as receiving US"""
def set_us(dataset):
    index_with_us = random.randrange(len(dataset))
    for lists in dataset:
        if dataset.index(lists) == index_with_us:
            lists.append('US')
        else:
            lists.append('No US')
    return dataset

"""need to id what the index is of the US in the dataset then add the identical
sequence to the training or testing set, dpending on where the US is.  example
if the US is in index 101 and the dataset is split at 125 then the copy
needs to get added to above 125"""
def copy_us(dataset):
    for item in dataset:
        if 'US' in item:
            us_index = dataset.index(item)
            break
        else:
            continue
    mid_dataset = len(dataset) / 2
    if us_index < mid_dataset:
        copy_index = mid_dataset + 10
        dataset[int(copy_index)].pop()
        dataset[int(copy_index)].append('US')
        print('Original US Index located at {}, so copy US set at {}'.format(us_index, copy_index))
    else:
        copy_index = mid_dataset - 10
        dataset[int(copy_index)].pop()
        dataset[int(copy_index)].append('US')
        print('Original US Index located at {}, so copy US set at {}'.format(us_index, copy_index))
    return dataset
    #
