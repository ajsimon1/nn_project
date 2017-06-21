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

def create_cs(num_cs):
    """
    create dataset of 1 and 0 to feed into network, the cs element is a 5
    element vector that represnts the conditioned stimulus the network is
    responding to.  the cs will change with every new trial.  at run
    time the cs is coupled with the context to input into the network
        arguments
        num_cs: number of conditioned stimuli in each input vector
        batch_size: number of total input vectors for the network
    """
    cs_data_set = []
    for i in range(num_cs):
        cs_data_set.append(np.random.randint(0, high=2))
        # 2 indicates values in input vector can only be 1 or 0
    return cs_data_set

def create_input(cs_data_set, context_data_set):
    # returns combined cs and context for input to netwrk
    return list(cs_data_set + context_data_set)
    # NOTE: create_input must be run on indexed data sets, otherwise entire
    # dataset will be run

"""
def create_train_test_data(data_set, train_percent):
    pass
    splitting the data set into percentages for training and testing purposes
        arguments:
        dataset: complete data set to split
        train_percent: the percentage of the dataset to use for training
                        the remaining data will be used for testing
"""

"""create dataset 250, 15 with last column for labels"""
# create context first, this will be the same for all 250 samples
context = create_context(10)

def create_dataset(context, int_len_dataset):
    dataset = []
    for i in range(int_len_dataset + 1):
        temp_list = []
        for cs_int in range(5):
            temp_list.append(np.random.randint(0, high=2))
        temp_list2 = list(itertools.chain(temp_list, context))
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
