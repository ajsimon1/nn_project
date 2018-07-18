## A computational model of learning acquisition based on empirical results of animal responses in the eye blink classical conditioning paradigm

This computational model aim to mimic empirical results seen in animal models during the acquisition of a learned behavior via classical conditioning.  The conditioning paradigm involves training a rabbit to respond with a reflexive eyeblink to a conditioned stimulus, usually a tone.  This is accomplished by pairing the tone with an unconditioned stimulus, a puff of air, that naturally produces the reflexive eyeblink.  The acquisition can be seen by the below ***learning curve***.  

##### TODO add learning curve image from empirical study

In code, the empirical data can be mimicked by building a simple
multi-layer perceptron *(MLP)*, linked to an autoencoder. The purpose of 2 networks is beyond the scope of the post, but suffice it to say it more closely resembles the inner workings of the animal model.  On to the code.

The source code can be found at [this](https://github.com/ajsimon1/nn_project) repo...

Or you can clone by `git clone https://github.com/ajsimon1/nn_project.git`,

and

`pip install -r requirements.txt`

Installing the via *requirements.txt* will install jupyter notebooks
as well, which is not explicitly declared in the import statements.

Before we get started, I want to apologize to all my PEP8 peoples, during the meat of the program the line lengths extend beyond 88 chars and I apologize for that.  I can assure you, when time allows, the code will be cleaned to compliance.

We start by importing the necessary packages. Please note the Theano library has ceased development at the start of 2018.  A quick rundown of the used packages:
- [lasagne](http://lasagne.readthedocs.io/en/stable/index.html) :: The deep learning framework that provides the abstraction layer over top of theano.  I picked this library specifically because it allowed easy access to hidden layer activations and weights matrices.  This is usually not needed for mainstream use cases, but for research based use cases it is critical
- [theano](http://deeplearning.net/software/theano/) :: A library that allows for efficient mathematical operations involving multi-dimensional arrays.  While lasagne provides a layer of abstraction for defining and building the network architecture the actual functionality is written directly in theano functions.
- [numpy](http://www.numpy.org/) :: For this project numpy was used for its random module as well as some manipulation of multi-dimensional arrays
- [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) :: Used to organize the final data and export to excel or csv
- [matplotlib](https://pandas.pydata.org/pandas-docs/stable/index.html) :: create final charts

```python
import datetime
import lasagne
import theano

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import theano.tensor as T
```
Constants are set at the start of program to control the size of the input vector as well as the number of batches per simulation and total simulations.
```python
# ######################## Define Constants #################################
N_CS = 5 # num of elements in conditioned stimulus (CS)
N_CONTEXT = 10 # num of elements in context
N_SAMPLES = 25 # num of vectors within the input dataset
N_BATCHES = 250 # num of datasets present in a single pass through the network
N_SIMS = 20 # num of forward and backward iterations
output_file = 'network_output' # name of csv/xlsx file
```
