"""
attempt at research project with keras library
started 3/22/2017
adam simon
coninuted 4/21/2017, hours to date = approx. 10

the network is a model of the cotrical / cerebellur network
there are 15 inputs nodes that received an input vector, with 5 nodes defined
as the conditioned stimulus (CS) and 10 nodes deinfed as the environmental
context.  the input network is initialized with random values of 1 & 0
the lower layer weights between the input and hidden layers are fixed and
only updated by the hippocampal hidden layer.  when the HHL is missing the
weights are fixed.  as such the input vector is passed directly to the cortical
hidden layer, where there is an activation function.
"""
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Reshape
from keras import initializers
from keras import optimizers
from keras import losses
from keras import utils

from keras_helper import create_context, create_dataset, set_us

"""To do, start working through new procedure w/ github intro
https://github.com/fastforwardlabs/keras-hello-world/blob/master/kerashelloworld.ipynb
"""
# create optimizers
sgd = optimizers.SGD()

# create losses
msqe = losses.mean_squared_error

# define constants
NUM_CS = 5
NUM_CONTEXT = 10
NUM_BATCH = 250

# create dataset
context = create_context(NUM_CONTEXT)
dataset = create_dataset(context, NUM_BATCH)

# set US for 1 input set, this is set by random
dataset_final = set_us(dataset)

# clean dataset
data_array = np.array(dataset_final)
X = data_array[:, :15]
y = data_array[:, -1:]

# split data for training / testing
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, random_state=0)

# also need to add the US value to the training set as well


# no do something similar with keras part

# create model skeleton
model = Sequential()
# create layers
model.add(Dense(units=40,  # neurons in layer
                input_shape=(15,),
                name='hidden_layer',
                kernel_initializer=initializers.random_uniform(minval=-3.0,
                                                                maxval=3.0)))
model.add(Activation(activation='sigmoid'))
model.add(Dense(units=2,
                name='output_layer',
                ))
model.add(Activation(activation='softmax'))

# need to comple the model before training
model.compile(optimizer=sgd, loss=msqe,
              metrics=['accuracy'])
