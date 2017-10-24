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
import pydot
import graphviz
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Reshape
from keras import initializers, optimizers, losses, utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.callbacks import NBatchLogger

from keras_helper2 import create_context, create_cs, create_trial, prepare_data

"""To do, start working through new procedure w/ github intro
https://github.com/fastforwardlabs/keras-hello-world/blob/master/kerashelloworld.ipynb
"""
# define constants
# in website app these should come from UI
NUM_CS = 5
NUM_CONTEXT = 10
NUM_BATCH = 25
EPOCHS = 100

# set seed for reproducability
seed = 7
np.random.seed(seed)

# create dataset w/o target
# target dataset should be a n X 15 python based array
data_trial_array = create_trial(NUM_BATCH, NUM_CONTEXT, NUM_CS)

# add US as 16th element of array, applicable only for
# training dataset
# this function also returns the ndarray for the first time
data_final = prepare_data(data_trial_array)

# splitting dataset into the X = input(CS & context) and Y = output(US)
X = data_final[:, :15]
Y = data_final[:, -1:]


# encode the labels for y dataset to 1s and 0s
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# print(encoded_Y, X)

# build model function taken from machinelearningmastery website
# binary classification tutorial
# create model skeleton
def create_model():
    model = Sequential()
    # create layers
    model.add(Dense(40,  # neurons in layer
                    input_dim = 15,
                    name ='hidden_layer',
                    kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(1,
                    name = 'output_layer',
                    kernel_initializer = 'normal',
                    activation = 'sigmoid'))
    # need to comple the model before training
    model.compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])
    history = NBatchLogger()
    model.fit(X, encoded_Y, validation_split=.33, callbacks=[history])
    print(model.get_input_at(0))


    return model
# evaluate model without standardized dataset
estimator = KerasClassifier(build_fn=create_model,
                            nb_epoch=EPOCHS,
                            batch_size=NUM_BATCH,
                            verbose=0)

# evaluate model with non-standardized dataset
kfold = StratifiedKFold()
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# estimator holds results for epoch/batch and prints out on request
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# print(results)
# TODO write custom class to print better results, see github issue
# https://github.com/fchollet/keras/issues/2850
# https://github.com/fchollet/keras/issues/254
# TODO add variable for optimizer to adjust learning rate based on user entry
# this allows for the 2 drug related models


"""
# evaluate baseline model with standardized dataset
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model,
                                            epochs=EPOCHS,
                                            batch_size=NUM_BATCH,
                                            verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
print('Standardized: %.2f%% (%.2f%%)' % (results.mean()*100, results.std()*100))
"""
