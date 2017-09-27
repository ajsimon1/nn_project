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
from keras import initializers, optimizers, losses, utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from keras_helper import create_context, create_dataset, set_us, copy_us

"""To do, start working through new procedure w/ github intro
https://github.com/fastforwardlabs/keras-hello-world/blob/master/kerashelloworld.ipynb
"""
# define constants
NUM_CS = 5
NUM_CONTEXT = 10
NUM_BATCH = 250
EPOCHS = 100

# set seed for reproducability
seed = 7
np.random.seed(seed)

# create dataset w/o target
data_trial_array = create_trial(NUM_BATCH, NUM_CONTEXT, NUM_CS)

# clean dataset
# TO DO
data_final = prepare_data(data_trial_array)

# change to dataset_final if using copy_us function
X = data_array[:, :15]
Y = data_array[:, -1:]

# encode the labels for y dataset to 1s and 0s
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print(encoded_Y, X)
# split data for training / testing
# train_X, test_X, train_y, test_y =
#train_test_split(X, y, train_size=0.5, random_state=0)

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
    return model

# evaluate model without standardized dataset
estimator = KerasClassifier(build_fn=create_model,
                            nb_epoch=EPOCHS,
                            batch_size=NUM_BATCH,
                            verbose=0)
# evaluate model with non-standardized dataset
kfold = StratifiedKFold()
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

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
are we training the network on a single 'CS' then passing the CS to it?
do we need to return what the 'CS' is while using set_us function?
to we demonstrate that we can train the network then feed the network and have
its predictability be 100%
"""
