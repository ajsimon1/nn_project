from sklearn.neural_network import MLPRegressor
import numpy as np
from keras_helper import create_context


X = np.array([1,15])
y = np.ndarray([1,])

clf = MLPRegressor(solver='lbfgs',
                   hidden_layer_sizes=(1,40),
                   random_state=0,
                   )

clf.fit(X,y)
