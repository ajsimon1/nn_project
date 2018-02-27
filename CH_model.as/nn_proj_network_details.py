"""
author: adam

created on: 11/21/17

this file houses functions specific to the network
"""
import theano
import lasagne

def build_cort_network(input_vector_variable):
    """
    layer units are hardocded right now, but should be set as variables for
    flexability.
    arguments:
        input_vector --> theano.matrix object # note: this is a shared variable
                        and not actual data
    return:
        network_output --> formula of the network, to be used to calcualte
                        output as the network is currently hardocded, the output
                        formula is sigmoid((((TensorConstant{0.5} *
                                    (((input \\dot W) + b) +
                                    ((input \\dot W) + b)|)) \\dot W) + b))
        network_parameters --> paramters of the network, if trainable is true
                                only return parameters that are trainable, i.e.
                                the weights matrix, returns a list
    """
    # the input layer expects a 15 element vector, batch size is inconsequential
    #  input vector equals the defined tensor variable above, a matrix like dtype
    l1 = lasagne.layers.InputLayer(shape=(None, 15),
                                   input_var=input_vector_variable)
    # hidden layer has 40 units and rectifier activation function equivalent to
    # f(x) = max(0, x) where x is the input to the neuron
    l2 = lasagne.layers.DenseLayer(l1, num_units=40,
                                   nonlinearity=lasagne.nonlinearities.rectify)
    # output layer with 1 unit and sigmoid activation function to limit output
    # to values between 1 and 0, sigmoid funcion is defined as
    # f(x) = 1 / 1 + e**-x
    l3 = lasagne.layers.DenseLayer(l2, num_units=1,
                                   nonlinearity=lasagne.nonlinearities.sigmoid)
    # function that defines how data show move through the network
    network_output = lasagne.layers.get_output(l3)
    # begin to gather paramters to update with loss/error function
    # get the parameters, trainable=True only returns parameters that can be trained
    # only calling the output layer is necessary, all the trainable paramters
    # going backwards to input will be gathered also
    network_parameters = lasagne.layers.get_all_params(l3, trainable=True)
    return network_output, network_parameters


def define_network_updates(network_targets,
                            network_output,
                            network_parameters,
                            learning_rate=0.01):
    """
    create the update function to perform backward propagation to train the
    network
    arguments:
        network_targets --> array like object of training targets expected
                            for the network, this will compated against the
                            netowrk_ouput to determine loss function
        network_output --> formula for the network to produce the predictions
                            this value is returned by the build_cort_network
                            function
        network_parameters --> trainable parameters of the network, i.e the
                                weights matrix and/or biases, this is returned
                                by the build_cort_network function
        learning_rate --> an argument for the lasagne updates function,
                            lasagne.adam(), this is defaulted to the industry
                            standard 0.01.  this should be changed during
                            experiment to see how it can affect learning
    return:
        network_gradient --> the gradient value for the backward propagation
        network_updates --> lasagne updates value to pass into the backward
                            propogation function
    """
    # calculating loss based on cross entropy of actual output vs targets
    # binary cross entropy is the industry standard loss function for binary
    # classification
    loss = lasagne.objectives.binary_crossentropy(network_output,
                                                  network_targets)
    # calculating the mean of all loss calculations for the batch size
    loss = lasagne.objectives.aggregate(loss,
                                        mode='mean')
    # compute the derivative of loss with regards to the network_parameters
    # this is used in the following formula to update the netwrok parameters
    # gradients = theano.grad(loss, wrt=network_parameters)
    # create a return value to pass to the run_network function
    # updates contains the necessary updates to the network parameters based
    # backward propagation
    updates = lasagne.updates.adam(loss,
                                   network_parameters,
                                   learning_rate)
    return updates, loss


def run_network(input_vector_variable,
                output_formula,
                network_updates,
                actual_input_data,
                loss,
                batches=1,):
    """
    defining and running 2 functions that :: 1) feed information
    forward through the network (func_feed_data_through_nn), 2) update the
    network through backward propagation (func_update_network)
    arguments:
        input_vector
        output_formula
        network_updates
        actual_input_data
        batches --> input variable, default set to 1
        loss --> loss function defined in define_network_updates
    return:
        actual_network_data
        actual_network_loss
    """
    func_feed_data_through_nn = theano.function([input_vector_variable],
                                                output_formula,
                                                allow_input_downcast=True,
                                                on_unused_input='ignore',)
    func_update_network = theano.function([input_vector_variable],
                                            [output_formula, loss],
                                            updates=network_updates,
                                            allow_input_downcast=True,
                                            on_unused_input='ignore',)
    network_loss_per_batch_list = []
    network_output_per_batch_list = []
    network_loss_per_batch_variable = []
    network_output_per_batch_variable = []
    for item in range(batches):
        func_feed_data_through_nn(actual_input_data)
        network_output_per_batch_variable, network_loss_per_batch_variable = func_update_network(actual_input_data)
        network_output_per_batch_list.append(network_output_per_batch_variable)
        network_loss_per_batch_list.append(network_loss_per_batch_variable)
    return network_loss_per_batch_list, network_output_per_batch_list
