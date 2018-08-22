import datetime
import lasagne
import theano

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import theano.tensor as T

# ######################## Define Constants #################################
N_CS = 5 # num of elements in conditioned stimulus (CS)
N_CONTEXT = 10 # num of elements in context
N_SAMPLES = 25 # num of vectors within the input dataset
N_BATCHES = 250 # num of datasets present in a single pass through the network
N_SIMS = 20 # num of forward and backward iterations
output_file = 'network_output' # name of csv/xlsx file

# ######################### Create Data #####################################
def build_dataset(n_cs=N_CS, n_context=N_CONTEXT, n_samples=N_SAMPLES):
    """ Build input dataset using constants specified at start of application
        as defaults
        Returns -> ndarray of shape (n_samples, (n_cs + n_context))
    """
    # create 2d vector of zeroes of shape (n_samples, n_cs)
    cs = [[0 for i in range(n_cs)] for j in range(n_samples)]
    rand_num = np.random.randint(0, high=len(cs))
    # assign the first element of a random vector as 1, this respresents the
    # vector which the network responds with the unconditioned resopnse
    cs[rand_num][0] = 1.0
    # build context as random binary values of length n_context
    context = [float(np.random.randint(0, high=2)) for i in range(n_context)]
    input_var = []
    # add an indetical context vetor for each cs vector, creating input dataset
    for array_item in cs:
        input_var.append(array_item + context)
    return np.asarray(input_var)

def build_targets(input_var):
    """ Build target vector serve as prediction values during training
        Return -> ndarray of shape (n_samples, 1)
        Return -> cs_index indicating which vector in the dataset contains the
        expected unconditioned response
    """
    targets = []
    # using the input var to ensure shape consistency, a 1 is added to the
    # target vector at an index matching the input dataset
    for item in input_var:
        if np.any(item[0] == 1.0):
            targets.append(1.0)
        else:
            targets.append(0.0)
            # the index of the vector expecting the unconditioned response is
            # noted and returned
    cs_index = targets.index(1.)
    return np.asarray(targets), cs_index

# ################## Build Lower Cortical Network #############################
def build_cort_low_net(input_var=None):
    """ Lower cortical network represents the hidden layer in normal MLP
        architecture.  The output will be passed to the upper cortical network
        The weights are trained using the hidden layer output of the Hippocampal
        network
        Return -> output layer formula
    """
    # input shape ('None',15) allows for variable input constrained by size of
    # input datset
    l_input = lasagne.layers.InputLayer(shape=(None, 15),
                                        input_var=input_var)
    # 'hidden' layer contains 40 nodes randomly set to values between -3.0
    # and 3.0.  activation functions is rectify to optimize performance
    l_output = lasagne.layers.DenseLayer(
                l_input,
                num_units=40,
                W=lasagne.init.Uniform(range=3.0),
                nonlinearity=lasagne.nonlinearities.rectify)
    return l_output
# TODO start here
# ################## Build Upper Cortical Network #############################
def build_cort_up_net(input_var=None):
    l_input = lasagne.layers.InputLayer(shape=(None, 40),
                                        input_var=input_var)
    l_output = lasagne.layers.DenseLayer(
                l_input,
                num_units=1,
                W=lasagne.init.Uniform(range=3.0),
                nonlinearity=lasagne.nonlinearities.sigmoid)
    return l_output

# #################### Build Hippocampal Network ############################
def build_hipp_net(input_var=None):
    l_input = lasagne.layers.InputLayer(shape=(None, 15),
                                        input_var=input_var)
    l_hidden = lasagne.layers.DenseLayer(
            l_input,
            num_units=8,
            W=lasagne.init.Uniform(range=3.0),
            nonlinearity=lasagne.nonlinearities.rectify)
    l_output = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=15,
            nonlinearity=lasagne.nonlinearities.sigmoid)
    return l_hidden, l_output

def iter_hipp_net(num_batches, forward_func, update_func, data):
    # instatiate empty lists that are needed
    raw_output_list = []
    raw_hidden_list = []
    for batch in range(num_batches):
        forward_func(data)
        raw_hid_value, raw_out_value = update_func(data)
        raw_hidden_list.append(raw_hid_value)
        raw_output_list.append(raw_out_value)
    return raw_hidden_list, raw_output_list

def iter_cort_net(num_batches, forward_func, update_func, data):
    # instatiate empty lists that are needed
    raw_out_list = []
    for batch in range(num_batches):
        forward_func(data)
        raw_out_value = update_func(data)
        raw_out_list.append(raw_out_value)
    return raw_out_list

def find_us_absent_present(index, out_list):
    us_present_list = []
    us_absent_list = []
    for item in out_list:
        us_present_list.append(float(item[index]))
        try:
            us_absent_list.append(float(item[index + 1]))
        except IndexError:
            us_absent_list.append(float(item[index - 1]))
    return us_present_list, us_absent_list

def get_hid_abs_value(index, cort_list, hipp_list):
    cort_us_absent_list = []
    cort_us_present_list = []
    hipp_us_present_list = []
    hipp_us_absent_list = []
    for item in cort_list:
        cort_us_present_list.append(list(map(lambda x: abs(x), item[index])))
        try:
            cort_us_absent_list.append(list(map(lambda x: abs(x), item[index + 1])))
        except IndexError:
            cort_us_absent_list.append(list(map(lambda x: abs(x), item[index - 1])))
    for item in hipp_list:
        hipp_us_present_list.append(list(map(lambda x: abs(x), item[index])))
        try:
            hipp_us_absent_list.append(list(map(lambda x: abs(x), item[index + 1])))
        except IndexError:
            hipp_us_absent_list.append(list(map(lambda x: abs(x), item[index - 1])))
    return cort_us_present_list, cort_us_absent_list, hipp_us_present_list, hipp_us_absent_list

def get_hamm_dist(cort_abs_list, cort_pres_list, hipp_abs_list, hipp_pres_list):
    c_dist_list = []
    h_dist_list = []
    for item in range(len(cort_pres_list)):
        c_dist = np.absolute(np.subtract(np.asarray(cort_pres_list[item]), np.asarray(cort_abs_list[item])))
        c_dist_list.append(np.sum(c_dist))
    for item in range(len(hipp_pres_list)):
        h_dist = np.absolute(np.subtract(np.asarray(hipp_pres_list[item]), np.asarray(hipp_abs_list[item])))
        h_dist_list.append(np.sum(h_dist))
    return c_dist_list, h_dist_list

def create_dataframe(cort_us_abs, cort_us_pres, c_dist, h_dist):
    final_data = {
        'X': cort_us_abs,
        'XA': cort_us_pres,
        'C-Dist': c_dist,
        'H-Dist': h_dist,
        }
    net_output = pd.DataFrame(final_data, columns=final_data.keys())
    return net_output

def convert_hipp_hidd_layer(hipp_hidd_list):
    return_list = [list(x) * 5 for x in hipp_hidd_list[len(hipp_hidd_list) - 1]]
    # print(return_list)
    return return_list

def run_nets(model='i', **kwargs):
    start_time = datetime.datetime.now()
    model_dict = {
        'i': 'intact',
        'l': 'lesion',
        's': 'scopolamine',
        'p': 'physostigmine',
        }
    # define theano shared variables for both networks
    X_data_cort_low = T.matrix('X_data_cort_low')
    X_data_cort_up = T.matrix('X_data_cort_up')
    X_data_hipp = T.matrix('X_data_hipp')
    # create nn models
    print('Building networks based on {} model type, {} ' \
            'simulation...'.format(model_dict[str(model)], kwargs['count']))
    cort_low_out_layer = build_cort_low_net(input_var=X_data_cort_low)
    cort_low_out_formula = lasagne.layers.get_output(cort_low_out_layer)
    cort_up_out_layer = build_cort_up_net(input_var=X_data_cort_up)
    cort_up_out_formula = lasagne.layers.get_output(cort_up_out_layer)
    hipp_hid_layer, hipp_out_layer = build_hipp_net(input_var=X_data_hipp)
    hipp_hid_formula, hipp_out_formula = lasagne.layers.get_output([hipp_hid_layer,
                                                                    hipp_out_layer])
    # branching point for different models based on model type
    if model == 'i':
        hipp_loss = lasagne.objectives.squared_error(hipp_out_formula, kwargs['input_var']).mean()
        hipp_params = lasagne.layers.get_all_params(hipp_out_layer, trainable=True)
        hipp_updates = lasagne.updates.momentum(hipp_loss, hipp_params, learning_rate=0.05, momentum=0.5)
        feed_forward_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], allow_input_downcast=True)
        back_update_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], updates=hipp_updates, allow_input_downcast=True)
        hipp_hidd_list, hipp_out_list = iter_hipp_net(N_BATCHES, feed_forward_hipp, back_update_hipp, kwargs['input_var'])
        cort_low_targets = convert_hipp_hidd_layer(hipp_hidd_list)
        cort_up_loss = lasagne.objectives.binary_crossentropy(cort_up_out_formula, kwargs['targets'])
        cort_up_loss = lasagne.objectives.aggregate(cort_up_loss, mode='mean')
        cort_low_loss = lasagne.objectives.squared_error(cort_low_out_formula, cort_low_targets).mean()
        cort_up_params = lasagne.layers.get_all_params(cort_up_out_layer, trainable=True)
        cort_up_grads = theano.grad(cort_up_loss, wrt=cort_up_params)
        cort_up_updates = lasagne.updates.sgd(cort_up_loss, cort_up_params, learning_rate=0.5)
        cort_low_params = lasagne.layers.get_all_params(cort_low_out_layer, trainable=True)
        cort_low_grads = theano.grad(cort_low_loss, wrt=cort_low_params)
        cort_low_updates = lasagne.updates.sgd(cort_low_loss, cort_low_params, learning_rate=0.1)
        feed_forward_cort_low = theano.function([X_data_cort_low], cort_low_out_formula, allow_input_downcast=True)
        feed_forward_cort_up = theano.function([X_data_cort_up], cort_up_out_formula, allow_input_downcast=True)
        back_update_cort_low = theano.function([X_data_cort_low], cort_low_out_formula, updates=cort_low_updates, allow_input_downcast=True)
        back_update_cort_up = theano.function([X_data_cort_up], cort_up_out_formula, updates=cort_up_updates, allow_input_downcast=True)
        cort_low_out_list = iter_cort_net(N_BATCHES, feed_forward_cort_low, back_update_cort_low, kwargs['input_var'])
        cort_up_out_list = iter_cort_net(N_BATCHES, feed_forward_cort_up, back_update_cort_up, cort_low_out_list[len(cort_low_out_list)-1])
        cort_us_present_up_out_list, cort_us_absent_up_out_list = find_us_absent_present(kwargs['index'], cort_up_out_list)
        cort_us_present_low_out_list, cort_us_absent_low_out_list, hipp_us_present_hid_list, hipp_us_absent_hid_list = get_hid_abs_value(kwargs['index'], cort_low_out_list, hipp_hidd_list)
        c_dist, h_dist = get_hamm_dist(cort_us_absent_low_out_list, cort_us_present_low_out_list, hipp_us_absent_hid_list, hipp_us_present_hid_list)
        net_output = create_dataframe(cort_us_absent_up_out_list, cort_us_present_up_out_list, c_dist, h_dist)
    elif model == 'p':
        hipp_loss = lasagne.objectives.squared_error(hipp_out_formula, kwargs['input_var']).mean()
        hipp_params = lasagne.layers.get_all_params(hipp_out_layer, trainable=True)
        hipp_updates = lasagne.updates.momentum(hipp_loss, hipp_params, learning_rate=1.0, momentum=0.5)
        feed_forward_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], allow_input_downcast=True)
        back_update_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], updates=hipp_updates, allow_input_downcast=True)
        hipp_hidd_list, hipp_out_list = iter_hipp_net(N_BATCHES, feed_forward_hipp, back_update_hipp, kwargs['input_var'])
        cort_low_targets = convert_hipp_hidd_layer(hipp_hidd_list)
        cort_up_loss = lasagne.objectives.binary_crossentropy(cort_up_out_formula, kwargs['targets'])
        cort_up_loss = lasagne.objectives.aggregate(cort_up_loss, mode='mean')
        cort_low_loss = lasagne.objectives.squared_error(cort_low_out_formula, cort_low_targets).mean()
        cort_up_params = lasagne.layers.get_all_params(cort_up_out_layer, trainable=True)
        cort_up_grads = theano.grad(cort_up_loss, wrt=cort_up_params)
        cort_up_updates = lasagne.updates.sgd(cort_up_loss, cort_up_params, learning_rate=0.5)
        cort_low_params = lasagne.layers.get_all_params(cort_low_out_layer, trainable=True)
        cort_low_grads = theano.grad(cort_low_loss, wrt=cort_low_params)
        cort_low_updates = lasagne.updates.sgd(cort_low_loss, cort_low_params, learning_rate=0.1)
        feed_forward_cort_low = theano.function([X_data_cort_low], cort_low_out_formula, allow_input_downcast=True)
        feed_forward_cort_up = theano.function([X_data_cort_up], cort_up_out_formula, allow_input_downcast=True)
        back_update_cort_low = theano.function([X_data_cort_low], cort_low_out_formula, updates=cort_low_updates, allow_input_downcast=True)
        back_update_cort_up = theano.function([X_data_cort_up], cort_up_out_formula, updates=cort_up_updates, allow_input_downcast=True)
        cort_low_out_list = iter_cort_net(N_BATCHES, feed_forward_cort_low, back_update_cort_low, kwargs['input_var'])
        cort_up_out_list = iter_cort_net(N_BATCHES, feed_forward_cort_up, back_update_cort_up, cort_low_out_list[len(cort_low_out_list)-1])
        cort_us_present_up_out_list, cort_us_absent_up_out_list = find_us_absent_present(kwargs['index'], cort_up_out_list)
        cort_us_present_low_out_list, cort_us_absent_low_out_list, hipp_us_present_hid_list, hipp_us_absent_hid_list = get_hid_abs_value(kwargs['index'], cort_low_out_list, hipp_hidd_list)
        c_dist, h_dist = get_hamm_dist(cort_us_absent_low_out_list, cort_us_present_low_out_list, hipp_us_absent_hid_list, hipp_us_present_hid_list)
        net_output = create_dataframe(cort_us_absent_up_out_list, cort_us_present_up_out_list, c_dist, h_dist)
    elif model == 's':
        hipp_loss = lasagne.objectives.squared_error(hipp_out_formula, kwargs['input_var']).mean()
        hipp_params = lasagne.layers.get_all_params(hipp_out_layer, trainable=True)
        hipp_updates = lasagne.updates.momentum(hipp_loss, hipp_params, learning_rate=0.0001, momentum=0.5)
        feed_forward_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], allow_input_downcast=True)
        back_update_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], updates=hipp_updates, allow_input_downcast=True)
        hipp_hidd_list, hipp_out_list = iter_hipp_net(N_BATCHES, feed_forward_hipp, back_update_hipp, kwargs['input_var'])
        cort_low_targets = convert_hipp_hidd_layer(hipp_hidd_list)
        cort_up_loss = lasagne.objectives.binary_crossentropy(cort_up_out_formula, kwargs['targets'])
        cort_up_loss = lasagne.objectives.aggregate(cort_up_loss, mode='mean')
        cort_low_loss = lasagne.objectives.squared_error(cort_low_out_formula, cort_low_targets).mean()
        cort_up_params = lasagne.layers.get_all_params(cort_up_out_layer, trainable=True)
        cort_up_grads = theano.grad(cort_up_loss, wrt=cort_up_params)
        cort_up_updates = lasagne.updates.sgd(cort_up_loss, cort_up_params, learning_rate=0.5)
        cort_low_params = lasagne.layers.get_all_params(cort_low_out_layer, trainable=True)
        cort_low_grads = theano.grad(cort_low_loss, wrt=cort_low_params)
        cort_low_updates = lasagne.updates.sgd(cort_low_loss, cort_low_params, learning_rate=0.1)
        feed_forward_cort_low = theano.function([X_data_cort_low], cort_low_out_formula, allow_input_downcast=True)
        feed_forward_cort_up = theano.function([X_data_cort_up], cort_up_out_formula, allow_input_downcast=True)
        back_update_cort_low = theano.function([X_data_cort_low], cort_low_out_formula, updates=cort_low_updates, allow_input_downcast=True)
        back_update_cort_up = theano.function([X_data_cort_up], cort_up_out_formula, updates=cort_up_updates, allow_input_downcast=True)
        cort_low_out_list = iter_cort_net(N_BATCHES, feed_forward_cort_low, back_update_cort_low, kwargs['input_var'])
        cort_up_out_list = iter_cort_net(N_BATCHES, feed_forward_cort_up, back_update_cort_up, cort_low_out_list[len(cort_low_out_list)-1])
        cort_us_present_up_out_list, cort_us_absent_up_out_list = find_us_absent_present(kwargs['index'], cort_up_out_list)
        cort_us_present_low_out_list, cort_us_absent_low_out_list, hipp_us_present_hid_list, hipp_us_absent_hid_list = get_hid_abs_value(kwargs['index'], cort_low_out_list, hipp_hidd_list)
        c_dist, h_dist = get_hamm_dist(cort_us_absent_low_out_list, cort_us_present_low_out_list, hipp_us_absent_hid_list, hipp_us_present_hid_list)
        net_output = create_dataframe(cort_us_absent_up_out_list, cort_us_present_up_out_list, c_dist, h_dist)
    elif model == 'l':
        cort_low_out_layer.params[cort_low_out_layer.W].remove('trainable')
        cort_low_out_layer.params[cort_low_out_layer.b].remove('trainable')
        hipp_loss = lasagne.objectives.squared_error(hipp_out_formula, kwargs['input_var']).mean()
        hipp_params = lasagne.layers.get_all_params(hipp_out_layer, trainable=True)
        hipp_updates = lasagne.updates.momentum(hipp_loss, hipp_params, learning_rate=0.05, momentum=0.5)
        feed_forward_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], allow_input_downcast=True)
        back_update_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], updates=hipp_updates, allow_input_downcast=True)
        hipp_hidd_list, hipp_out_list = iter_hipp_net(N_BATCHES, feed_forward_hipp, back_update_hipp, kwargs['input_var'])
        cort_low_targets = convert_hipp_hidd_layer(hipp_hidd_list)
        cort_up_loss = lasagne.objectives.binary_crossentropy(cort_up_out_formula, kwargs['targets'])
        cort_up_loss = lasagne.objectives.aggregate(cort_up_loss, mode='mean')
        cort_low_loss = lasagne.objectives.squared_error(cort_low_out_formula, cort_low_targets).mean()
        cort_up_params = lasagne.layers.get_all_params(cort_up_out_layer, trainable=True)
        cort_up_grads = theano.grad(cort_up_loss, wrt=cort_up_params)
        cort_up_updates = lasagne.updates.sgd(cort_up_loss, cort_up_params, learning_rate=0.5)
        cort_low_params = lasagne.layers.get_all_params(cort_low_out_layer, trainable=True)
        cort_low_grads = theano.grad(cort_low_loss, wrt=cort_low_params)
        cort_low_updates = lasagne.updates.sgd(cort_low_loss, cort_low_params, learning_rate=0.1)
        feed_forward_cort_low = theano.function([X_data_cort_low], cort_low_out_formula, allow_input_downcast=True)
        feed_forward_cort_up = theano.function([X_data_cort_up], cort_up_out_formula, allow_input_downcast=True)
        back_update_cort_low = theano.function([X_data_cort_low], cort_low_out_formula, updates=cort_low_updates, allow_input_downcast=True)
        back_update_cort_up = theano.function([X_data_cort_up], cort_up_out_formula, updates=cort_up_updates, allow_input_downcast=True)
        cort_low_out_list = iter_cort_net(N_BATCHES, feed_forward_cort_low, back_update_cort_low, kwargs['input_var'])
        cort_up_out_list = iter_cort_net(N_BATCHES, feed_forward_cort_up, back_update_cort_up, cort_low_out_list[len(cort_low_out_list)-1])
        cort_us_present_up_out_list, cort_us_absent_up_out_list = find_us_absent_present(kwargs['index'], cort_up_out_list)
        cort_us_present_low_out_list, cort_us_absent_low_out_list, hipp_us_present_hid_list, hipp_us_absent_hid_list = get_hid_abs_value(kwargs['index'], cort_low_out_list, hipp_hidd_list)
        c_dist, h_dist = get_hamm_dist(cort_us_absent_low_out_list, cort_us_present_low_out_list, hipp_us_absent_hid_list, hipp_us_present_hid_list)
        net_output = create_dataframe(cort_us_absent_up_out_list, cort_us_present_up_out_list, c_dist, h_dist)
    build_time = datetime.datetime.now() - start_time
    print('...............Total build time for simulation {} was {} seconds'.format(kwargs['count'], build_time.seconds))
    return net_output

def find_criterion(df, column, threshold, model):
    try:
        crit = df.loc[df[str(column)] >= threshold].index.values[0]
        print('Threshold {} for {} model reached at block {}'.format(threshold, model, crit))
    except IndexError:
        return 'Criterion not reached'
    return crit

def run_sims(num_sims, **kwargs):
    df_list = []
    for sim in range(num_sims):
        df = run_nets(model=kwargs['model'],
                        targets=kwargs['targets'],
                        input_var=kwargs['input_var'],
                        index=kwargs['index'],
                        count=int(sim))
        df_list.append(df)
    return df_list

def create_output(df_list, filename, filetype, model):
    df_concat = pd.concat(df_list)
    df_concat_by_index = df_concat.groupby(df_concat.index)
    df_final = df_concat_by_index.mean().round(decimals=2)
    mod_suf = '_' + str(model)
    find_criterion(df_final, 'XA', 0.9, model)
    if filetype == 'xls':
        xl_writer = pd.ExcelWriter(filename + mod_suf + '.xlsx')
        df_final.to_excel(xl_writer, 'Sheet1')
        xl_writer.save()
    elif filetype == 'csv':
        df_final.to_csv(filename + mod_suf + '.csv')
    df_final[['X', 'XA']].plot()
    return df_final

if __name__ == '__main__':
    user_response = input('Select model type: intact(i), lesion(l), phystogimine(p), scopolomine(s): ')
    user_filetype = input('Select file type: CSV(csv), Excel(xls): ')
    print('Building datasets...')
    input_var = build_dataset()
    print('Dataset built as :')
    print(input_var)
    print('Building targets based on dataset...')
    targets, cs_index = build_targets(input_var)
    print('Targets built as: ')
    print(targets)
    print('The CS has built into the input vector at {} element'.format(cs_index + 1))
    df_list = run_sims(N_SIMS,
                        model=user_response,
                        targets=targets,
                        input_var=input_var,
                        index=cs_index)
    df_final = create_output(df_list, output_file, user_filetype, user_response)
    plt.show()
