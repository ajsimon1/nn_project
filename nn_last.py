import lasagne
import theano
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theano.tensor as T

from lasagne.layers import DenseLayer, InputLayer
from lasagne.objectives import binary_crossentropy, squared_error

# ######################## Define Constants #################################
N_CS = 5
N_CONTEXT = 10
N_SAMPLES = 25
N_BATCHES = 250
N_SIMS = 2
output_file = 'output.xlsx'
# ######################### Create Datasets #################################
# create dataset with a conditioned stimulus (CS) and context.  the CS has
# 5 elements and the context has 10 creating an input vector of 15 total
# for the purposes of this model the number of samples will be 25 creating
# a total input shape of (25,15)

def build_dataset(n_cs=N_CS, n_context=N_CONTEXT, n_samples=N_SAMPLES):
    # build out cs portion of input var
    cs = [[0 for i in range(n_cs)] for j in range(n_samples)]
    rand_num = np.random.randint(0, high=len(cs))
    cs[rand_num][0] = 1.0

    # build out context portion of input var
    context = [float(np.random.randint(0, high=2)) for i in range(n_context)]

    # build input var
    input_var = []
    for array_item in cs:
        input_var.append(array_item + context)
    return np.asarray(input_var)

def build_targets(input_var):
    targets = []
    for item in input_var:
        if np.any(item[0] == 1.0):
            targets.append(1.0)
        else:
            targets.append(0.0)
    cs_index = targets.index( 1.)
    return np.asarray(targets), cs_index

# ################## Build Lower Cortical Network #############################
def build_cort_low_net(input_var=None):
    l_input = lasagne.layers.InputLayer(shape=(None, 15),
                                        input_var=input_var)
    l_output = lasagne.layers.DenseLayer(
                l_input,
                num_units=40,
                W=lasagne.init.Uniform(range=3.0),
                nonlinearity=lasagne.nonlinearities.rectify)
    return l_output

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

def iter_net(num_batches, forward_func, update_func, data):
    # instatiate empty lists that are needed
    raw_output_list = []
    raw_hidden_list = []
    for batch in range(num_batches):
        forward_func(data)
        raw_hid_value, raw_out_value = update_func(data)
        raw_hidden_list.append(raw_hid_value)
        raw_output_list.append(raw_out_value)
    return raw_hidden_list, raw_output_list

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
        c_dist = np.absolute(np.subtract(np.asarray(cort_abs_list[item]), np.asarray(cort_pres_list[item])))
        c_dist_list.append(np.sum(c_dist))
    for item in range(len(hipp_pres_list)):
        h_dist = np.absolute(np.subtract(np.asarray(hipp_abs_list[item]), np.asarray(hipp_pres_list[item])))
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

def run_nets(model='i', **kwargs):
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
    print('Building networks based on {} model type, {} simulation...'.format(model_dict[str(model)], kwargs['count']))
    cort_low_out_layer = build_cort_low_net(input_var=X_data_cort_low)
    cort_up__out_layer = build_cort_up_net(input_var=X_data_cort_up)
    hipp_hid_layer, hipp_out_layer = build_hipp_net(input_var=X_data_hipp)
    cort_low_out_formula, cort_up_out_formula = lasagne.layers.get_output([cort_low_out_layer, cort_up_out_layer])
    hipp_hid_formula, hipp_out_formula = lasagne.layers.get_output([hipp_hid_layer, hipp_out_layer])
    cort_loss = lasagne.objectives.binary_crossentropy(cort_out_formula, kwargs['targets'])
    cort_loss = lasagne.objectives.aggregate(cort_loss, mode='mean')
    hipp_loss = lasagne.objectives.squared_error(hipp_out_formula, kwargs['input_var']).mean()
    # branching point for different models based on model type
    if model == 'i':
        cort_params = lasagne.layers.get_all_params(cort_out_layer, trainable=True)
        cort_grads = theano.grad(cort_loss, wrt=cort_params)
        cort_updates = lasagne.updates.adam(cort_grads, cort_params, learning_rate=0.1)
        hipp_params = lasagne.layers.get_all_params(hipp_out_layer, trainable=True)
        hipp_updates = lasagne.updates.momentum(hipp_loss, hipp_params, learning_rate=0.05, momentum=0.9)
        feed_forward_cort = theano.function([X_data_cort], [cort_hid_formula, cort_out_formula], allow_input_downcast=True)
        back_update_cort = theano.function([X_data_cort], [cort_hid_formula, cort_out_formula], updates=cort_updates, allow_input_downcast=True)
        feed_forward_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], allow_input_downcast=True)
        back_update_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], updates=hipp_updates, allow_input_downcast=True)
        cort_hidd_list, cort_out_list = iter_net(N_BATCHES, feed_forward_cort, back_update_cort, kwargs['input_var'])
        hipp_hidd_list, hipp_out_list = iter_net(N_BATCHES, feed_forward_hipp, back_update_hipp, kwargs['input_var'])
        cort_us_present_out_list, cort_us_absent_out_list = find_us_absent_present(kwargs['index'], cort_out_list)
        cort_us_present_hid_list, cort_us_absent_hid_list, hipp_us_present_hid_list, hipp_us_absent_hid_list = get_hid_abs_value(kwargs['index'], cort_hidd_list, hipp_hidd_list)
        c_dist, h_dist = get_hamm_dist(cort_us_absent_hid_list, cort_us_present_hid_list, hipp_us_absent_hid_list, hipp_us_present_hid_list)
        net_output = create_dataframe(cort_us_absent_out_list, cort_us_present_out_list, c_dist, h_dist)
    elif model == 'p':
        cort_params = lasagne.layers.get_all_params(cort_out_layer, trainable=True)
        cort_grads = theano.grad(cort_loss, wrt=cort_params)
        cort_updates = lasagne.updates.adam(cort_grads, cort_params, learning_rate=0.1)
        hipp_params = lasagne.layers.get_all_params(hipp_out_layer, trainable=True)
        hipp_updates = lasagne.updates.momentum(hipp_loss, hipp_params, learning_rate=1.0, momentum=0.9)
        feed_forward_cort = theano.function([X_data_cort], [cort_hid_formula, cort_out_formula], allow_input_downcast=True)
        back_update_cort = theano.function([X_data_cort], [cort_hid_formula, cort_out_formula], updates=cort_updates, allow_input_downcast=True)
        feed_forward_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], allow_input_downcast=True)
        back_update_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], updates=hipp_updates, allow_input_downcast=True)
        cort_hidd_list, cort_out_list = iter_net(N_BATCHES, feed_forward_cort, back_update_cort, kwargs['input_var'])
        hipp_hidd_list, hipp_out_list = iter_net(N_BATCHES, feed_forward_hipp, back_update_hipp, kwargs['input_var'])
        cort_us_present_out_list, cort_us_absent_out_list = find_us_absent_present(kwargs['index'], cort_out_list)
        cort_us_present_hid_list, cort_us_absent_hid_list, hipp_us_present_hid_list, hipp_us_absent_hid_list = get_hid_abs_value(kwargs['index'], cort_hidd_list, hipp_hidd_list)
        c_dist, h_dist = get_hamm_dist(cort_us_absent_hid_list, cort_us_present_hid_list, hipp_us_absent_hid_list, hipp_us_present_hid_list)
        net_output = create_dataframe(cort_us_absent_out_list, cort_us_present_out_list, c_dist, h_dist)
    elif model == 's':
        cort_params = lasagne.layers.get_all_params(cort_out_layer, trainable=True)
        cort_grads = theano.grad(cort_loss, wrt=cort_params)
        cort_updates = lasagne.updates.adam(cort_grads, cort_params, learning_rate=0.1)
        hipp_params = lasagne.layers.get_all_params(hipp_out_layer, trainable=True)
        hipp_updates = lasagne.updates.momentum(hipp_loss, hipp_params, learning_rate=.005, momentum=0.9)
        feed_forward_cort = theano.function([X_data_cort], [cort_hid_formula, cort_out_formula], allow_input_downcast=True)
        back_update_cort = theano.function([X_data_cort], [cort_hid_formula, cort_out_formula], updates=cort_updates, allow_input_downcast=True)
        feed_forward_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], allow_input_downcast=True)
        back_update_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], updates=hipp_updates, allow_input_downcast=True)
        cort_hidd_list, cort_out_list = iter_net(N_BATCHES, feed_forward_cort, back_update_cort, kwargs['input_var'])
        hipp_hidd_list, hipp_out_list = iter_net(N_BATCHES, feed_forward_hipp, back_update_hipp, kwargs['input_var'])
        cort_us_present_out_list, cort_us_absent_out_list = find_us_absent_present(kwargs['index'], cort_out_list)
        cort_us_present_hid_list, cort_us_absent_hid_list, hipp_us_present_hid_list, hipp_us_absent_hid_list = get_hid_abs_value(kwargs['index'], cort_hidd_list, hipp_hidd_list)
        c_dist, h_dist = get_hamm_dist(cort_us_absent_hid_list, cort_us_present_hid_list, hipp_us_absent_hid_list, hipp_us_present_hid_list)
        net_output = create_dataframe(cort_us_absent_out_list, cort_us_present_out_list, c_dist, h_dist)
    else:
        cort_hid_layer.params[cort_hid_layer.W].remove('trainable')
        cort_hid_layer.params[cort_hid_layer.b].remove('trainable')
        cort_params = lasagne.layers.get_all_params(cort_out_layer, trainable=True)
        cort_grads = theano.grad(cort_loss, wrt=cort_params)
        cort_updates = lasagne.updates.adam(cort_grads, cort_params, learning_rate=0.1)
        hipp_params = lasagne.layers.get_all_params(hipp_out_layer, trainable=True)
        hipp_updates = lasagne.updates.momentum(hipp_loss, hipp_params, learning_rate=.005, momentum=0.9)
        feed_forward_cort = theano.function([X_data_cort], [cort_hid_formula, cort_out_formula], allow_input_downcast=True)
        back_update_cort = theano.function([X_data_cort], [cort_hid_formula, cort_out_formula], updates=cort_updates, allow_input_downcast=True)
        feed_forward_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], allow_input_downcast=True)
        back_update_hipp = theano.function([X_data_hipp], [hipp_hid_formula, hipp_out_formula], updates=hipp_updates, allow_input_downcast=True)
        cort_hidd_list, cort_out_list = iter_net(N_BATCHES, feed_forward_cort, back_update_cort, kwargs['input_var'])
        hipp_hidd_list, hipp_out_list = iter_net(N_BATCHES, feed_forward_hipp, back_update_hipp, kwargs['input_var'])
        cort_us_present_out_list, cort_us_absent_out_list = find_us_absent_present(kwargs['index'], cort_out_list)
        cort_us_present_hid_list, cort_us_absent_hid_list, hipp_us_present_hid_list, hipp_us_absent_hid_list = get_hid_abs_value(kwargs['index'], cort_hidd_list, hipp_hidd_list)
        c_dist, h_dist = get_hamm_dist(cort_us_absent_hid_list, cort_us_present_hid_list, hipp_us_absent_hid_list, hipp_us_present_hid_list)
        net_output = create_dataframe(cort_us_absent_out_list, cort_us_present_out_list, c_dist, h_dist)

    return net_output

def find_criterion(df, column, threshold):
    try:
        crit = df.loc[df[str(column)] >= threshold].index.values[0]
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

def create_output(df_list, filename):
    df_concat = pd.concat(df_list)
    df_concat_by_index = df_concat.groupby(df_concat.index)
    df_final = df_concat_by_index.mean().round(decimals=2)
    xl_writer = pd.ExcelWriter('output.xlsx')
    df_final.to_excel(xl_writer, 'Sheet1')
    xl_writer.save()
    return df_final

if __name__ == '__main__':
    user_response = input('Select model type: intact(i), lesion(l), phystogimine(p), scopolomine(s): ')
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
    # df_final = create_output(df_list, output_file)
    print(df_final)
    # df_final.plot()
    # plt.show()
