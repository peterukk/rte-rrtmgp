#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing neural network emulators of 
RRTMGP gas optics scheme

This program takes existing input-output data generated with RRTMGP and
user-specified hyperparameters such as the number of neurons, 
scales the data if requested, and trains a neural network. 

Alternatively, an automatic tuning method can be used for
finding a good set of hyperparameters (expensive).

Right now just a placeholder, pasted some of the code I used in my paper

Contributions welcome!

@author: Peter Ukkonen
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras import losses, optimizers
import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation, Flatten,Input
import numpy as np
import h5py
import tensorflow.keras.backend as K
# import optuna

from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.training.optimizer import Optimizer


class COCOB(Optimizer):
    def __init__(self, alpha=100, use_locking=False, name='COCOB'):
        '''
        constructs a new COCOB optimizer
        '''
        super(COCOB, self).__init__(use_locking, name)
        self._alpha = alpha

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                gradients_sum = constant_op.constant(0, 
                                                     shape=v.get_shape(),
                                                     dtype=v.dtype.base_dtype)
                grad_norm_sum = constant_op.constant(0, 
                                                     shape=v.get_shape(),
                                                     dtype=v.dtype.base_dtype)
                L = constant_op.constant(1e-8, shape=v.get_shape(), dtype=v.dtype.base_dtype)
                tilde_w = constant_op.constant(0.0, shape=v.get_shape(), dtype=v.dtype.base_dtype)
                reward = constant_op.constant(0.0, shape=v.get_shape(), dtype=v.dtype.base_dtype)

            self._get_or_make_slot(v, L, "L", self._name)
            self._get_or_make_slot(v, grad_norm_sum, "grad_norm_sum", self._name)
            self._get_or_make_slot(v, gradients_sum, "gradients_sum", self._name)
            self._get_or_make_slot(v, tilde_w, "tilde_w", self._name)
            self._get_or_make_slot(v, reward, "reward", self._name)

    def _apply_dense(self, grad, var):
        gradients_sum = self.get_slot(var, "gradients_sum")
        grad_norm_sum = self.get_slot(var, "grad_norm_sum")
        tilde_w = self.get_slot(var, "tilde_w")
        L = self.get_slot(var, "L")
        reward = self.get_slot(var, "reward")

        L_update = tf.maximum(L,tf.abs(grad))
        gradients_sum_update = gradients_sum + grad
        grad_norm_sum_update = grad_norm_sum + tf.abs(grad)
        reward_update = tf.maximum(reward-grad*tilde_w,0)
        new_w = -gradients_sum_update/(L_update*(tf.maximum(grad_norm_sum_update+L_update,self._alpha*L_update)))*(reward_update+L_update)
        var_update = var-tilde_w+new_w
        tilde_w_update=new_w
        
        gradients_sum_update_op = state_ops.assign(gradients_sum, gradients_sum_update)
        grad_norm_sum_update_op = state_ops.assign(grad_norm_sum, grad_norm_sum_update)
        var_update_op = state_ops.assign(var, var_update)
        tilde_w_update_op = state_ops.assign(tilde_w, tilde_w_update)
        L_update_op = state_ops.assign(L, L_update)
        reward_update_op = state_ops.assign(reward, reward_update)

        return control_flow_ops.group(*[gradients_sum_update_op,
                             var_update_op,
                             grad_norm_sum_update_op,
                             tilde_w_update_op,
                             reward_update_op,
                             L_update_op])

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)

# 1. Define an objective function to be maximized.
def create_model_hyperopt(trial, nx, ny):
    model = Sequential()
    
    # We define our MLP.
    # number of hidden layers
    n_layers = trial.suggest_int("n_layers", 1, 3)
    model = Sequential()
    # Input layer
    activ0 = trial.suggest_categorical('activation', ['relu', 'softsign'])
    num_hidden0 = trial.suggest_int("n_neurons_l0_l", 64, 256)
    model.add(Dense(num_hidden0, input_dim=nx, activation=activ0))
     
    for i in range(1, n_layers):
         num_hidden = trial.suggest_int("n_neurons_l{}".format(i), 64, 256)
         activ =trial.suggest_categorical('activation', ['relu', 'softsign']),
         model.add(Dense(num_hidden, activation=activ))
         
    # output layer
    model.add(Dense(ny, activation='linear'))
    
    # We compile our model with a sampled learning rate.
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    lossfunc    = losses.mean_squared_error
    model.compile(
        loss=lossfunc, 
        optimizer=optimizers.Adam(learning_rate=lr),
        metrics   = ['mean_absolute_error'],
        )
    return model


def create_model_mlp(nx,ny,neurons=[40,40], activ0='softsign',activ='softsign',
                 kernel_init='he_uniform',activ_last='linear'):
    model = Sequential()
    # input layer (first hidden layer)
    model.add(Dense(neurons[0], input_dim=nx, kernel_initializer=kernel_init, activation=activ0))
    # further hidden layers
    for i in range(1,np.size(neurons)):
      model.add(Dense(neurons[i], activation=activ,kernel_initializer=kernel_init))
    # output layer
    model.add(Dense(ny, activation=activ_last,kernel_initializer=kernel_init))
    
    return model


def mse_weights(y_true,y_pred):
    wg = np.array([2.0, 1.0, 2.0, 2.0], dtype=np.float32)
    # wg = np.array([2.5, 1.0, 2.0, 2.5], dtype=np.float32)

    y_true = y_true*wg
    y_pred = y_pred*wg
    return K.mean(K.square(y_true - y_pred),axis=0)

def mae_weights(y_true,y_pred):
    wg = np.array([2.0, 1.0, 2.0, 2.0], dtype=np.float32)
    # wg = np.array([2.5, 1.0, 2.0, 2.5], dtype=np.float32)

    y_true = y_true*wg
    y_pred = y_pred*wg
    return K.mean(K.abs(y_true - y_pred),axis=0)

def mae_weights2(y_true,y_pred):
    wg = np.array([4.0, 1.0, 4.0, 4.0], dtype=np.float32)
    y_true = y_true*wg
    y_pred = y_pred*wg
    return K.mean(K.abs(y_true - y_pred),axis=0)

def mae_sine_and_y_weight(y_true,y_pred):
    wg = np.array([2.0, 1.0, 2.0, 2.0], dtype=np.float32)
    # wg = np.array([2.5, 1.0, 2.0, 2.5], dtype=np.float32)
    weights = 0.5 + (0.5* K.sin(3.14 * y_true))
    
    y_true = y_true*wg
    y_pred = y_pred*wg
    
    return K.mean(K.abs(weights*(y_true - y_pred)),axis=0)

def mse_sigweight(y_true, y_pred):
    weights = K.sigmoid(5.0 * y_true)
    return K.mean(K.square(weights*(y_true - y_pred),axis=0))

def mse_sineweight(y_true, y_pred):
    weights = 0.5 + (K.sin(3.14 * y_true))
    return K.mean(K.square(weights*(y_true - y_pred)),axis=0)

def mse_sineweight_nfac2(y_true, y_pred):
    weights = 0.5 + 0.8*(K.sin(3.14 * K.square(y_true)))
    return K.mean(K.square(weights*(y_true - y_pred)),axis=0)

def mse_sineweight_nfac2_2(y_true, y_pred):
    weights = 2.0 * (K.sin(0.8 * K.square(y_true) )) - 1.0
    return K.mean(K.square(weights*(y_true - y_pred)),axis=0)

def mse_sineweight_nfac2_3(y_true, y_pred):
    # weights = 2.0 * (K.sin(0.5 * K.square(y_true) )) - 1.0
    weights = 2.0 * (K.sin(0.5 * K.square(y_true) )) - 0.4
    # weights[:,1] = 1.0
    # weights_n = tf.unstack(weights)
    # weights_n[:,1] = 1.0
    # weights = tf.stack(weights_n)
    return K.mean(K.square(weights*(y_true - y_pred)),axis=-1)



def savemodel(kerasfile, model):
   model.summary()
   newfile = kerasfile[:-3]+".txt"
   # model.save(kerasfile)
   try:
    model.save(kerasfile)
   except Exception:
        pass
   print("saving to {}".format(newfile))
   h5_to_txt(kerasfile,newfile)
   
   
def get_available_layers(model_layers, available_model_layers=[b"dense"]):
    parsed_model_layers = []
    for l in model_layers:
        for g in available_model_layers:
            if g in l:
                parsed_model_layers.append(l)
    return parsed_model_layers

# # KERAS HDF5 NEURAL NETWORK MODEL FILE TO NEURAL-FORTRAN ASCII MODEL FILE
# def h5_to_txt(weights_file_name, output_file_name=''):

#     #check and open file
#     with h5py.File(weights_file_name,'r') as weights_file:

#         weights_group_key=list(weights_file.keys())[0]

#         # activation function information in model_config
#         model_config = weights_file.attrs['model_config'].decode('utf-8') # Decode using the utf-8 encoding
#         model_config = model_config.replace('true','True')
#         model_config = model_config.replace('false','False')

#         model_config = model_config.replace('null','None')
#         model_config = eval(model_config)

#         model_layers = list(weights_file['model_weights'].attrs['layer_names'])
#         model_layers = get_available_layers(model_layers)
#         print("names of layers in h5 file: %s \n" % model_layers)

#         # attributes needed for .txt file
#         # number of model_layers + 1(Fortran includes input layer),
#         #   dimensions, biases, weights, and activations
#         num_model_layers = len(model_layers)+1

#         dimensions = []
#         bias = {}
#         weights = {}
#         activations = []

#         print('Processing the following {} layers: \n{}\n'.format(len(model_layers),model_layers))
#         if 'Input' in model_config['config']['layers'][0]['class_name']:
#             model_config = model_config['config']['layers'][1:]
#         else:
#             model_config = model_config['config']['layers']

#         for num,l in enumerate(model_layers):
#             layer_info_keys=list(weights_file[weights_group_key][l][l].keys())

#             #layer_info_keys should have 'bias:0' and 'kernel:0'
#             for key in layer_info_keys:
#                 if "bias" in key:
#                     bias.update({num:np.array(weights_file[weights_group_key][l][l][key])})

#                 elif "kernel" in key:
#                     weights.update({num:np.array(weights_file[weights_group_key][l][l][key])})
#                     if num == 0:
#                         dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[0]))
#                         dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[1]))
#                     else:
#                         dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[1]))

#             if 'Dense' in model_config[num]['class_name']:
#                 activations.append(model_config[num]['config']['activation'])
#             else:
#                 print('Skipping bad layer: \'{}\'\n'.format(model_config[num]['class_name']))

#     if not output_file_name:
#         # if not specified will use path of weights_file with txt extension
#         output_file_name = weights_file_name.replace('.h5', '.txt')

#     with open(output_file_name,"w") as output_file:
#         output_file.write(str(num_model_layers) + '\n')

#         output_file.write("\t".join(dimensions) + '\n')
#         if bias:
#             for x in range(len(model_layers)):
#                 bias_str="\t".join(list(map(str,bias[x].tolist())))
#                 output_file.write(bias_str + '\n')
#         if weights:
#             for x in range(len(model_layers)):
#                 weights_str="\t".join(list(map(str,weights[x].T.flatten())))
#                 output_file.write(weights_str + '\n')
#         if activations:
#             for a in activations:
#                 if a == 'softmax':
#                     print('WARNING: Softmax activation not allowed... Replacing with Linear activation')
#                     a = 'linear'
#                 output_file.write(a + "\n")

def h5_to_txt(weights_file_name, output_file_name=''):

    #check and open file
    with h5py.File(weights_file_name,'r') as weights_file:

        weights_group_key=list(weights_file.keys())[0]

        # activation function information in model_config
        model_config = weights_file.attrs['model_config']#.decode('utf-8') # Decode using the utf-8 encoding
        model_config = model_config.replace('true','True')
        model_config = model_config.replace('false','False')

        model_config = model_config.replace('null','None')
        model_config = eval(model_config)

        model_layers = list(weights_file['model_weights'].attrs['layer_names'])
        # model_layers = get_available_layers(model_layers)
        print("names of layers in h5 file: %s \n" % model_layers)

        # attributes needed for .txt file
        # number of model_layers + 1(Fortran includes input layer),
        #   dimensions, biases, weights, and activations
        num_model_layers = len(model_layers)+1

        dimensions = []
        bias = {}
        weights = {}
        activations = []

        print('Processing the following {} layers: \n{}\n'.format(len(model_layers),model_layers))
        if 'Input' in model_config['config']['layers'][0]['class_name']:
            model_config = model_config['config']['layers'][1:]
        else:
            model_config = model_config['config']['layers']

        for num,l in enumerate(model_layers):
            layer_info_keys=list(weights_file[weights_group_key][l][l].keys())

            #layer_info_keys should have 'bias:0' and 'kernel:0'
            for key in layer_info_keys:
                if "bias" in key:
                    bias.update({num:np.array(weights_file[weights_group_key][l][l][key])})

                elif "kernel" in key:
                    weights.update({num:np.array(weights_file[weights_group_key][l][l][key])})
                    if num == 0:
                        dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[0]))
                        dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[1]))
                    else:
                        dimensions.append(str(np.array(weights_file[weights_group_key][l][l][key]).shape[1]))

            if 'Dense' in model_config[num]['class_name']:
                activations.append(model_config[num]['config']['activation'])
            else:
                print('Skipping bad layer: \'{}\'\n'.format(model_config[num]['class_name']))

    if not output_file_name:
        # if not specified will use path of weights_file with txt extension
        output_file_name = weights_file_name.replace('.h5', '.txt')

    with open(output_file_name,"w") as output_file:
        output_file.write(str(num_model_layers) + '\n')

        output_file.write("\t".join(dimensions) + '\n')
        if bias:
            for x in range(len(model_layers)):
                bias_str="\t".join(list(map(str,bias[x].tolist())))
                output_file.write(bias_str + '\n')
        if weights:
            for x in range(len(model_layers)):
                weights_str="\t".join(list(map(str,weights[x].T.flatten())))
                output_file.write(weights_str + '\n')
        if activations:
            for a in activations:
                if a == 'softmax':
                    print('WARNING: Softmax activation not allowed... Replacing with Linear activation')
                    a = 'linear'
                output_file.write(a + "\n")
