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
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Input
import numpy as np
import h5py

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

def savemodel(kerasfile, model):
   model.summary()
   newfile = kerasfile[:-3]+".txt"
   model.save(kerasfile)
   print("saving to {}".format(newfile))
   h5_to_txt(kerasfile,newfile)
   
   
def get_available_layers(model_layers, available_model_layers=[b"dense"]):
    parsed_model_layers = []
    for l in model_layers:
        for g in available_model_layers:
            if g in l:
                parsed_model_layers.append(l)
    return parsed_model_layers

# KERAS HDF5 NEURAL NETWORK MODEL FILE TO NEURAL-FORTRAN ASCII MODEL FILE
def h5_to_txt(weights_file_name, output_file_name=''):

    #check and open file
    with h5py.File(weights_file_name,'r') as weights_file:

        weights_group_key=list(weights_file.keys())[0]

        # activation function information in model_config
        model_config = weights_file.attrs['model_config'].decode('utf-8') # Decode using the utf-8 encoding
        model_config = model_config.replace('true','True')
        model_config = model_config.replace('false','False')

        model_config = model_config.replace('null','None')
        model_config = eval(model_config)

        model_layers = list(weights_file['model_weights'].attrs['layer_names'])
        model_layers = get_available_layers(model_layers)
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
