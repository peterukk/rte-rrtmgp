#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 15:38:01 2022

@author: peter
"""



import os
from netCDF4 import Dataset,num2date
import numpy as np



# Create a netCDF file specifying the NN. in case of two hidden layers it will
# look like this:
# dimensions
#   nn_layers  : 3        <--- does not include the input layer
#   nn_dim_inp    : 7     <--- nn_dim_* specifies the length of a dimension
#   nn_dim_hidden1: 16
#   nn_dim_hidden2: 16
#   nn_dim_outp   : 224

# variables:
#   integer nn_dimsize(layers) = [16,16,224]
#   float nn_weights_1(nn_dim_inp,     nn_dim_hidden1)
#   float nn_bias_1(nn_dim_hidden1)
#   float nn_weights_2(nn_dim_hidden1, nn_dim_hidden2)
#   float nn_bias_2(nn_dim_hidden2)
#   float nn_weights_3(nn_dim_hidden2, nn_dim_outp)
#   float nn_bias_3(nn_dim_outp)
#   string nn_inputs(nn_dim_inp)
#   string nn_activation(layers)


fpath_new = 'testfile.nc'

dat_new     = Dataset(fpath_new,'w')

nlay = np.size(model.layers) 

# Create initial dimensions before loop
dat_new.createDimension('nn_layers',nlay)

str_dim_prev = 'nn_dim_input'
dat_new.createDimension(str_dim_prev,nx)
# Create initial variables before loop
nc_dimsize      = dat_new.createVariable("nn_dimsize","i4",("nn_layers"))
nc_activ        = dat_new.createVariable("nn_activation","str",("nn_layers"))
nc_inputnames   = dat_new.createVariable("nn_inputs","str",(str_dim_prev))


# Create a netCDF file specifying the feedforward NN (Multilayer Perceptron)
# loop over hidden layers + output layer, where each of these are associated
# with weights, biases and activation - create the dimension and variables of
# each layer
# does not include the input layer as this should not be considered a NN layer 
# according to Bishop's book (Pattern recognition and Machine Learning)
for i in range(nlay):
    j = i+1

    weight = model.layers[i].get_weights()[0]
    bias = model.layers[i].get_weights()[1]
    
    dimsize         = weight.shape[1]
    
    # Create dimension corresponding to this layer
    if (i<nlay-1):
        str_dim_this    = 'nn_dim_hidden' + str(j)
    else:
        str_dim_this    = 'nn_dim_outp'
    dat_new.createDimension(str_dim_this,dimsize)

    
    # Create weight variable
    str_weight = 'nn_weights_'+str(j)
    str_bias  =  'nn_bias_'+str(j)
    nc_weight  = dat_new.createVariable(str_weight,"f4",(str_dim_prev,str_dim_this))
    nc_bias    = dat_new.createVariable(str_bias,  "f4",(str_dim_this))
                                           
    # Write the data
    nc_dimsize[i]   = dimsize
    nc_weight[:]    = weight
    nc_bias[:]      = bias
    # Write the activation function as a string
    nc_activ[i]     = activ[i]
    
    str_dim_prev = str_dim_this

for i in range(nx):
    nc_inputnames[i] = input_names[i]

dat_new.close()

