#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing neural networks to replace radiative
transfer computations, either fully or just one component

This code is for emulating RTE (the solver)

This program takes existing input-output data generated with RTE+RRTMGP and
user-specified hyperparameters such as the number of neurons, optionally
scales the data, and trains a neural network. 

Temporary code

Contributions welcome!

@author: Peter Ukkonen
"""
import os
import gc
import numpy as np

from ml_loaddata import load_inp_outp_rte_sw,  preproc_minmax_inputs
from sklearn.model_selection import train_test_split

def broadband_reduce(y):
    ns,ny = y.shape
    ngpt = 224
    ncol = np.int(ns / ngpt)
    y = np.reshape(y,(ncol,ngpt,ny))
    return np.sum(y,axis=1)

def scale_outputs_gpt(y_raw):
    ns,ny = y_raw.shape
    ngpt = 224
    ncol = np.int(ns / ngpt)
    y_sc = np.copy(y_raw)
    y_sc = np.reshape(y_sc,(ncol,ngpt,ny))
    
    y_mean = np.zeros(ngpt)
    for igpt in range(ngpt):
        y_mean[igpt] = np.mean(y_sc[:,igpt,:])
        
    y_sigma = np.repeat(y_sc.std(),ngpt)  # 467.72
    
    for igpt in range(ngpt):
        y_sc[:,igpt,:] = (y_sc[:,igpt,:] - y_mean[igpt]) / y_sigma[igpt]

    y_sc = np.reshape(y_sc,(ns,ny))
    return y_sc, y_mean, y_sigma

def reverse_scale_outputs_gpt(y_sc, y_mean, y_sigma):
    ns,ny = y_sc.shape
    ngpt = 224
    ncol = np.int(ns / ngpt)
    y = np.copy(y_sc)
    y = np.reshape(y,(ncol,ngpt,ny))
    
    for igpt in range(ngpt):
        y[:,igpt,:] = y[:,igpt,:]*y_sigma[igpt] + y_mean[igpt]

    y = np.reshape(y,(ns,ny))
    return y

def scale_outputs_gpt_lay(y_raw):
    ns,ny = y_raw.shape
    ngpt = 224
    ncol = np.int(ns / ngpt)
    y_sc = np.copy(y_raw)
    y_sc = np.reshape(y_sc,(ncol,ngpt*ny))
    
    ny2 = y_sc.shape[1]
    
    y_mean = np.zeros(ny2)
    for i in range(ny2):
        y_mean[i] = np.mean(y_sc[:,i])
        
    y_sigma = np.repeat(y_sc.std(),ny2)  # 467.72
    
    for i in range(ny2):
        y_sc[:,i] = (y_sc[:,i] - y_mean[i]) / y_sigma[i]

    y_sc = np.reshape(y_sc,(ns,ny))
    return y_sc, y_mean, y_sigma

def reverse_scale_outputs_gpt_lay(y_sc, y_mean, y_sigma):
    ns,ny = y_sc.shape
    ngpt = 224
    ncol = np.int(ns / ngpt)
    y = np.copy(y_sc)
    y = np.reshape(y,(ncol,ngpt*ny))
    
    for i in range(y.shape[1]):
        y[:,i] = y[:,i]*y_sigma[i] + y_mean[i]

    y = np.reshape(y,(ns,ny))
    return y
# ----------------------------------------------------------------------------
# ----------------- TEMP. CODE, RTE EMULATION  ------------------------
# ----------------------------------------------------------------------------

# --- QUICK TRAINING EXERCISE FOR WHOLE SCHEME
#  ----------------- File paths -----------------
dat_file = "ml_data_g224_withclouds_CAMS_2018_RFMIPstyle.nc"     
dat_dir = '/media/peter/samlinux/data/data_training/'
dat_path = dat_dir + dat_file

x_raw, y_raw, y_bb = load_inp_outp_rte_sw(dat_path)

scale_inputs = True
scale_outputs = True

if scale_inputs:
    x,xmax,xmin = preproc_minmax_inputs(x_raw)
else:
    x = x_raw
    
if scale_outputs: 
    
    # Scale by GPT+LAY 
    y,y_mean,y_sigma = scale_outputs_gpt_lay(y_raw)
    
    # Scale by GPT
    # y,y_mean,y_sigma = scale_outputs_gpt(y_raw)
    
    
    # Original
    # ny = y_raw.shape[1]      # y.shape (14400, 122)
    # nobs = y_raw.shape[0]
    # y_mean = np.zeros(ny)
    # y_sigma = np.zeros(ny)
    # for i in range(ny):
    #     y_mean[i] = y_raw[:,i].mean()
    #     # y_sigma[i] = y_raw[:,i].std()
    # # y_mean = np.repeat(y_raw.mean(),ny)
    # y_sigma = np.repeat(y_raw.std(),ny)  # 467.72

    # nfac = 1
    # y  = preproc_pow_gptnorm(y_raw, nfac, y_mean, y_sigma)
else:
    y = y_raw

gc.collect()
(ns,ny) = y.shape
# Ready for training


import warnings
warnings.filterwarnings("ignore")

train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

# train is now 75% of the entire data set
# the _junk suffix means that we drop that variable completely
x_tr, x_test, y_tr, y_test = train_test_split(x, y, test_size=1 - train_ratio)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

# Ready for training

train_keras = True

if train_keras:
    from tensorflow.keras import losses, optimizers
    from tensorflow.keras.callbacks import EarlyStopping
    from ml_trainfuncs_keras import create_model_mlp, savemodel
    
    mymetrics   = ['mean_absolute_error']
    valfunc     = 'val_mean_absolute_error'
    activ       = 'softsign'
    activ       = 'relu'
    # activ           ='tanh'
    # activ           ='sigmoid'
    
    epochs      = 100000
    patience    = 25
    lossfunc    = losses.mean_squared_error
    ninputs     = x_tr.shape[1]
    lr          = 0.001
    # lr          = 0.0001 
    # lr          = 0.0002 
    batch_size  = 256
    batch_size  = 512
    neurons = [182, 182]
    
    optim = optimizers.Adam(lr=lr,rescale_grad=1/batch_size) 
    # optim = optimizers.Adam(lr=lr)
    
    # Create model
    model = create_model_mlp(nx=ninputs,ny=ny,neurons=neurons,activ=activ,kernel_init='he_uniform')
    # Compile model
    model.compile(loss=lossfunc, optimizer=optim,
                  metrics=mymetrics,  context= ["gpu(0)"])
    model.summary()
    
    # Create earlystopper
    earlystopper = EarlyStopping(monitor=valfunc,  patience=patience, verbose=1, mode='min',restore_best_weights=True)
    
    # START TRAINING
    history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
                        validation_data=(x_val,y_val), callbacks=[earlystopper])
    gc.collect()
    
    # TEST
    
    
    
    y_pred      = model.predict(x);  
    y_pred      = reverse_scale_outputs_gpt_lay(y_pred,y_mean,y_sigma)
    
    # plot_hist2d(y_raw,y_pred,20,True)      #  0.975
    
    
    y_bb_pred = broadband_reduce(y_pred) 
    
    plot_hist2d(y_bb,y_bb_pred,20,True)      #  0.979
        
    diff = np.abs(y_bb-y_bb_pred)
    np.max(diff)
