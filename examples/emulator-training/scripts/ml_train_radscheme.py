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
import os
import gc
import numpy as np

from ml_loaddata import load_inp_outp_rte_rrtmgp_sw, preproc_tau_to_crossection, \
                        preproc_pow_gptnorm, preproc_pow_gptnorm_reverse, preproc_rrtmgp_inputs
from ml_eval_funcs import plot_hist2d
 
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------------
# ----------------- TEMP. CODE, RTE+RRTMGP EMULATION  ------------------------
# ----------------------------------------------------------------------------

# --- QUICK TRAINING EXERCISE FOR WHOLE SCHEME
#  ----------------- File paths -----------------
fpath =  "/media/peter/samlinux/data/data_training/ml_data_g224_withclouds_CAMS_2011-2013_RFMIPstyle.nc"  
fpath_test = "/media/peter/samlinux/data/data_training/ml_data_g224_withclouds_CAMS_2018_RFMIPstyle.nc" 

x_raw, y_raw = load_inp_outp_rte_rrtmgp_sw(fpath, 'rsu_rsd')
x_raw_test, y_raw_test = load_inp_outp_rte_rrtmgp_sw(fpath, 'rsu_rsd')

scale_inputs = True
scale_outputs = True

if scale_inputs:
    x,xmax,xmin = preproc_rrtmgp_inputs(x_raw)
    x_test      = preproc_rrtmgp_inputs(x_raw_test)

else:
    x = x_raw
    
if scale_outputs: 
    ngpt = y_raw.shape[1]      # y.shape (14400, 122)
    y_mean = np.zeros(ngpt)
    y_sigma = np.zeros(ngpt)
    for igpt in range(ngpt):
        y_mean[igpt] = y_raw[:,igpt].mean()
        # y_sigma[igpt] = y_raw[:,igpt].std()
    # y_mean = np.repeat(y_raw.mean(),ngpt)
    y_sigma = np.repeat(y_raw.std(),ngpt)  # 467.72

    nfac = 1

    y       = preproc_pow_gptnorm(y_raw, nfac, y_mean, y_sigma)
    y_test  = preproc_pow_gptnorm(y_raw_test, nfac, y_mean, y_sigma)
else:
    y = y_raw

gc.collect()

# Validation data as a subset of the training data
val_ratio = 0.25
x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=val_ratio)

# Ready for training

train_keras = True

if train_keras:
    
    from keras import losses, optimizers
    from keras.callbacks import EarlyStopping
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
    # lr          = 0.001
    # lr          = 0.0001 
    lr          = 0.0002 
    
    batch_size  = 128
    neurons     = [128,128]
    neurons     = [256,256]
    neurons     = [512,256]
    
    optim = optimizers.Adam(lr=lr,rescale_grad=1/batch_size) 
    # optim = optimizers.Adam(lr=lr)
    
    # Create model
    model = create_model_mlp(nx=ninputs,ny=ngpt,neurons=neurons,activ=activ,kernel_init='he_uniform')
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
    y_pred_sc    = model.predict(x);  
    y_pred       = preproc_pow_gptnorm_reverse(y_pred_sc, nfac, y_mean,y_sigma)
    
    plot_hist2d(y_raw,y_pred,20,True)      #  
        
    diff = np.abs(y_raw-y_pred)
    np.max(diff)
