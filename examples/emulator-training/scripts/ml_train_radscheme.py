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
                        preproc_pow_gptnorm, preproc_pow_gptnorm_reverse
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------------
# ----------------- TEMP. CODE, RTE+RRTMGP EMULATION  ------------------------
# ----------------------------------------------------------------------------

# --- QUICK TRAINING EXERCISE FOR WHOLE SCHEME
dat_file = 'ml_data_g224_clouds_CAMS_2011_2012_2018_RFMIPstyle.nc'
dat_path = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/" + dat_file

x_raw, y_raw = load_inp_outp_rte_rrtmgp_sw(dat_path, 'rsu_rsd')

scale_inputs = True
scale_outputs = True

if scale_inputs:
    print("input scaling code here")
else:
    x = x_raw
    
if scale_outputs: 
    
    # SCALE
    ngpt = y_raw.shape[1]      # y.shape (14400, 122)
    nobs = y_raw.shape[0]
    y_mean = np.zeros(ngpt)
    for igpt in range(ngpt):
        y_mean[igpt] = y_raw[:,igpt].mean()
        
    y_sigma = np.std(y_raw) # 467.72
    y_sigma = np.repeat(y_sigma,ngpt)
    nfac = 1

    y  = preproc_pow_gptnorm(y_raw, nfac, y_mean, y_sigma)
else:
    y = y_raw

gc.collect()
# Ready for training

#
#import warnings
#warnings.filterwarnings("ignore")
#
#train_ratio = 0.75
#validation_ratio = 0.15
#test_ratio = 0.10
#
## train is now 75% of the entire data set
## the _junk suffix means that we drop that variable completely
#x_tr, x_test, y_tr, y_test = train_test_split(x, y, test_size=1 - train_ratio)
#
## test is now 10% of the initial data set
## validation is now 15% of the initial data set
#x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
#
#
#mymetrics   = ['mean_absolute_error']
#valfunc     = 'val_mean_absolute_error'
#activ       = 'softsign'
#epochs      = 100000
#patience    = 25
#lossfunc    = losses.mean_squared_error
#ninputs     = x_tr.shape[1]
#lr          = 0.001 
#batch_size  = 1024
#neurons     = [64,64]
#
#optim = optimizers.Adam(lr=lr,rescale_grad=1/batch_size) 
#
## Create model
#model = create_model(nx=ninputs,ny=ngpt,neurons=neurons,activ=activ,kernel_init='he_uniform')
## Compile model
#model.compile(loss=lossfunc, optimizer=optim,
#              metrics=mymetrics,  context= ["gpu(0)"])
#model.summary()
#
## Create earlystopper
#earlystopper = EarlyStopping(monitor=valfunc,  patience=patience, verbose=1, mode='min',restore_best_weights=True)
#
## START TRAINING
#history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
#                    validation_data=(x_val,y_val), callbacks=[earlystopper])
#gc.collect()
#
## TEST
#y_pred_sc    = model.predict(x);  
#y_pred       = preproc_pow_gptnorm_reverse(y_pred_sc, nfac, y_mean,y_sigma)
#
#plot_hist2d(y_raw,y_pred,20,True)      #  
    
