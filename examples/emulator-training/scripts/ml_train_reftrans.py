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


from ml_loaddata import load_inp_outp_rte_sw,  preproc_minmax_inputs, load_inp_outp_reftrans, \
    preproc_pow_gptnorm
from sklearn.model_selection import train_test_split



# ----------------------------------------------------------------------------
# ----------------- TEMP. CODE, REFTRANS EMULATION  ------------------------
# ----------------------------------------------------------------------------


import warnings
warnings.filterwarnings("ignore")

#  ----------------- File paths -----------------
dat_file = "ml_data_g224_withclouds_CAMS_2018_RFMIPstyle.nc"     
dat_dir = '/media/peter/samlinux/data/data_training/'
dat_path = dat_dir + dat_file

x_raw, y_raw  = load_inp_outp_reftrans(dat_path)

scale_inputs = True
scale_outputs = True

if scale_inputs:
    x,xmax,xmin = preproc_minmax_inputs(x_raw)
else:
    x = x_raw
    
if scale_outputs:
        # Original
    ny = y_raw.shape[1]      # y.shape (14400, 122)
    nobs = y_raw.shape[0]
    y_mean = np.zeros(ny)
    y_sigma = np.zeros(ny)
    for i in range(ny):
        y_mean[i] = y_raw[:,i].mean()
        # y_sigma[i] = y_raw[:,i].std()
    # y_mean = np.repeat(y_raw.mean(),ny)
    y_sigma = np.repeat(y_raw.std(),ny)  # 467.72

    nfac = 1

    y  = preproc_pow_gptnorm(y_raw, nfac, y_mean, y_sigma)
else:
    y = y_raw
    
# xmin = np.array([4.1395123e-09, 4.0951968e-12, 0.0000000e+00, 1.2061903e-04],
#       dtype=np.float32)

# xmax = np.array([2.9715678e+04, 1.0000000e+00, 4.8579213e-01, 9.9999982e-01],
#       dtype=np.float32)


gc.collect()
(ns,ny) = y.shape
nx = x.shape[1]

# Ready for training

import warnings
warnings.filterwarnings("ignore")

# train_ratio = 0.75
# validation_ratio = 0.15
# test_ratio = 0.10
train_ratio = 0.1
validation_ratio = 0.1
test_ratio = 0.8

# train is now 75% of the entire data set
# the _junk suffix means that we drop that variable completely
x_tr, x_test, y_tr, y_test = train_test_split(x, y, test_size=1 - train_ratio)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 


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
batch_size  = 1024
neurons     = [3]

optim = optimizers.Adam(lr=lr,rescale_grad=1/batch_size) 
# optim = optimizers.Adam(lr=lr)

# Create model
model = create_model(nx=nx,ny=ny,neurons=neurons,activ=activ,kernel_init='he_uniform')
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

plot_hist2d(y,y_pred,20,True)      #  0.9838
#1 : R2 0.46
#2 : R2 0.9667
#3 : R2 0.468
#4 : R2 0.584

y_bb_pred = broadband_reduce(y_pred) 

plot_hist2d(y_bb,y_bb_pred,20,True)      #  0.979
    
diff = np.abs(y_bb-y_bb_pred)
np.max(diff)
