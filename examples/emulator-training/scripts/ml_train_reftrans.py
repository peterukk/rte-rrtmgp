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

from ml_loaddata import load_inp_outp_reftrans, preproc_minmax_inputs_reftrans, \
    preproc_pow_gptnorm, preproc_pow_gptnorm_reverse, gen_synthetic_inp_outp_reftrans

from sklearn.model_selection import train_test_split

from ml_eval_funcs import plot_hist2d


# ----------------------------------------------------------------------------
# ----------------- TEMP. CODE, REFTRANS EMULATION  ------------------------
# ----------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

# ----------- config

scale_inputs = True
scale_outputs = False

synthetic_data = False
synthetic_data_supplement = True


#  ----------------- File paths -----------------
dat_file = "ml_data_g224_withclouds_CAMS_2018_RFMIPstyle.nc"     
dat_dir = '/media/peter/samlinux/data/data_training/'
dat_path = dat_dir + dat_file

# dat_file = "ml_data_g224_CAMS_2018_clouds.nc"
# dat_dir  = '/home/puk/soft/rte-rrtmgp-nn/examples/emulator-training/data_training/'
# dat_path = dat_dir + dat_file

# Lets try hypercube sampling instead of real data, because there's only 4 inputs

# x_raw[:,i].min(), x_raw[:,i].max()
# i0 (4.1395123e-09, 29715.678)
# i1 (4.095197e-12, 1.0)
# i2 (0.0, 0.48579213)
# i3  (0.00012061903, 0.9999998)
minmax_tau  = (1e-09, 20000.0)
minmax_ssa  = (0.0, 1.0)
minmax_g    = (0.0, 0.55)
minmax_mu0  = (0.0, 1.0)

if synthetic_data:

    nsamples    = 500000
    x_raw, y_raw = gen_synthetic_inp_outp_reftrans(nsamples, minmax_tau, minmax_ssa, minmax_g,
                                    minmax_mu0)
    
    minmax_tau  = (1.0, 10.0)

    x_raw2, y_raw2 = gen_synthetic_inp_outp_reftrans(nsamples, minmax_tau, minmax_ssa, minmax_g,
                                    minmax_mu0)
    x_raw = np.concatenate((x_raw,x_raw2),axis=0)
    y_raw = np.concatenate((y_raw,y_raw2),axis=0)
    
    
    x_raw_test,y_raw_test =  load_inp_outp_reftrans(dat_path)
    # this "real" dataset is large, lets pick random 5% for testing
    frac = 0.05 
    nrows = x_raw_test.shape[0]
    inds_rand = np.sort(np.random.choice(np.arange(nrows),np.int(frac*nrows),replace=False))
    x_raw_test = x_raw_test[inds_rand,:]; y_raw_test = y_raw_test[inds_rand,:]
    
else:
    x_raw, y_raw  = load_inp_outp_reftrans(dat_path)
    
    # this "real" dataset is large, lets pick random 10%
    frac = 0.1
    nrows = x_raw.shape[0]
    inds_rand = np.sort(np.random.choice(np.arange(nrows),np.int(frac*nrows),replace=False))
    x_raw = x_raw[inds_rand,:]; y_raw = y_raw[inds_rand,:]
    
    # of this, random 15% for testing
    test_frac = 0.15
    
    x_raw, x_raw_test, y_raw, y_raw_test = train_test_split(x_raw, y_raw, test_size=test_frac)


    if synthetic_data_supplement:
        minmax_tau  = (1.0, 10.0)
        nsamples    = 800000

        x_raw2, y_raw2 = gen_synthetic_inp_outp_reftrans(nsamples, minmax_tau, minmax_ssa, minmax_g,
                                    minmax_mu0)
        x_raw = np.concatenate((x_raw,x_raw2),axis=0)
        y_raw = np.concatenate((y_raw,y_raw2),axis=0)
    


if scale_inputs:
    nfac_tau = 4
    nfac_tau = 2

    xmin = np.array([0, 0, 0, 0])
    # xmax = np.array([4.0,  1,  0.54999859,  0.99999514]) # nfac 8
    # xmax = np.array([13.101879,  1,  0.54999859,  0.99999514])# nfac 4
    xmax = np.array([180.0,  1,  0.54999859,  0.99999514]) # nfac  2
    x = preproc_minmax_inputs_reftrans(x_raw, nfac_tau, (xmin,xmax))
    # x,xmin,xmax = preproc_minmax_inputs_reftrans(x_raw,nfac_tau)
    
    x_test   = preproc_minmax_inputs_reftrans(x_raw_test, nfac_tau, (xmin,xmax))
else:
    x = x_raw
    x_test = x_raw_test
    
if scale_outputs:
    # Original
    ny = y_raw.shape[1]      
    nobs = y_raw.shape[0]
    y_mean = np.zeros(ny)
    y_sigma = np.zeros(ny)
    for i in range(ny):
        y_mean[i] = y_raw[:,i].mean()
        # y_sigma[i] = y_raw[:,i].std()
    # y_mean = np.repeat(y_raw.mean(),ny)
    y_sigma = np.repeat(y_raw.std(),ny)  

    nfac = 1
    y  = preproc_pow_gptnorm(y_raw, nfac, y_mean, y_sigma)
    
    y_test = preproc_pow_gptnorm(y_raw_test, nfac, y_mean, y_sigma)
else:
    y = y_raw
    y_test = y_raw_test
    
# xmin = np.array([4.1395123e-09, 4.0951968e-12, 0.0000000e+00, 1.2061903e-04],
#       dtype=np.float32)

# xmax = np.array([2.9715678e+04, 1.0000000e+00, 4.8579213e-01, 9.9999982e-01],
#       dtype=np.float32)


gc.collect()
(ns,ny) = y.shape
nx = x.shape[1]

# Validation data as a subset of the training data

import warnings
warnings.filterwarnings("ignore")

val_ratio = 0.25
x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=0.25)

# Ready for training


train_pytorch = False

if train_pytorch:
    import torch
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, TensorDataset
    from ml_trainfuncs_pytorch import MLP

    batch_size = 256
    
    x_tr_torch = torch.from_numpy(x_tr); y_tr_torch = torch.from_numpy(y_tr)
    data_tr  =  TensorDataset(x_tr_torch,y_tr_torch)
    
    x_val_torch = torch.from_numpy(x_val); y_val_torch = torch.from_numpy(y_val)
    data_val    = TensorDataset(x_val_torch,y_val_torch)
    
    x_test_torch = torch.from_numpy(x_test); y_test_torch = torch.from_numpy(y_test)
    data_test    = TensorDataset(x_test_torch,y_test_torch)
    
    mlp = MLP(nx=nx,ny=ny)
    #trainer = pl.Trainer(auto_scale_batch_size='power', gpus=0, deterministic=True, max_epochs=5)
    
    trainer = pl.Trainer(gpus=0, deterministic=True, max_epochs=10)
    #trainer = pl.Trainer(gpus=0, deterministic=True, max_epochs=5,num_processes=3)
    
    trainer.fit(mlp, train_dataloader=DataLoader(data_tr,batch_size=batch_size), 
                val_dataloaders=DataLoader(data_val,batch_size=batch_size))




train_keras = True

if train_keras:
    
    from keras import losses, optimizers
    from keras.callbacks import EarlyStopping
    from ml_trainfuncs_keras import create_model_mlp, savemodel

    mymetrics   = ['mean_absolute_error']
    valfunc     = 'val_mean_absolute_error'
    
    # activ0      = 'softsign'
    # activ0      = 'sigmoid'
    activ0       = 'relu'
    activ       =  activ0
    
    activ_last = 'linear'
    
    epochs      = 100000
    patience    = 25
    lossfunc    = losses.mean_squared_error
    
    lr          = 0.001
    # lr          = 0.0001 
    # lr          = 0.0002 
    # batch_size  = 512
    batch_size  = 1024
    neurons     = [16,16]
    neurons     = [8]
    
    
    optim = optimizers.Adam(lr=lr,rescale_grad=1/batch_size) 
    # optim = optimizers.Adam(lr=lr)
    
    # Create model
    model = create_model_mlp(nx=nx,ny=ny,neurons=neurons,activ0=activ0,activ=activ,
                             activ_last = activ_last, kernel_init='he_uniform')
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
    
    y_test_pred      = model.predict(x_test);  
    
    if scale_outputs:
        y_test_pred      = preproc_pow_gptnorm_reverse(y_test_pred,nfac, y_mean,y_sigma)
        
    plot_hist2d(y_raw_test,y_test_pred,20,True)      # [6] 0.9988
    i = 3
    np.corrcoef(y_raw_test[:,i],y_test_pred[:,i])
    #0 0.99664619
    #1 0.9704285
    #2 0.99718657
    #3 0.96724492
    
    kerasfile = "/media/peter/samlinux/gdrive/phd/soft/soft/rte-rrtmgp-nn/neural/data/reftrans-8.h5"
    # kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/pfrac-18-16-16.h5"
    
    # from keras.models import load_model
    # kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/tau-sw-ray-7-16-16.h5"
    # model = load_model(kerasfile,compile=False)
    
    savemodel(kerasfile, model)


