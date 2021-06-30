#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:52:23 2021

@author: peter
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:23:10 2020

@author: peter
"""


import os
rootdir  = '/media/peter/samlinux/gdrive/phd/'
os.chdir(rootdir+'soft/python/gas_optics')
from gasopt_load_train_funcs import load_data_all,create_model, gptnorm_numba,gptnorm_numba_reverse
from gasopt_load_train_funcs import preproc_inputs,losses
from gasopt_load_train_funcs import optimizers, EarlyStopping, savemodel
from gasopt_load_train_funcs import ymeans_lw, ysigma_lw, ymeans_sw, ysigma_sw, ysigmas_sw, ysigmas_lw
from gasopt_load_train_funcs import ymeans_sw_ray, ysigma_sw_ray, ymeans_sw_abs, ysigma_sw_abs
from gasopt_load_train_funcs import plot_hist2d_T, plot_hist2d
from gasopt_load_train_funcs import load_data_cams

import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from keras.callbacks import ModelCheckpoint_totxt

# ------------ LOAD DATA ------------ 

shortwave       = False
ychoice         = 1  # = 0 for planck fraction (longwave) or ssa(shortwave), 
# =1 for optical depth (sw/lw), 2 for tau_rau (sw), 3 for tau_abs(sw)

dcol            = 1
frac_cams       = 1.0
frac_val        = 0.025  # What fraction of datasets to use for validation
seed            = 7

cams_only = False

if shortwave:
    ngpt = 224
    fname_rfmip = rootdir+'rrtmgp_dev/inputs_outputs/inp_outp_sw_RFMIP-Halton_1f1.nc'
    fname_cams = rootdir+'rrtmgp_dev/inputs_outputs/inp_outp_sw_CAMS_1f1.nc'
    fname_nwp  =  rootdir+'rrtmgp_dev/inputs_outputs/inp_outp_sw_NWPSAF_1f1.nc'
    fname_GCM  =  rootdir+'rrtmgp_dev/inputs_outputs/inp_outp_sw_GCM_1f1.nc'
    fname_Garand = rootdir+'rrtmgp_dev/inputs_outputs/inp_outp_sw_Garand_1f1.nc'
    fname_CKDMIP = rootdir+'rrtmgp_dev/inputs_outputs/inp_outp_sw_CKDMIP-MM_1f1.nc'
else:
    ngpt = 256
    fname_rfmip = rootdir+'rrtmgp_dev/inputs_outputs/inp_outp_lw_RFMIP-Halton_1f1.nc'
    fname_cams = rootdir+'rrtmgp_dev/inputs_outputs/inp_outp_lw_CAMS_1f1.nc'
    fname_nwp  =  rootdir+'rrtmgp_dev/inputs_outputs/inp_outp_lw_NWPSAF_1f1.nc'
    fname_GCM  =  rootdir+'rrtmgp_dev/inputs_outputs/inp_outp_lw_GCM_1f1.nc'
    fname_Garand = rootdir+'rrtmgp_dev/inputs_outputs/inp_outp_lw_Garand_1f1.nc'
    fname_CKDMIP = rootdir+'rrtmgp_dev/inputs_outputs/inp_outp_lw_CKDMIP-MM_1f1.nc'


x_tr, x_val, x_test, y_tr, y_val, y_test, ngas, ngpt = load_data_cams(fname_rfmip,
            fname_cams, shortwave, ychoice, dcol,frac_val, frac_cams, seed)

gc.collect()

ngas = x_tr.shape[1]

if  ychoice>0:
    idx_dry = ngas-1
    col_dry_tr = x_tr[:,idx_dry]
    col_dry_val = x_val[:,idx_dry]    
    col_dry_test = x_test[:,idx_dry] 
    y_tr = y_tr/ (np.repeat(x_tr[:,idx_dry,np.newaxis],ngpt,axis=1))
    y_val = y_val/ (np.repeat(x_val[:,idx_dry,np.newaxis],ngpt,axis=1))
    tau_test = np.copy(y_test)
    y_test = y_test/ (np.repeat(x_test[:,idx_dry,np.newaxis],ngpt,axis=1))
    # Don't need col_dry anymore
    x_tr = np.delete(x_tr,idx_dry,1)
    x_val = np.delete(x_val,idx_dry,1)
    x_test = np.delete(x_test,idx_dry,1)
    

# Scale inputs
# This was already done within RRTMGP
# varlist_new = ['tlay', 'play', 'h2o',    'o3',      'co2',    'n2o',   'ch4',   
#       'cfc11', 'cfc12', 'co',  'ccl4',  'cfc22',  'hfc143a', 'hfc125', 'hfc23', 'hfc32', 'hfc134a', 'cf4']  
# data_min_ = np.array([1.60e+02, 5.15e-03, 1.01e-02, 4.36e-03, 1.41e-04, 0.00e+00,
#        2.55e-08, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
#        0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00])
# data_max_ = np.array([3.2047600e+02, 1.1550600e+01, 5.0775300e-01, 6.3168340e-02,
#        2.3000003e-03, 5.8135214e-07, 3.6000001e-06, 2.0000002e-09,
#        5.3385213e-10, 1.3127458e-06, 1.0316801e-10, 2.3845328e-10,
#        7.7914392e-10, 9.8880004e-10, 3.1067642e-11, 1.3642075e-11,
#        4.2330001e-10, 1.6702625e-10])
# data_min_ = data_min_[0:ngas]; data_max_ = data_max_[0:ngas]
# x_tr = preproc_inputs(x_tr, data_min_, data_max_)
# x_val = preproc_inputs(x_val, data_min_, data_max_)

# Scale outputs and set model architecture depending on what we're predicting

stdnorm = False # Regular standard scaling using sigmas for individual outputs

if shortwave:
    if ychoice==1:
        y_means = ymeans_sw
        sigma   = ysigmas_sw
        neurons = [36,36]
        nfac = 8
    elif ychoice==2:
        neurons = [36,36]
        nfac = 8
        y_means = ymeans_sw_ray
        sigma = ysigma_sw_ray
    elif ychoice==3:
        neurons = [36,36]
        nfac = 8
        y_means = ymeans_sw_abs
        sigma = ysigma_sw_abs
    y_tr    = gptnorm_numba(nfac,y_tr,y_means,sigma)
    y_val   = gptnorm_numba(nfac,y_val,y_means,sigma)
    y_test  = gptnorm_numba(nfac,y_test,y_means,sigma)
else:
    if ychoice==1:
        if stdnorm:
            sigma   = ysigmas_lw
        else:
            sigma   = ysigma_lw
        neurons = [58,58]
        nfac = 8
        y_means = ymeans_lw
        y_tr    = gptnorm_numba(nfac,y_tr,ymeans_lw,sigma)
        y_val   = gptnorm_numba(nfac,y_val,ymeans_lw,sigma)
        y_test   = gptnorm_numba(nfac,y_test,ymeans_lw,sigma)
    else:
        neurons = [16,16]
        nfac      = 2
        y_tr    = np.power(y_tr, (1.0/nfac))
        y_val   = np.power(y_val, (1.0/nfac))
        y_test   = np.power(y_test, (1.0/nfac))

gc.collect()

mymetrics   = ['mean_absolute_error']
valfunc     = 'val_mean_absolute_error'
activ       = 'softsign'
fpath       = rootdir+'data/tmp/tmp.h5'
epochs      = 800
patience    = 15
lossfunc    = losses.mean_squared_error
ninputs     = x_tr.shape[1]
lr          = 0.001 
batch_size  = 1024

neurons = [64,64]
neurons = [58,58]
# neurons = [52,52,52]

# batch_size  = 3*batch_size
# lr          = 2 * lr

optim = optimizers.Adam(lr=lr,rescale_grad=1/batch_size) 

# Create model
model = create_model(nx=ninputs,ny=ngpt,neurons=neurons,activ=activ,kernel_init='he_uniform')

model.compile(loss=lossfunc, optimizer=optim,
              metrics=mymetrics,  context= ["gpu(0)"])
model.summary()

gc.collect()
# Create earlystopper
earlystopper = EarlyStopping(monitor=valfunc,  patience=patience, verbose=1, mode='min',restore_best_weights=True)

# START TRAINING

history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
                    validation_data=(x_val,y_val), callbacks=[earlystopper])
gc.collect()


# # ------------- RECOMPILE WITH MEAN-ABS-ERR -------------
# model.compile(loss=losses.mean_absolute_error, optimizer=optim,metrics=['mean_squared_error'])
# earlystopper = EarlyStopping(monitor='val_loss',  patience=patience, verbose=1, mode='min',restore_best_weights=True)
# history2 = model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size, shuffle=True,  verbose=1, validation_data=(x_val,y_val), callbacks=[earlystopper])
# gc.collect()


# # ------------- RECOMPILE WITH MEAN-SQUARED_ERR -------------
# model.compile(loss=losses.mean_squared_error, optimizer=optim,metrics=['mean_absolute_error'])
# earlystopper = EarlyStopping(monitor='val_loss',  patience=patience, verbose=1, mode='min',restore_best_weights=True)
# history = model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size, shuffle=True,  verbose=1, validation_data=(x_val,y_val), callbacks=[earlystopper])
# gc.collect()



# batch_size = 1024
# Epoch 133/800
# 6981852/6981852 [==============================] - 21s 3us/step - loss: 8.9374e-05 - mean_absolute_error: 0.0051 - val_loss: 7.8724e-05 - val_mean_absolute_error: 0.0049
# Restoring model weights from the end of the best epoch
# Epoch 00133: early stopping


y_test_nn       = model.predict(x_test);  
y_test_nn       = gptnorm_numba_reverse(nfac,y_test_nn, y_means,sigma)
tau_test_nn     = y_test_nn * (np.repeat(col_dry_test[:,np.newaxis],ngpt,axis=1))
    
plot_hist2d(tau_test,tau_test_nn,20,True)        # 
plot_hist2d_T(tau_test,tau_test_nn,20,True)      #  


# ------------------ SAVE AND LOAD MODELS ------------------ 


# kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/tau-sw-ray-7-16-16_2.h5"
kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/tau-lw-18-58-58_cams.h5"
# kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/pfrac-18-16-16.h5"

# from keras.models import load_model
# kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/tau-sw-ray-7-16-16.h5"
# model = load_model(kerasfile,compile=False)

savemodel(kerasfile, model)

