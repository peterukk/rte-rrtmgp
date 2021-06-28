#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing neural network emulators of RRTMGP gas optics
scheme; to be run from /examples/ml-training within the RTE+RRTMGP library

This script generates training data according to user choices and can be ran
several times with different options to create different training datasets.

It can be combined with gasopt_train to train neural networks

Contributions welcome!

@author: Peter Ukkonen
"""



import os
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


rootdir  = '/media/peter/samlinux/gdrive/phd/'

# rte_rrtmgp_dir   = os.path.join("..", "..")
this_dir         = os.getcwd()
rte_rrtmgp_dir   = this_dir + '/../../'
# This files lives in $RRTMGP_ROOT/examples/all-sky/


# 1. Choose shortwave or longwave
# Two output variables which depend on this choice will be generated with RRTMGP
# So far the following outputs are supported:
# Longwave (LW) variables
#   "lw_abs" --> Absorption optical depth / cross-section (LW)
#   "lw_planck" --> Planck fraction, needed to compute Planck source functions
#                   from temperature
# Shortwave (SW) variables
#   "sw_abs" --> Absorption optical depth / cross-section (SW)
#   "sw_ray"--> Rayleigh optical depth / cross-section (SW), measuring
#   scattering by gas molecules
# These output variables are 1D arrays which include all g-points, so have sizes
# NGPT which depends on the k-distribution.
spectrum = 'longwave'
#spectrum = 'shortwave'

# 2. Choose input data source / generation method 
# 
# inputs and outputs for NN training (where the inputs are gas concentrations, 
# temperature, and pressure, and the outputs listed above) will be generated
#  according to the specified method. Usually, some "real" atmospheric profiles
# will be used as basis.

# 'ckdmip_mmm' = "Minimum, maximum, mean" dataset in CKDMIP, which originally
#             includes the MMM profiles of T, H2O, O3 (in present-day climate)
#             + MMM values of CH4 and CO2 from RFMIP 
# 'cams'       = (NOT YET IMPLEMENTED) CAMS reanalysis data
# 'garand'     = (NOT YET IMPLEMENTED) Garand profiles which were used to tune RRTMGP
# 'synthetic'  = (NOT YET IMPLEMENTED) create input data completely on-the-fly
#  by choosing a set of gases
# to vary in a user-defined range, as well as the range of temperatures;
# the co dependence of T,p,H2O and O3 needs to be parameterized.

data_source  = "ckdmip_mmm"

# these inital datasets can be extended synthetically by sampling the concentrations of 
# user-specified gases which are not in the initial datasets
# Hypercube sampling of gases can be done using a Halton sequence, similarly to
# Ukkonen et al. 2020.
# In the longwave, this can amount up to a hypercube with up to 14 dimensions!
# We want to cover this huge space evenly with a small number of samples.
#
# 
hypercube_sampling = True



# 3. Choose file specifying the RRTMGP k-distribution 
lw_gas_coeffs_file = "rrtmgp-data-lw-g256-2018-12-04.nc"
sw_gas_coeffs_file = "rrtmgp-data-sw-g224-2018-12-04.nc"
lw_gas_coeffs_path   = os.path.join(rte_rrtmgp_dir, "rrtmgp", "data", lw_gas_coeffs_file)
sw_gas_coeffs_path   = os.path.join(rte_rrtmgp_dir, "rrtmgp", "data", sw_gas_coeffs_file)


inp_file_CAMS  =  rootdir+'CAMS_profiles.nc'
inp_file_CKDMIP = rootdir+'CAMS_profiles.nc'

if spectrum=='shortwave':
    gas_coeffs_path = sw_gas_coeffs_path
else:
    gas_coeffs_path = lw_gas_coeffs_path

# Get 

# Choose which minor gases to sample

# Do NOT include nitrogen or oxygen - they can be considered constant

# Example: use RFMIP gases
gas_names_rfmip = ['carbon_dioxide','carbon_monoxide','carbon_tetrachloride', 'cf4',
 'cfc11', 'cfc12','hcfc22','hfc125','hfc134a','hfc143a','hfc23',
 'hfc32', 'methane', 'nitrous_oxide'] 
# 14 non-constant gases. Ozone and water vapour not included (height dependency)

gas_names = gas_names_rfmip

# Set the range of each gas. NOTE CFC-11 can be used to represent a large number
# of longwave gases by using artificially increased concentrations 
# (CFC-11-equivalent, from Meinshausen et al., 2017)
# This is done in CKDMIP, which only includes CO2, CH4, N2O, CFC-11-eq and CFC-12

# 14 non-constant gases. Ozone and water vapour not included (height dependency)

i = 0
maxvals = np.zeros(np.size(gas_names_rfmip))
minvals = np.zeros(np.size(gas_names_rfmip))

for var in gas_names_rfmip:
    # maxvals[i] = np.max( dat.variables[var][0:14].data)
    # minvals[i] = np.min( dat.variables[var][0:14].data)
    #minvals[i] = 0
    print("max for {}: {}".format(var,maxvals[i]))
    print("min for {}: {}".format(var,minvals[i]))
    i = i + 1

fac = 1.0
gas_conc_ranges = {
gas_names_rfmip[0]:      [minvals[0],    fac*maxvals[0]],
gas_names_rfmip[1]:      [minvals[1],    fac*maxvals[1]],
gas_names_rfmip[2]:      [minvals[2],    fac*maxvals[2]],
gas_names_rfmip[3]:      [minvals[3],    fac*maxvals[3]],
gas_names_rfmip[4]:      [minvals[4],    fac*maxvals[4]],
gas_names_rfmip[5]:      [minvals[5],    fac*maxvals[5]],
gas_names_rfmip[6]:      [minvals[6],    fac*maxvals[6]],
gas_names_rfmip[7]:      [minvals[7],    fac*maxvals[7]],
gas_names_rfmip[8]:      [minvals[8],    fac*maxvals[8]],
gas_names_rfmip[9]:      [minvals[9],    fac*maxvals[9]],
gas_names_rfmip[10]:     [minvals[10],   fac*maxvals[10]],
gas_names_rfmip[11]:     [minvals[11],   fac*maxvals[11]],
gas_names_rfmip[12]:     [minvals[12],   fac*maxvals[12]],
gas_names_rfmip[13]:     [minvals[13],   fac*maxvals[13]],
}

# or

gas_conc_ranges = {'carbon_dioxide':  [0.0,4.4]}

gas_conc_ranges['mygas'] = [0.0,     4.4]

# nsamp = 140
# samples_HALTON = build.halton(gas_conc_ranges, num_samples = nsamp)

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
    if predictand==1:
        y_means = ymeans_sw
        sigma   = ysigmas_sw
        neurons = [36,36]
        nfac = 8
    elif predictand==2:
        neurons = [36,36]
        nfac = 8
        y_means = ymeans_sw_ray
        sigma = ysigma_sw_ray
    elif predictand==3:
        neurons = [36,36]
        nfac = 8
        y_means = ymeans_sw_abs
        sigma = ysigma_sw_abs
    y_tr    = gptnorm_numba(nfac,y_tr,y_means,sigma)
    y_val   = gptnorm_numba(nfac,y_val,y_means,sigma)
    y_test  = gptnorm_numba(nfac,y_test,y_means,sigma)
else:
    if predictand==1:
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


# ------------- RECOMPILE WITH MEAN-ABS-ERR -------------
model.compile(loss=losses.mean_absolute_error, optimizer=optim,metrics=['mean_squared_error'])
earlystopper = EarlyStopping(monitor='val_loss',  patience=patience, verbose=1, mode='min',restore_best_weights=True)
history2 = model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size, shuffle=True,  verbose=1, validation_data=(x_val,y_val), callbacks=[earlystopper])
gc.collect()


# ------------- RECOMPILE WITH MEAN-SQUARED_ERR -------------
model.compile(loss=losses.mean_squared_error, optimizer=optim,metrics=['mean_absolute_error'])
earlystopper = EarlyStopping(monitor='val_loss',  patience=patience, verbose=1, mode='min',restore_best_weights=True)
history = model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size, shuffle=True,  verbose=1, validation_data=(x_val,y_val), callbacks=[earlystopper])
gc.collect()



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

# SHORT-WAVE RESULTS

# tau-tot
# gptnorm-orig
# 24-24 : MAE .0114 | .9976 | .99985
# 42-42 : MAE .0103 | .9976 | .99985
# gptnorm-stdscaler
# 42-42 : MAE .0309 | .9980 | .99988
# 36-36 : MAE .0315 | .9972 | .99987 | mse .00002

# SSA
# 36-36 : .99914 | mse .00012

# TAU-RAY
# 36-36 22 epochs:  .99944  | .99946
# 16-16 epochs:     .99964  | .99965

# TAU-ABS
# 16-16             .9967   | .99972
# 36-36             .9982   | .99988
# 48-48             .9989   | .999990

# LONG-WAVE RESULTS

# Tau
 # .99946   |  .9995
 
# Super long trained 64 64 model
#   . 99996, . 99993


# ------------------ SAVE AND LOAD MODELS ------------------ 


# kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/tau-sw-ray-7-16-16_2.h5"
# kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/tau-lw-18-58-58_2.h5"
# kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/pfrac-18-16-16.h5"

# from keras.models import load_model
# kerasfile = rootdir+"soft/rte-rrtmgp-nn/neural/data/tau-sw-ray-7-16-16.h5"
# model = load_model(kerasfile,compile=False)

# savemodel(kerasfile, model)


# SANITY CHECK
# data_name = 'rfmip'
# # data_name = 'nwpsaf'
# data_name = 'cams'
# # data_name = 'gcm'
# # data_name = 'garand'
# i0 = inds[data_name][0]; i1 = inds[data_name][1]
# np.corrcoef(x_tr[i0:i1,1],y_tr[i0:i1,:].mean(axis=1))
# np.corrcoef(x_val[:,1],y_val[:,:].mean(axis=1))





# ------------- SAVE MODEL AT EVERY EPOCH  -------------
# from keras.callbacks import ModelCheckpoint_totxt, EarlyStopping


# lr          = 0.001 
# lr          = 0.5* lr
# optim       = optimizers.Adam(lr=lr,rescale_grad=1/batch_size)

# fpath = rootdir+"soft/rte-rrtmgp-nn/neural/data/tau-sw-ray-tmp.h5"
# checkpointer = ModelCheckpoint_totxt(filepath=fpath, monitor='val_loss',verbose=1,period=1)
# earlystopper = EarlyStopping(monitor='val_loss',  patience=patience, verbose=1, mode='min',restore_best_weights=True)

# model.compile(loss=losses.mean_absolute_error, optimizer=optim,metrics=['mean_squared_error'])
# history2 = model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size, shuffle=True,  verbose=1, validation_data=(x_val,y_val), 
#                      callbacks=[earlystopper, checkpointer])
# gc.collect()


# # ------------- RECOMPILE WITH MEAN-SQUARED_ERR -------------
# earlystopper = EarlyStopping(monitor='val_loss',  patience=patience, verbose=1, mode='min',restore_best_weights=True)
# model.compile(loss=losses.mean_squared_error, optimizer=optim,metrics=['mean_absolute_error'])
# history = model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size, shuffle=True,  verbose=1, validation_data=(x_val,y_val),
#                     callbacks=[earlystopper, checkpointer])
# gc.collect()


# # ------------- RECOMPILE WITH MEAN-ABS-ERR -------------

# model.compile(loss=losses.mean_absolute_error, optimizer=optim,metrics=['mean_squared_error'])
# history2 = model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size, shuffle=True,  verbose=1, validation_data=(x_val,y_val), 
#                      callbacks=[earlystopper, checkpointer])
# gc.collect()