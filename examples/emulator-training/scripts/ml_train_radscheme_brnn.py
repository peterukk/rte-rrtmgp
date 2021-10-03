#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing neural networks to replace radiative
transfer computations, either fully or just one component

This code is for emulating RTE+RRTMGP (entire radiation scheme)

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
import time

from ml_loaddata import load_radscheme_rnn, preproc_divbymax, \
    preproc_pow_gptnorm, preproc_pow_gptnorm_reverse
from ml_eval_funcs import plot_hist2d
import matplotlib.pyplot as plt

def build_y(rsd_s,rsu_s,rsd0,albedo):
    nlay = rsd_s.shape[-1]
    ns = rsd_s.shape[0]
    rsd_pred = np.zeros((ns,nlay+1))
    rsu_pred = np.zeros((ns,nlay+1))
    rsd_pred[:,0] =  rsd0
    rsd_pred[:,1:] = rsd_s * rsd0.reshape(-1,1).repeat(nlay,axis=1)
    
    rsu_pred[:,-1] = albedo * rsd_pred[:,-1]
    rsu_pred[:,0:-1] = rsu_s * rsd0.reshape(-1,1).repeat(nlay,axis=1)
    return rsd_pred, rsu_pred
    
def calc_heatingrates(y, p):
    fluxup = y[:,0:61]
    fluxdn = y[:,61:]
    F = fluxdn - fluxup
    dF = np.gradient(F,axis=1)
    dp = np.gradient(p,axis=1)
    dFdp = dF/dp
    g = 9.81 # m s-2
    cp = 1004 # J K-1  kg-1
    dTdt = -(g/cp)*(dFdp) # K / s
    dTdt_day = (24*3600)*dTdt
    return dTdt_day

def rmse(predictions, targets,ax=0):
    return np.sqrt(((predictions - targets) ** 2).mean(axis=ax))

def mse(predictions, targets,ax=0):
    return ((predictions - targets) ** 2).mean(axis=ax)

def mae(predictions,targets,ax=0):
    diff = predictions - targets
    return np.mean(np.abs(diff),axis=ax)

def plot_heatingrate_error(hr_true, hr_pred, pres):
    # errfunc = mae
    errfunc = rmse
    ind_p = 5
    hre_radscheme       = errfunc(hr_true[:,ind_p:], hr_pred[:,ind_p:])
    yy = 0.01*pres[:,:].mean(axis=0)
    figtitle = 'ERrror in shortwave heating rate'
    fig, ax = plt.subplots(1)
    ax.plot(hre_radscheme,  yy[ind_p:])
    ax.invert_yaxis(); ax.grid()
    ax.set_ylabel('Pressure (hPa)',fontsize=15)
    ax.set_xlabel('Heating rate (W m$^{-2}$)',fontsize=15); 
    fig.suptitle(figtitle, fontsize=16)
    
def plot_flux_and_hr_error(fluxup_true, fluxdn_true, fluxup_pred, fluxdn_pred, pres):
    dF_true = np.gradient((fluxdn_true - fluxup_true),axis=1)
    dF_pred = np.gradient((fluxdn_pred - fluxup_pred),axis=1)
    dp = np.gradient(pres,axis=1)
    dFdp_true = dF_true/dp; dFdp_pred = dF_pred/dp
    g = 9.81; cp = 1004 
    dTdt_true = -(24*3600)*(g/cp)*(dFdp_true) # K / h
    dTdt_pred = -(24*3600)*(g/cp)*(dFdp_pred) # K / h
    bias_tot = np.mean(dTdt_pred.flatten()-dTdt_true.flatten())
    rmse_tot = rmse(dTdt_true.flatten(), dTdt_pred.flatten())
    mae_tot = mae(dTdt_true.flatten(), dTdt_pred.flatten())
    mae_percent = 100 * np.abs(mae_tot / dTdt_true.mean())
    r2 =  np.corrcoef(dTdt_pred.flatten(),dTdt_true.flatten())[0,1]; r2 = r2**2
    str_hre = 'Heating rate error \nR$^2$: {:0.4f} \nBias: {:0.3f} \nRMSE: {:0.3f} \nMAE: {:0.3f} ({:0.1f}%)'.format(r2,
                                    bias_tot,rmse_tot, mae_tot, mae_percent)
    mae_rsu = mae(fluxup_true.flatten(), fluxup_pred.flatten())
    mae_rsd = mae(fluxdn_true.flatten(), fluxdn_pred.flatten())
    mae_rsu_p = 100 * np.abs(mae_rsu / fluxup_true.mean())
    mae_rsd_p = 100 * np.abs(mae_rsd / fluxdn_true.mean())

    str_rsu =  'Upwelling flux error \nMAE: {:0.2f} ({:0.1f}%)'.format(mae_rsu, mae_rsu_p)
    str_rsd =  'Downwelling flux error \nMAE: {:0.2f} ({:0.1f}%)'.format(mae_rsd, mae_rsd_p)
    errfunc = mae
    #errfunc = rmse
    ind_p = 5
    hr_err      = errfunc(dTdt_true[:,ind_p:], dTdt_pred[:,ind_p:])
    fluxup_err  = errfunc(fluxup_true[:,ind_p:], fluxup_pred[:,ind_p:])
    fluxdn_err  = errfunc(fluxdn_true[:,ind_p:], fluxdn_pred[:,ind_p:])
    fluxnet_err  = errfunc((fluxdn_true[:,ind_p:] - fluxup_true[:,ind_p:]), \
                           (fluxdn_pred[:,ind_p:] - fluxup_pred[:,ind_p:] ))
    yy = 0.01*pres[:,:].mean(axis=0)
    fig, (ax0,ax1) = plt.subplots(ncols=2, sharey=True)
    ax0.plot(hr_err,  yy[ind_p:], label=str_hre)
    ax0.invert_yaxis()
    ax0.set_ylabel('Pressure (hPa)',fontsize=15)
    ax0.set_xlabel('Heating rate (K h$^{-1}$)',fontsize=15); 
    ax1.set_xlabel('Flux (W m$^{-2}$)',fontsize=15); 
    ax1.plot(fluxup_err,  yy[ind_p:], label=str_rsu)
    ax1.plot(fluxdn_err,  yy[ind_p:], label=str_rsd)
    ax1.plot(fluxnet_err,  yy[ind_p:], label='Net flux error')
    ax0.legend(); ax1.legend()
    ax0.grid(); ax1.grid()

# ----------------------------------------------------------------------------
# ----------------- RTE+RRTMGP EMULATION  ------------------------
# ----------------------------------------------------------------------------

#  ----------------- File paths -----------------
datadir     = "/media/peter/samsung/data/CAMS/ml_training/"
# datadir     = "/home/puk/data/"
fpath       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.nc"
fpath_val   = datadir + "/RADSCHEME_data_g224_CAMS_2014.nc"
# fpath_test  = datadir +  "/RADSCHEME_data_g224_CAMS_2015.nc"
fpath_test  = datadir +  "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
# fpath_test  = datadir +  "/RADSCHEME_data_g224_NWPSAFtest.nc"

# ----------- config ------------

scale_inputs    = True
scale_outputs   = False

# didn't seem to improve results
# include_deltap = True
include_deltap = False


# ----------- config ------------

# Load data
x_tr_raw, y_tr_raw, rsd0_tr, rsu0_tr, rsd_tr, rsu_tr, pres_tr = load_radscheme_rnn(fpath,  \
                        scale_p_h2o_o3 = scale_inputs, return_pressures=True)

x_val_raw, y_val_raw, rsd0_val, rsu0_val,rsd_val,rsu_val,  pres_val = load_radscheme_rnn(fpath_val,  \
                        scale_p_h2o_o3 = scale_inputs, return_pressures=True)

x_test_raw, y_test_raw, rsd0_test, rsu0_test, rsd_test, rsu_test, pres_test = load_radscheme_rnn(fpath_test,  \
                        scale_p_h2o_o3 = scale_inputs, return_pressures=True)


# Number of inputs and outputs    
nx = x_tr_raw.shape[-1]
ny = y_tr_raw.shape[-1]   
nlay = x_tr_raw.shape[-2]


if scale_inputs:
    x_tr        = np.copy(x_tr_raw)
    x_val       = np.copy(x_val_raw)
    x_test      = np.copy(x_test_raw)
    # x[0:7] gas concentrations, x[7] lwp, x[8] iwp, x[9] mu0, x[10] sfc_alb
    xmax = np.array([3.20104980e+02, 1.15657101e+01, 4.46762830e-01, 5.68951890e-02,
           6.59545418e-04, 3.38450207e-07, 5.08802714e-06, 2.13372910e+02,
           1.96923096e+02, 1.00000000e+00, 1.00],dtype=np.float32)
    # We do not scale albedo values to (0..1) because its already in the 
    # approximate range and we need the true albedo value for postprocessing
    # the max here for x_tr[:,:,index_albedo] is around 0.85
    x_tr            = preproc_divbymax(x_tr_raw, xmax)
    # x_tr, xmax      = preproc_divbymax(x_tr_raw)
    x_val           = preproc_divbymax(x_val_raw, xmax)
    x_test          = preproc_divbymax(x_test_raw, xmax)
else:
    x_tr    = x_tr_raw
    x_val   = x_val_raw
    x_test  = x_test_raw
    
    

rsd0_tr_big     = rsd0_tr.reshape(-1,1).repeat(nlay,axis=1)
rsd0_val_big    = rsd0_val.reshape(-1,1).repeat(nlay,axis=1)
rsd0_test_big   = rsd0_test.reshape(-1,1).repeat(nlay,axis=1)

x_tr_m = x_tr[:,:,0:-2];  x_val_m = x_val[:,:,0:-2];  x_test_m = x_test[:,:,0:-2]
x_tr_aux = x_tr[:,0,-2:]; x_val_aux = x_val[:,0,-2:]; x_test_aux = x_test[:,0,-2:]

nx_aux = 2
nx_main = nx - nx_aux

if include_deltap:
    pmax = 4083.2031
    dpres_tr    = np.expand_dims(pres_tr[:,1:] - pres_tr[:,0:-1],axis=2) / pmax
    dpres_val    = np.expand_dims(pres_val[:,1:] - pres_val[:,0:-1],axis=2) / pmax
    dpres_test    = np.expand_dims(pres_test[:,1:] - pres_test[:,0:-1],axis=2) / pmax

    x_tr_m      = np.concatenate((x_tr_m, dpres_tr), axis=2)
    x_val_m     = np.concatenate((x_val_m, dpres_val), axis=2)
    x_test_m    = np.concatenate((x_test_m, dpres_test), axis=2)
    
    nx = nx + 1
    nx_main = nx_main + 1
    
if scale_outputs: 
    print("do stuff")
else:
    y_tr    = y_tr_raw    
    y_val   = y_val_raw
    y_test  = y_test_raw
    
    
hre_loss = True
if hre_loss:
    dp_tr = np.gradient(pres_tr,axis=1)
    dp_val = np.gradient(pres_val,axis=1)
    dp_test = np.gradient(pres_test,axis=1)


gc.collect()
# Ready for training


import tensorflow as tf
from tensorflow.keras import losses, optimizers, layers, Input, Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed

mymetrics   = ['mean_absolute_error']
valfunc     = 'val_mean_absolute_error'

# Model architecture
# First hidden layer (input layer) activation
activ0      = 'softsign'
# activ0       = 'relu'
# Activation in other hidden layers
activ       =  activ0    
# Activation for last layer
activ_last   = 'relu'
# activ_last   = 'sigmoid'

epochs      = 100000
patience    = 25
lossfunc    = losses.mean_squared_error
lr          = 0.0001 
batch_size  = 1024

#                   up      dn      HR      HR-bias     loss
# 64, LSTM: MAE up  1.26,   1.13,   1.4                                   3.44 s
# 96, LSTM:         1.05,   0.98,   1.0,    0.3,        8.4287e-06 
# 128, SimpleRNN    4.22
# GRU               1.12,   1.15,   1.42,   0.09,       loss: 1.0998e-05

# using scalar features to predict initial state instead: 
# 96, GRU,'ave'     1.07,   1.02,   0.84,               7.e-06           3.646s
# 96, SimpleRNN: bad
# 64, GRU,'ave'     1,32,   1.16,   1.14
# 64, GRU, 'mult'   1.12,   0.77,   1.57,   0.244       8.2e-06

# --:--,  relu slightly better than sigmoid?

    
    
   
def my_gradient_tf(a):
    rght = tf.concat((a[..., 1:], tf.expand_dims(a[..., -1], -1)), -1)
    left = tf.concat((tf.expand_dims(a[...,0], -1), a[..., :-1]), -1)
    ones = tf.ones_like(rght[..., 2:], tf.float32)
    one = tf.expand_dims(ones[...,0], -1)
    divi = tf.concat((one, ones*2, one), -1)
    return (rght-left) / divi

def calc_heatingrates_tf_dp(flux_dn, flux_up, dp):
    #  flux_net =   flux_up   - flux_dn
    F = tf.subtract(flux_up, flux_dn)
    dF = my_gradient_tf(F)
    dFdp = tf.divide(dF, dp)
    coeff = -844.2071713147411#  -(24*3600) * (9.81/1004)  
    dTdt_day = tf.multiply(coeff, dFdp)
    return dTdt_day

def flux_from_y(y_true,y_pred, x_aux, rsd_top):
    # rsd_top is already shaped as (batchsize, nlev)
    # scale flux_down by rsd_top and add rsd_top 
    rsd_true = tf.math.multiply(y_true[:,:,0], rsd_top)
    rsd_pred = tf.math.multiply(y_pred[:,:,0], rsd_top)
    rsd_true = tf.concat([rsd_top[:,0:1], rsd_true],axis=-1)
    rsd_pred = tf.concat([rsd_top[:,0:1], rsd_pred],axis=-1)
    # scale flux_up by rsd_top
    rsu_true = tf.math.multiply(y_true[:,:,1], rsd_top)
    rsu_pred = tf.math.multiply(y_pred[:,:,1], rsd_top)
    
    # # upwelling flux at lowest level: dw flux times albedo
    rsu0_true = tf.math.multiply(rsd_true[:,-1], x_aux[:,1])
    rsu0_pred = tf.math.multiply(rsd_pred[:,-1], x_aux[:,1])          
                                 
    rsu_true = tf.concat([rsu_true,tf.expand_dims(rsu0_true,axis=1)],axis=-1)
    rsu_pred = tf.concat([rsu_pred,tf.expand_dims(rsu0_pred,axis=1)],axis=-1)
    return rsd_true, rsu_true, rsd_pred, rsu_pred

def CustomLoss(y_true, y_pred, x_aux, dp, rsd_top):
    # err_flux = K.sqrt(K.mean(K.square(y_true - y_pred)))
    err_flux = K.mean(K.square(y_true - y_pred))

    rsd_true, rsu_true, rsd_pred, rsu_pred = flux_from_y(y_true, y_pred, x_aux, rsd_top)

    HR_true = calc_heatingrates_tf_dp(rsd_true, rsu_true, dp)
    HR_pred = calc_heatingrates_tf_dp(rsd_pred, rsu_pred, dp)
    err_hr = K.sqrt(K.mean(K.square(HR_true - HR_pred)))
    
    # alpha   = 1e-6
    alpha   = 1e-5
    # alpha   = 1e-4
    # alpha = 0.3

    # alpha   = 0.0

    return (alpha) * err_hr + (1 - alpha)*err_flux   

def rmse_hr(y_true, y_pred, x_aux, dp, rsd_top):
    
    rsd_true, rsu_true, rsd_pred, rsu_pred = flux_from_y(y_true, y_pred, x_aux, rsd_top)

    HR_true = calc_heatingrates_tf_dp(rsd_true, rsu_true, dp)
    HR_pred = calc_heatingrates_tf_dp(rsd_pred, rsu_pred, dp)

    return K.sqrt(K.mean(K.square(HR_true - HR_pred)))
   
def rmse_flux(y_true, y_pred, x_aux, rsd_top):
        
    rsd_true, rsu_true, rsd_pred, rsu_pred = flux_from_y(y_true, y_pred, x_aux, rsd_top)

    flux_true = tf.concat([rsd_true,rsu_true],axis=-1)
    flux_pred = tf.concat([rsd_pred,rsu_pred],axis=-1)
  
    return K.sqrt(K.mean(K.square(flux_true - flux_pred)))

    
# batch_size = 512

# nneur = 64 
# nneur = 32   
nneur = 96
# neur  = 128
# Input for variable-length sequences of integers
# inputs = Input(shape=(None,nlay,nx))

# mergemode = 'concat'
# # mergemode = 'sum' #worse
# mergemode = 'ave' # better
mergemode = 'mul' # best?


# shape(x_tr_m) = (nsamples, nseq, nfeatures_seq)
# shape(x_tr_s) = (nsamples, nfeatures_aux)
# shape(y_tr)   = (nsamples, nseq, noutputs)

activ0 = 'relu'
#activ0 = 'softsign' #worse?

# Main inputs associated with RNN layer (sequence dependent)
inputs = Input(shape=(None,nx_main),name='inputs_main')
# Auxiliary inputs that do not dependend on sequence
inp_aux = Input(shape=(nx_aux),name='inputs_aux')
# Target outputs: these are fed as part of the input to avoid problem with 
# validation data where TF complained about wrong shape
target  = Input((None,ny))
# other inputs required to compute heating rate
dpres   = Input((nlay+1,))
incflux = Input((nlay))

# aux input layer; this predicts the initial state of the RNN from the
# auxiliary inputs; the dense layer mapping transforms them into right shape
mlp_dense_inp1 = Dense(nneur, activation=activ0,name='dense_inputs_aux1')(inp_aux)
mlp_dense_inp2 = Dense(nneur, activation=activ0,name='dense_inputs_aux2')(inp_aux)

# The Bidirectional RNN layer
# layer_rnn = layers.SimpleRNN(nneur,return_sequences=True)
layer_rnn = layers.GRU(nneur,return_sequences=True)
# layer_rnn = layers.LSTM(nneur,return_sequences=True)

hidden = layers.Bidirectional(layer_rnn, merge_mode=mergemode, name ='bidirectional')\
    (inputs, initial_state= [mlp_dense_inp1,mlp_dense_inp2])
# outputs
outputs = TimeDistributed(layers.Dense(ny, activation=activ_last),name='dense_output')(hidden)
#
model = Model(inputs=[inputs, inp_aux, target, dpres, incflux], outputs=outputs)
# model.add_metric(math_ops.reduce_sum(x), name='metric_1')

model.add_metric(rmse_hr(target,outputs,inp_aux,dpres,incflux),'rmse_hr')
# model.add_metric(rmse_flux(target,outputs,inp_aux,incflux),'rmse_flux')

model.add_loss(CustomLoss(target,outputs,inp_aux,dpres, incflux))
model.compile(optimizer='adam', loss='mse')

model.summary()

# model.add_metric(heatingrateloss,'heating_rate_mse')
# model.metrics_names.append("heating_rate_mse")

# # Create earlystopper and possibly other callbacks
callbacks = [EarlyStopping(monitor='rmse_hr',  patience=patience, verbose=1, \
                             mode='min',restore_best_weights=True)]


# START TRAINING
history = model.fit(x=[x_tr_m, x_tr_aux, y_tr, dp_tr, rsd0_tr_big], y=None, \
    epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1,  \
    validation_data=[x_val_m, x_val_aux, y_val, dp_val, rsd0_val_big], callbacks=callbacks)  
    


# EVALUATE
# validation data
start = time.time()
y_pred_val      = model.predict([x_val_m, x_val_aux, y_val, dp_val, rsd0_val_big]);  
end = time.time()
print(end - start)
rsd_pred_val, rsu_pred_val = build_y(y_pred_val[:,:,0],y_pred_val[:,:,1], rsd0_val, x_val_aux[:,1])
plot_flux_and_hr_error(rsu_val, rsd_val, rsu_pred_val, rsd_pred_val, pres_val)


# test data
y_pred_test      = model.predict([x_test_m, x_test_aux, y_test, dp_test, rsd0_test_big]);  
rsd_pred_test, rsu_pred_test = build_y(y_pred_test[:,:,0],y_pred_test[:,:,1], rsd0_test, x_test_aux[:,1])
plot_flux_and_hr_error(rsu_test, rsd_test, rsu_pred_test, rsd_pred_test, pres_test)

# save fluxes for test data

from netCDF4 import Dataset

rootdir = "../fluxes/"
fname_out = rootdir+'CAMS_2015_rsud_RADSCHEME_RNN.nc'

dat_out =  Dataset(fname_out,'a')
var_rsu = dat_out.variables['rsu']
var_rsd = dat_out.variables['rsd']

nsite = dat_out.dimensions['site'].size
ntime = dat_out.dimensions['time'].size
var_rsu[:] = rsu_pred_test.reshape(ntime,nsite,nlay+1)
var_rsd[:] = rsd_pred_test.reshape(ntime,nsite,nlay+1)

dat_out.close()


del inputs,inp_aux,target,mlp_dense_inp,layer_rnn,hidden,outputs,model



# inputs = Input(shape=(nlay,nx))
# layer_rnn = layers.GRU(nneur,return_sequences=True)
# hidden = layers.Bidirectional(layer_rnn, merge_mode=mergemode)(inputs)
# layer_dense = layers.Dense(ny, activation="sigmoid")
# outputs = TimeDistributed(layer_dense)(hidden)
# model = Model(inputs, outputs)
# model.compile(optimizer='adam', loss='mse')
# model.summary()



def flux_from_y2(y_true,y_pred, x, rsd_top):
    # rsd_top is already shaped as (batchsize, nlev)
    # scale flux_down by rsd_top and add rsd_top 
    rsd_true = tf.math.multiply(y_true[:,:,0], rsd_top)
    rsd_pred = tf.math.multiply(y_pred[:,:,0], rsd_top)
    rsd_true = tf.concat([rsd_top[:,0:1], rsd_true],axis=-1)
    rsd_pred = tf.concat([rsd_top[:,0:1], rsd_pred],axis=-1)
    # scale flux_up by rsd_top
    rsu_true = tf.math.multiply(y_true[:,:,1], rsd_top)
    rsu_pred = tf.math.multiply(y_pred[:,:,1], rsd_top)
    # # upwelling flux at lowest level: dw flux times albedo
    rsu0_true = tf.math.multiply(rsd_true[:,-1], x[:,0,-1])
    rsu0_pred = tf.math.multiply(rsd_pred[:,-1], x[:,0,-1])
    rsu_true = tf.concat([rsu_true,tf.expand_dims(rsu0_true,axis=1)],axis=-1)
    rsu_pred = tf.concat([rsu_pred,tf.expand_dims(rsu0_pred,axis=1)],axis=-1)
    return rsd_true, rsu_true, rsd_pred, rsu_pred

def CustomLoss2(y_true, y_pred, x, dp, rsd_top):
    err_flux = K.mean(K.square(y_true - y_pred))

    rsd_true, rsu_true, rsd_pred, rsu_pred = flux_from_y2(y_true, y_pred, x, rsd_top)
    HR_true = calc_heatingrates_tf_dp(rsd_true, rsu_true, dp)
    HR_pred = calc_heatingrates_tf_dp(rsd_pred, rsu_pred, dp)
    err_hr = K.sqrt(K.mean(K.square(HR_true - HR_pred)))
    
    # alpha   = 1e-5
    alpha   = 1e-4
    alpha   = 0.0
    return (alpha) * err_hr + (1 - alpha)*err_flux   

def rmse_hr2(y_true, y_pred, x, dp, rsd_top):
    
    rsd_true, rsu_true, rsd_pred, rsu_pred = flux_from_y2(y_true, y_pred, x, rsd_top)
    HR_true = calc_heatingrates_tf_dp(rsd_true, rsu_true, dp)
    HR_pred = calc_heatingrates_tf_dp(rsd_pred, rsu_pred, dp)

    return K.sqrt(K.mean(K.square(HR_true - HR_pred)))



inputs = Input(shape=(None,nx),name='inputs_main')
target  = Input((None,ny))
# other inputs required to compute heating rate
dpres   = Input((nlay+1,))
incflux = Input((nlay))

# The Bidirectional RNN layer
# layer_rnn = layers.GRU(nneur,return_sequences=True)
layer_rnn = layers.LSTM(nneur,return_sequences=True)

hidden = layers.Bidirectional(layer_rnn, merge_mode=mergemode, name ='bidirectional')(inputs)
outputs = TimeDistributed(layers.Dense(ny, activation=activ_last),name='dense_output')(hidden)
model = Model(inputs=[inputs, target, dpres, incflux], outputs=outputs)

model.add_metric(rmse_hr2(target,outputs,inputs, dpres,incflux),'rmse_hr')
# model.add_metric(rmse_flux(target,outputs,inp_aux,incflux),'rmse_flux')

model.add_loss(CustomLoss2(target,outputs, inputs, dpres, incflux))
model.compile(optimizer='adam', loss='mse')

model.summary()


# # Create earlystopper and possibly other callbacks
callbacks = [EarlyStopping(monitor='rmse_hr',  patience=patience, verbose=1, \
                             mode='min',restore_best_weights=True)]
    
# START TRAINING
history = model.fit(x=[x_tr, y_tr, dp_tr, rsd0_tr_big], y=None, \
    epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1,  \
    validation_data=[x_val, y_val, dp_val, rsd0_val_big], callbacks=callbacks)  
    
    

start = time.time()
y_pred_val      = model.predict([x_val, y_val, dp_val, rsd0_val_big]);  
end = time.time()
print(end - start)



# # Create earlystopper and possibly other callbacks
# earlystopper = EarlyStopping(monitor='val_loss',  patience=patience, verbose=1, mode='min',restore_best_weights=True)
# callbacks = [earlystopper]

# # START TRAINING
# history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
#                     validation_data=(x_val,y_val), callbacks=callbacks)    
        
# Evaluate the model both on validation data, and on TEST data where the
# number of vertical levels is different
# y_pred_val      = model.predict(x_val);  
# rsd_pred_val, rsu_pred_val = build_y(y_pred_val[:,:,0],y_pred_val[:,:,1], rsd0_val,albedo_val)

# plot_flux_and_hr_error(rsu_val, rsd_val, rsu_pred_val, rsd_pred_val, pres_val)

# y_pred      = model.predict(x_test);  
# rsd_pred, rsu_pred = build_y(y_pred[:,:,0],y_pred[:,:,1], rsd0_test,albedo_test)
# plot_flux_and_hr_error(rsu_test, rsd_test, rsu_pred, rsd_pred, pres_test)

# Fnet_test = rsd_test - rsu_test
# Fnet_pred = rsd_pred - rsu_pred
# fnet_err = mae(Fnet_test,Fnet_pred)

# for i in range(11): 
#     print("TR i = {} mean {:0.2f} min {:0.2f} max {:0.2f}".format(i,x_tr[:,:,i].mean(), \
#                   x_tr[:,:,i].min(), x_tr[:,:,i].max()))
#     print("TEST i = {} mean {:0.2f} min {:0.2f} max {:0.2f}".format(i,x_test[:,:,i].mean(), \
#                   x_test[:,:,i].min(), x_test[:,:,i].max()))    
        