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
    preproc_pow_standardization, preproc_pow_standardization_reverse
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
datadir     = "/home/peter/data/"
fpath       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.nc"
fpath_val   = datadir + "/RADSCHEME_data_g224_CAMS_2014.nc"
fpath_test  = datadir +  "/RADSCHEME_data_g224_CAMS_2014.nc"
# fpath_test  = datadir +  "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
# fpath_test  = datadir +  "/RADSCHEME_data_g224_NWPSAFtest.nc"

# ----------- config ------------

predict_hr = True

scale_inputs    = True
scale_outputs   = False

# didn't seem to improve results
# include_deltap = True
include_deltap = False

include_coldry = True

only_albedo_as_auxinput = True # 
mu0_and_albedo_as_auxinput = False

use_auxinputs = mu0_and_albedo_as_auxinput or only_albedo_as_auxinput

# Model training: use GPU or CPU?
# use_gpu = False

# ----------- config ------------

# Load data
if include_coldry:
    x_tr_raw, y_tr_raw, rsd0_tr, rsu0_tr, rsd_tr, rsu_tr, pres_tr, coldry_tr = \
        load_radscheme_rnn(fpath,  scale_p_h2o_o3 = scale_inputs, return_p=True, return_coldry=True)
    
    x_val_raw, y_val_raw, rsd0_val, rsu0_val,rsd_val,rsu_val,  pres_val, coldry_val = \
        load_radscheme_rnn(fpath_val, scale_p_h2o_o3 = scale_inputs, return_p=True, return_coldry=True)
    
    x_test_raw, y_test_raw, rsd0_test, rsu0_test, rsd_test, rsu_test, pres_test, coldry_test = \
        load_radscheme_rnn(fpath_test,  scale_p_h2o_o3 = scale_inputs, return_p=True, return_coldry=True)
else:
    x_tr_raw, y_tr_raw, rsd0_tr, rsu0_tr, rsd_tr, rsu_tr, pres_tr = \
        load_radscheme_rnn(fpath,  scale_p_h2o_o3 = scale_inputs, return_p=True)
    
    x_val_raw, y_val_raw, rsd0_val, rsu0_val,rsd_val,rsu_val,  pres_val = \
        load_radscheme_rnn(fpath_val, scale_p_h2o_o3 = scale_inputs, return_p=True)
    
    x_test_raw, y_test_raw, rsd0_test, rsu0_test, rsd_test, rsu_test, pres_test = \
        load_radscheme_rnn(fpath_test,  scale_p_h2o_o3 = scale_inputs, return_p=True)



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
    # approximate range and we need the true albedo value for postprocessing.
    # the max for x_tr[:,:,index_albedo] is around 0.85
    x_tr            = preproc_divbymax(x_tr_raw, xmax)
    # x_tr, xmax      = preproc_divbymax(x_tr_raw)
    x_val           = preproc_divbymax(x_val_raw, xmax)
    x_test          = preproc_divbymax(x_test_raw, xmax)
else:
    x_tr    = x_tr_raw
    x_val   = x_val_raw
    x_test  = x_test_raw
    

if scale_outputs: 
    # lets scale the (already physically scaled) upwelling fluxes so that they
    # have a similar mean to downwelling fluxes. 
    # Didn't seem to work very well, better to do weighting within the loss function
    #fac = y_tr[:,:,0].mean() / y_tr[:,:,1].mean()
    fac = 2.625
    y_tr= np.copy(y_tr_raw); y_val= np.copy(y_val_raw); y_test = np.copy(y_test_raw)
    y_tr[:,:,1] = fac*y_tr[:,:,1] 
    y_val[:,:,1] = fac*y_val[:,:,1] 
    y_test[:,:,1] = fac*y_test[:,:,1] 
else:
    y_tr = y_tr_raw; y_val = y_val_raw; y_test = y_test_raw


rsd0_tr_big     = rsd0_tr.reshape(-1,1).repeat(nlay,axis=1)
rsd0_val_big    = rsd0_val.reshape(-1,1).repeat(nlay,axis=1)
rsd0_test_big   = rsd0_test.reshape(-1,1).repeat(nlay,axis=1)

if not use_auxinputs: # everything as layer inputs
    x_tr_m = x_tr; x_val_m = x_val; x_test_m = x_test
    # add albedo as aux input anyway to use in loss function
    x_tr_aux1 = x_tr[:,0,-1:]; x_val_aux1 = x_val[:,0,-1:]; x_test_aux1 = x_test[:,0,-1:]  
    nx_aux = 1
else:
    if only_albedo_as_auxinput: # only one scalar input (albedo)
        x_tr_m = x_tr[:,:,0:-1];  x_val_m = x_val[:,:,0:-1];  x_test_m = x_test[:,:,0:-1]
        x_tr_aux1 = x_tr[:,0,-1:]; x_val_aux1 = x_val[:,0,-1:]; x_test_aux1 = x_test[:,0,-1:]    
        nx_aux = 1
    else: # two scalar inputs (mu0 and albedo)
        x_tr_m = x_tr[:,:,0:-2];  x_val_m = x_val[:,:,0:-2];  x_test_m = x_test[:,:,0:-2]
        # x_tr_aux = x_tr[:,0,-2:]; x_val_aux = x_val[:,0,-2:]; x_test_aux = x_test[:,0,-2:]
        
        x_tr_aux1 = x_tr[:,0,-1:]; x_tr_aux2 = x_tr[:,0,-2:-1]
        x_val_aux1 = x_val[:,0,-1:];  x_val_aux2 = x_val[:,0,-2:-1]; 
        x_test_aux1 = x_test[:,0,-1:]; x_test_aux2 = x_test[:,0,-2:-1]
        nx_aux = x_tr_aux1.shape[-1]

nx_main = x_tr_m.shape[-1]

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
    
if include_coldry:
    coldry_max = 8.159878628698766e+23
    
    x_tr_m      = np.concatenate((x_tr_m, np.expand_dims(coldry_tr/coldry_max,axis=2)), axis=2)
    x_val_m     = np.concatenate((x_val_m,  np.expand_dims(coldry_val/coldry_max,axis=2)), axis=2)
    x_test_m    = np.concatenate((x_test_m,  np.expand_dims(coldry_test/coldry_max,axis=2)), axis=2)
    nx = nx + 1
    nx_main = nx_main + 1
    
hre_loss = True
if hre_loss:
    dp_tr = np.gradient(pres_tr,axis=1)
    dp_val = np.gradient(pres_val,axis=1)
    dp_test = np.gradient(pres_test,axis=1)


# rsd_mean_tr = y_tr[:,:,0].mean(axis=0)
# rsu_mean_tr = y_tr[:,:,1].mean(axis=0)
# yy = 0.01*pres_tr[:,1:].mean(axis=0)
# fig, ax = plt.subplots()
# ax.plot(rsd_mean_tr,  yy, label='RSD')
# ax.plot(rsu_mean_tr,  yy, label='RSU')
# ax.invert_yaxis()
# ax.set_ylabel('Pressure (hPa)',fontsize=15)
# in the loss function, test a weight profile to normalize 
# everything to 1 (so that different levels have equal contributions)
weight_prof = 1/y_tr.mean(axis=0).data
# ax.plot(weight_prof[:,0],yy,label='RSD weight')
# ax.plot(weight_prof[:,1],yy,label='RSU weight')
# ax.legend(); ax.grid()

# Ready for training


import tensorflow as tf
from tensorflow.keras import losses, optimizers, layers, Input, Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense,TimeDistributed

mymetrics   = ['mean_absolute_error']
valfunc     = 'val_mean_absolute_error'

# if use_gpu:
#     devstr = '/gpu:0'
#     # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# else:
#     num_cpu_threads = 12
#     devstr = '/cpu:0'
#     # Maximum number of threads to use for OpenMP parallel regions.
#     os.environ["OMP_NUM_THREADS"] = str(num_cpu_threads)
#     # Without setting below 2 environment variables, it didn't work for me. Thanks to @cjw85 
#     os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_cpu_threads)
#     os.environ["TF_NUM_INTEROP_THREADS"] = str(1)
#     os.environ['KMP_BLOCKTIME'] = '1' 

#     tf.config.threading.set_intra_op_parallelism_threads(
#         num_cpu_threads
#     )
#     tf.config.threading.set_inter_op_parallelism_threads(
#         1
#     )
#     tf.config.set_soft_device_placement(True)
#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    
   
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
    rsu0_true = tf.math.multiply(rsd_true[:,-1], x_aux[:,0])
    rsu0_pred = tf.math.multiply(rsd_pred[:,-1], x_aux[:,0])          
                                 
    rsu_true = tf.concat([rsu_true,tf.expand_dims(rsu0_true,axis=1)],axis=-1)
    rsu_pred = tf.concat([rsu_pred,tf.expand_dims(rsu0_pred,axis=1)],axis=-1)
    return rsd_true, rsu_true, rsd_pred, rsu_pred

def CustomLoss(y_true, y_pred, x_aux, dp, rsd_top):
    # err_flux = K.sqrt(K.mean(K.square(y_true - y_pred)))
    # err_flux = K.mean(K.square(y_true - y_pred))
    err_flux = K.mean(K.square(weight_prof*(y_true - y_pred)),axis=-1)

    rsd_true, rsu_true, rsd_pred, rsu_pred = flux_from_y(y_true, y_pred, x_aux, rsd_top)

    HR_true = calc_heatingrates_tf_dp(rsd_true, rsu_true, dp)
    HR_pred = calc_heatingrates_tf_dp(rsd_pred, rsu_pred, dp)
    err_hr = K.sqrt(K.mean(K.square(HR_true - HR_pred)))
    # alpha   = 1e-6
    # alpha   = 1e-5
    alpha   = 1e-4
    return (alpha) * err_hr + (1 - alpha)*err_flux   


def rmse_hr(y_true, y_pred, x_aux, dp, rsd_top):
    
    rsd_true, rsu_true, rsd_pred, rsu_pred = flux_from_y(y_true, y_pred, x_aux, rsd_top)

    HR_true = calc_heatingrates_tf_dp(rsd_true, rsu_true, dp)
    HR_pred = calc_heatingrates_tf_dp(rsd_pred, rsu_pred, dp)

    return K.sqrt(K.mean(K.square(HR_true - HR_pred)))
   

# Model architecture
# Activation for first
activ0 = 'relu'
activ0 = 'linear'
activ_last   = 'relu'
activ_last   = 'sigmoid'
# activ_last   = 'linear'

epochs      = 100000
patience    = 25
lossfunc    = losses.mean_squared_error
batch_size  = 1024

# nneur = 64 
nneur = 16  
nneur = 32 
# neur  = 128
# Input for variable-length sequences of integers
# inputs = Input(shape=(None,nlay,nx))

mergemode = 'concat'
# # mergemode = 'sum' #worse
# mergemode = 'ave' # better
# mergemode = 'mul' # best?

# lr          = 0.0001 
# lr          = 0.0001 
lr = 0.001 # DEFAULT!
optim = optimizers.Adam(learning_rate=lr)

# shape(x_tr_m) = (nsamples, nseq, nfeatures_seq)
# shape(x_tr_s) = (nsamples, nfeatures_aux)
# shape(y_tr)   = (nsamples, nseq, noutputs)

# Main inputs associated with RNN layer (sequence dependent)
inputs = Input(shape=(None,nx_main),name='inputs_main')
# Optionally, use auxiliary inputs that do not dependend on sequence?

# if use_auxinputs:  # commented out cos I need albedo for the loss function, easier to have it separate
inp_aux_albedo = Input(shape=(nx_aux),name='inputs_aux_albedo') # sfc_albedo
if mu0_and_albedo_as_auxinput: inp_aux_mu = Input(shape=(nx_aux),name='inputs_aux_mu0') # mu0

# Target outputs: these are fed as part of the input to avoid problem with 
# validation data where TF complained about wrong shape
target  = Input((None,ny))
# other inputs required to compute heating rate
dpres   = Input((nlay+1,))
incflux = Input((nlay))

# aux input layer; this predicts the initial state of the RNN from the
# auxiliary inputs; the dense layer mapping transforms them into right shape
if use_auxinputs:
    mlp_dense_inp1 = Dense(nneur, activation=activ0,name='dense_inputs_albedo')(inp_aux_albedo)
    if mu0_and_albedo_as_auxinput:
        mlp_dense_inp2 = Dense(nneur, activation=activ0,name='dense_inputs_mu0')(inp_aux_albedo)
    else: # if only albedo is aux input, we still need two mlp layers, one for each RNN in the BiRNN 
        mlp_dense_inp2 = Dense(nneur, activation=activ0,name='dense_inputs_albedo2')(inp_aux_albedo)

# The Bidirectional RNN layer
layer_rnn = layers.SimpleRNN(nneur,return_sequences=True)
# layer_rnn = layers.GRU(nneur,return_sequences=True)

if use_auxinputs:
    hidden = layers.Bidirectional(layer_rnn, merge_mode=mergemode, name ='bidirectional')\
            (inputs, initial_state= [mlp_dense_inp1,mlp_dense_inp2])
else:
    hidden = layers.Bidirectional(layer_rnn, merge_mode=mergemode, name ='bidirectional')\
            (inputs)
            
            
# hidden2 = layers.GRU(nneur,return_sequences=True)(hidden)
# hidden2 = layers.SimpleRNN(nneur,return_sequences=True)(hidden)

# outputs = TimeDistributed(layers.Dense(ny, activation=activ_last),name='dense_output')(hidden2)
outputs = TimeDistributed(layers.Dense(ny, activation=activ_last),name='dense_output')(hidden)

# if use_auxinputs:
#     if only_albedo_as_auxinput:
#         model = Model(inputs=[inputs, inp_aux_albedo, target, dpres, incflux], outputs=outputs)
#     else:
#         model = Model(inputs=[inputs, inp_aux_mu, inp_aux_albedo, target, dpres, incflux], outputs=outputs)
# else:
#     model = Model(inputs=[inputs, target, dpres, incflux], outputs=outputs)
if mu0_and_albedo_as_auxinput:
    model = Model(inputs=[inputs, inp_aux_mu, inp_aux_albedo, target, dpres, incflux], outputs=outputs)
else:
    model = Model(inputs=[inputs, inp_aux_albedo, target, dpres, incflux], outputs=outputs)

model.add_metric(rmse_hr(target,outputs,inp_aux_albedo,dpres,incflux),'rmse_hr')
# model.add_metric(rmse_flux(target,outputs,inp_aux,incflux),'rmse_flux')

model.add_loss(CustomLoss(target,outputs,inp_aux_albedo,dpres, incflux))
model.compile(optimizer=optim,loss='mse')

model.summary()

# model.add_metric(heatingrateloss,'heating_rate_mse')
# model.metrics_names.append("heating_rate_mse")

# # Create earlystopper and possibly other callbacks
callbacks = [EarlyStopping(monitor='rmse_hr',  patience=patience, verbose=1, \
                             mode='min',restore_best_weights=True)]


# START TRAINING
# with tf.device(devstr):
    
if not mu0_and_albedo_as_auxinput:
    history = model.fit(x=[x_tr_m, x_tr_aux1, y_tr, dp_tr, rsd0_tr_big], y=None, \
    epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1,  \
    validation_data=[x_val_m, x_val_aux1, y_val, dp_val, rsd0_val_big], callbacks=callbacks)
else:
    history = model.fit(x=[x_tr_m, x_tr_aux1, x_tr_aux2, y_tr, dp_tr, rsd0_tr_big], y=None, \
        epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1,  \
        validation_data=[x_val_m, x_val_aux1, x_val_aux2, y_val, dp_val, rsd0_val_big], callbacks=callbacks)  
    
# 96 BiRNN , mul:               4.81 sec, Heating rate R2 0.75, sfc MAE 1.0
# 64, concat, lr 0.001 , rubbish
# -:-, 0.0001, alpha 1e-6 , again rubbish rmse_hr: 20.4619
# back to 96: rubbish 20.7712
# 96, concat, lr 0.0001, alpha 1e-6 ; rmse_hr 12
# BACK TO MUL!
# 96, mul lr 0.001, alpha 1e-6    HR R2 0.56
# retrain with alpha 1e-4,    HR R2 0.84

# 32, 2x BiRNN:  5.7s, low hr errors except at surface
# 32, 2x BiRNN concat, aux1=mu0,aux2=alb: HR R2 0.83, bias -0.5
# 32, 2x BiRNN mult, aux1=mu0, aux2=alb, HR R2 shit
# 32, 2x biRNN, concat, -:-, SIGMOIDLAST  HR R2 0.993 but sfc still a problem
# same but mu0 as layerinp: HR r2 0.996, sfc MAE 1.5
# same but 64, 1x, not so good
# 32, 1x BiRNN 1x RNN, mu0 lay: HR R2 0.996             9.6s RTX 3060
# # same but SimpleRNN
# Epoch 100/100000
# 320/320 [==============================] - 23s 72ms/step - loss: 7.9130e-04 - rmse_hr: 6.4901 - val_loss: 7.1506e-04 - val_rmse_hr: 5.5656
# validation data
# Epoch 260/100000
# 320/320 [==============================] - 20s 63ms/step - loss: 3.2985e-04 - rmse_hr: 4.6232 - val_loss: 2.8943e-04 - val_rmse_hr: 2.9510
# Remove last RNN, alpha 1e-
# again stuck at 4-5 HR-RMSE
# albedo removed as layer input: same or worse
# 16, 2x biRNN: 
#     Epoch 440/100000
# 320/320 [==============================] - 4s 13ms/step - loss: 3.6992e-04 - rmse_hr: 0.9837 - val_loss: 3.5444e-04 - val_rmse_hr: 0.9154

start = time.time()
if not mu0_and_albedo_as_auxinput:

    y_pred_val      = model.predict([x_val_m, x_val_aux1, y_val, dp_val, rsd0_val_big]);  
    end = time.time()
    print(end - start)
    
    if scale_outputs:   y_pred_val[:,:,1] = y_pred_val[:,:,1] /fac
    rsd_pred_val, rsu_pred_val = build_y(y_pred_val[:,:,0],y_pred_val[:,:,1], rsd0_val, x_val_aux1)

else:
    y_pred_val      = model.predict([x_val_m, x_val_aux1, x_val_aux2, y_val, dp_val, rsd0_val_big]); 
    end = time.time()
    print(end - start)
    if scale_outputs:   y_pred_val[:,:,1] = y_pred_val[:,:,1] /fac
    rsd_pred_val, rsu_pred_val = build_y(y_pred_val[:,:,0],y_pred_val[:,:,1], rsd0_val, x_val_aux1[:,1])

plot_flux_and_hr_error(rsu_val, rsd_val, rsu_pred_val, rsd_pred_val, pres_val)




# extract weights to save as simpler model without custom functions
# layers 2,3,4,5,9 have weights
all_weights = []
for layer in model.layers:
  w = layer.weights
  try:      
    w[0]
    all_weights.append(w)
  except:
    pass # not all layers have weights !


# make a new model without the functions
# inputs = Input(shape=(None,nx_main),name='inputs_main')
# if not only_albedo_as_auxinput: inp_aux_mu = Input(shape=(nx_aux),name='inputs_aux1') # mu0
# inp_aux_albedo = Input(shape=(nx_aux),name='inputs_aux2') # sfc_albedo
# if only_albedo_as_auxinput:
#     mlp_dense_inp1 = Dense(nneur, activation=activ0,name='dense_inputs_alb')(inp_aux_albedo)
# else:
#     mlp_dense_inp1 = Dense(nneur, activation=activ0,name='dense_inputs_alb')(inp_aux_mu)
# mlp_dense_inp2 = Dense(nneur, activation=activ0,name='dense_inputs_mu')(inp_aux_albedo)
# layer_rnn = layers.GRU(nneur,return_sequences=True)
# layer_rnn2 = layers.GRU(nneur,return_sequences=True)
# hidden = layers.Bidirectional(layer_rnn, merge_mode=mergemode, name ='bidirectional')\
#     (inputs, initial_state= [mlp_dense_inp1,mlp_dense_inp2])
# hidden2 = layer_rnn2(hidden)
# outputs = TimeDistributed(layers.Dense(ny, activation=activ_last),name='dense_output')(hidden2)
if only_albedo_as_auxinput:
    newmodel = Model(inputs=[inputs, inp_aux_albedo], outputs=outputs)
else:
    newmodel = Model(inputs=[inputs, inp_aux_mu, inp_aux_albedo], outputs=outputs)
newmodel.compile()
newmodel.summary()
# add weights
i = 0
for layer in newmodel.layers:
  w = layer.weights
  try:      
    w[0]
    layer.weights = all_weights[i]
    i = i + 1
  except:
    pass # not all layers have weights !



import onnxruntime as ort
import tf2onnx
# Test saving as SaveModel
# fpath = 'saved_model/tmp2_bigru_gru_32'
# fpath = 'saved_model/tmp2_bisimple_simple_32_100epochs'
fpath = 'saved_model/tmp_bigru_32'

newmodel.save(fpath, save_format='tf')
# newmodel.save(fpath, save_format='tf',save_traces=False)


fpath_onnx = fpath+".onnx"
os.system("python -m tf2onnx.convert --saved-model {} --output {} --opset 13".format(fpath,fpath_onnx)) 


# INFERENCE USING EXISTING ONNX MODEL

# providers = [
#     ('CUDAExecutionProvider', {
#         'device_id': 0,
#         'arena_extend_strategy': 'kNextPowerOfTwo',
#         'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
#         'cudnn_conv_algo_search': 'EXHAUSTIVE',
#         'do_copy_in_default_stream': True,
#     }),
#     'CPUExecutionProvider',
# ]


xaux = x_val_aux1.reshape(-1,1)


# python -m tf2onnx.convert --saved-model saved_model/tmp2_bigru_gru_32/ --output model.onnx --opset 13
# sess = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
sess = ort.InferenceSession(fpath_onnx)
# sess = ort.InferenceSession("model.onnx",providers=providers)

# sess.get_providers()

start = time.time()
results_ort = sess.run(["dense_output"], {"inputs_aux_albedo": xaux, "inputs_main": x_val_m})
end = time.time()
print(end - start)
# BiGRU+GRU 32 : 4.688 s
# GRU16 : 1.28s
# GRU16, softsign: 1.28 still?





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


