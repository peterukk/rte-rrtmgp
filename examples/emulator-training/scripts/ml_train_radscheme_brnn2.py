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

        

def calc_heatingrate(F, p):
    dF = F[:,1:] - F[:,0:-1] 
    dp = p[:,1:] - p[:,0:-1] 
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

def mape(a,b):
    mask = a != 0
    return 100*(np.fabs(a[mask] - b[mask])/a[mask]).mean()

def plot_flux_and_hr_error(rsu_true, rsd_true, rsu_pred, rsd_pred, pres):
    pres_lay = 0.5 * (pres[:,1:] + pres[:,0:-1])

    dTdt_true = calc_heatingrate(rsd_true - rsu_true, pres)
    dTdt_pred = calc_heatingrate(rsd_pred - rsu_pred, pres)
    
    bias_tot = np.mean(dTdt_pred.flatten()-dTdt_true.flatten())
    rmse_tot = rmse(dTdt_true.flatten(), dTdt_pred.flatten())
    mae_tot = mae(dTdt_true.flatten(), dTdt_pred.flatten())
    mae_percent = 100 * np.abs(mae_tot / dTdt_true.mean())
    # mae_percent = mape(dTdt_true.flatten(), dTdt_pred.flatten())
    r2 =  np.corrcoef(dTdt_pred.flatten(),dTdt_true.flatten())[0,1]; r2 = r2**2
    str_hre = 'Heating rate error \nR$^2$: {:0.4f} \nBias: {:0.3f} \nRMSE: {:0.3f} \nMAE: {:0.3f} ({:0.1f}%)'.format(r2,
                                    bias_tot,rmse_tot, mae_tot, mae_percent)
    mae_rsu = mae(rsu_true.flatten(), rsu_pred.flatten())
    mae_rsd = mae(rsd_true.flatten(), rsd_pred.flatten())
    mae_rsu_p = 100 * np.abs(mae_rsu / rsu_true.mean())
    mae_rsd_p = 100 * np.abs(mae_rsd / rsd_true.mean())

    str_rsu =  'Upwelling flux error \nMAE: {:0.2f} ({:0.1f}%)'.format(mae_rsu, mae_rsu_p)
    str_rsd =  'Downwelling flux error \nMAE: {:0.2f} ({:0.1f}%)'.format(mae_rsd, mae_rsd_p)
    errfunc = mae
    #errfunc = rmse
    ind_p = 0
    hr_err      = errfunc(dTdt_true[:,ind_p:], dTdt_pred[:,ind_p:])
    fluxup_err  = errfunc(rsu_true[:,ind_p:], rsu_pred[:,ind_p:])
    fluxdn_err  = errfunc(rsd_true[:,ind_p:], rsd_pred[:,ind_p:])
    fluxnet_err  = errfunc((rsd_true[:,ind_p:] - rsu_true[:,ind_p:]), \
                           (rsd_pred[:,ind_p:] - rsu_pred[:,ind_p:] ))
    yy = 0.01*pres[:,:].mean(axis=0)
    yy2 = 0.01*pres_lay[:,:].mean(axis=0)

    fig, (ax0,ax1) = plt.subplots(ncols=2, sharey=True)
    ax0.plot(hr_err,  yy2[ind_p:], label=str_hre)
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
fpath_test  = datadir +  "/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.nc"
# fpath_test  = datadir +  "/RADSCHEME_data_g224_NWPSAFtest.nc"

# ----------- config ------------

scale_inputs    = True

# didn't seem to improve results
# include_deltap = True
include_deltap = False

include_coldry = False

only_albedo_as_auxinput = True # 
mu0_and_albedo_as_auxinput = False

use_auxinputs = mu0_and_albedo_as_auxinput or only_albedo_as_auxinput

# do outputs consist of the entire profile, i.e. fluxes on halv levels (nlay+1)
# otherwise the TOA incoming flux and surface downward flux can be omitted for nlay outputs
train_on_levflux = True

# third upward RNN before output layer
third_rnn = True

# Model training and evaluation: use GPU or CPU?
use_gpu = True
# test inference of existing model 
evaluate_onnx = True
final_evaluation = True

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
    # x_test  = x_test_raw
  
del x_tr_raw
    
# Number of inputs and outputs    
nx = x_tr.shape[-1]
ny = y_tr_raw.shape[-1]   
nlay = x_tr.shape[-2]
  
if train_on_levflux:
    y_tr = np.concatenate((np.expand_dims(rsd_tr,2), np.expand_dims(rsu_tr,2)),axis=2)
    y_val = np.concatenate((np.expand_dims(rsd_val,2), np.expand_dims(rsu_val,2)),axis=2)
    y_test = np.concatenate((np.expand_dims(rsd_test,2), np.expand_dims(rsu_test,2)),axis=2)
    del y_tr_raw, y_val_raw, y_test_raw
    for i in range(2):
        for j in range(61):
            y_tr[:,j,i] = y_tr[:,j,i] / rsd0_tr
            y_val[:,j,i] = y_val[:,j,i] / rsd0_val
            y_test[:,j,i] = y_test[:,j,i] / rsd0_test
else:
    y_tr = y_tr_raw; y_val = y_val_raw; y_test = y_test_raw


rsd0_tr_big     = rsd0_tr.reshape(-1,1).repeat(nlay+1,axis=1)
rsd0_val_big    = rsd0_val.reshape(-1,1).repeat(nlay+1,axis=1)
rsd0_test_big   = rsd0_test.reshape(-1,1).repeat(nlay+1,axis=1)

if not use_auxinputs: # everything as layer inputs
    x_tr_m = x_tr; x_val_m = x_val; 
    x_test_m = x_test
    # add albedo as aux input anyway to use in loss function
    x_tr_aux1 = x_tr[:,0,-1:]; x_val_aux1 = x_val[:,0,-1:]
    x_test_aux1 = x_test[:,0,-1:]  
    nx_aux = 1
else:
    if only_albedo_as_auxinput: # only one scalar input (albedo)
        x_tr_m = x_tr[:,:,0:-1];  x_val_m = x_val[:,:,0:-1]
        x_test_m = x_test[:,:,0:-1]
        x_tr_aux1 = x_tr[:,0,-1:]; x_val_aux1 = x_val[:,0,-1:]; 
        x_test_aux1 = x_test[:,0,-1:]    
        nx_aux = 1
    else: # two scalar inputs (mu0 and albedo)
        x_tr_m = x_tr[:,:,0:-2];  x_val_m = x_val[:,:,0:-2];  
        x_test_m = x_test[:,:,0:-2]
        
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
    # dp_tr = np.gradient(pres_tr,axis=1)
    # dp_val = np.gradient(pres_val,axis=1)
    dp_tr    = pres_tr[:,1:] - pres_tr[:,0:-1] 
    dp_val   = pres_val[:,1:] - pres_val[:,0:-1] 
    dp_test  = pres_test[:,1:] - pres_test[:,0:-1] 


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
weight_prof = 1/y_tr.mean(axis=0)
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
    return a[:,1:] - a[:,0:-1]


def calc_heatingrates_tf_dp(flux_dn, flux_up, dp):
    #  flux_net =   flux_up   - flux_dn
    F = tf.subtract(flux_up, flux_dn)
    dF = my_gradient_tf(F)
    dFdp = tf.divide(dF, dp)
    coeff = -844.2071713147411#  -(24*3600) * (9.81/1004)  
    dTdt_day = tf.multiply(coeff, dFdp)
    return dTdt_day


def CustomLoss(y_true, y_pred, dp, rsd_top):
    # err_flux =  tf.math.reduce_mean(tf.math.square(weight_prof*(y_true - y_pred)),axis=-1)
    err_flux =  K.mean(K.square(weight_prof*(y_true - y_pred)))
    # err_flux =  K.mean(K.square((y_true - y_pred)))

    rsd_true = tf.math.multiply(y_true[:,:,0], rsd_top)
    rsd_pred = tf.math.multiply(y_pred[:,:,0], rsd_top)
    rsu_true = tf.math.multiply(y_true[:,:,1], rsd_top)
    rsu_pred = tf.math.multiply(y_pred[:,:,1], rsd_top)

    HR_true = calc_heatingrates_tf_dp(rsd_true, rsu_true, dp)
    HR_pred = calc_heatingrates_tf_dp(rsd_pred, rsu_pred, dp)
    # err_hr = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(HR_true - HR_pred),axis=-1))
    err_hr = K.sqrt(K.mean(K.square(HR_true - HR_pred)))

    # alpha   = 1e-6
    # alpha   = 1e-5
    alpha   = 1e-4
    return (alpha) * err_hr + (1 - alpha)*err_flux   


def rmse_hr(y_true, y_pred, dp, rsd_top):
    
    rsd_true = tf.math.multiply(y_true[:,:,0], rsd_top)
    rsd_pred = tf.math.multiply(y_pred[:,:,0], rsd_top)
    rsu_true = tf.math.multiply(y_true[:,:,1], rsd_top)
    rsu_pred = tf.math.multiply(y_pred[:,:,1], rsd_top)

    HR_true = calc_heatingrates_tf_dp(rsd_true, rsu_true, dp)
    HR_pred = calc_heatingrates_tf_dp(rsd_pred, rsu_pred, dp)

    # return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(HR_true - HR_pred),axis=-1))
    return K.sqrt(K.mean(K.square(HR_true - HR_pred)))

# def my_sigmoid(x):
#     x = tf.keras.activations.sigmoid(x)
#     xshape = tf.shape(x) # = (batch_size, 61, 2)
#     xones = tf.ones(xshape)

#     boolmask = tf.expand_dims(boolmask,2)
#     boolmask = tf.transpose(boolmask, perm=[2, 1,0])
#     boolmask = tf.repeat(boolmask, xshape[0],axis=0)
#     xn = tf.where(boolmask, xones, x)

#     return xn

# MODEL TRAINING CODE
if not final_evaluation:
    
    # Model architecture
    # Activation for first
    # activ0 = 'relu'
    activ0 = 'linear'
    # activ0 = 'softsign'
    # activ0 = 'tanh'
    activ_last   = 'relu'
    activ_last   = 'sigmoid'
    # activ_last   = 'linear'
    
    epochs      = 100000
    patience    = 25
    lossfunc    = losses.mean_squared_error
    batch_size  = 1024
    batch_size  = 2048
    
    # nneur = 64 
    nneur = 16  
    # nneur = 24
    # nneur = 32 
    # neur  = 128
    # nneur = 12
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
    # inputs = Input(shape=(None,nx_main),name='inputs_main')
    inputs = Input(shape=(nlay,nx_main),name='inputs_main')
    
    # Optionally, use auxiliary inputs that do not dependend on sequence?
    
    # if use_auxinputs:  # commented out cos I need albedo for the loss function, easier to have it separate
    inp_aux_albedo = Input(shape=(nx_aux),name='inputs_aux_albedo') # sfc_albedo
    if mu0_and_albedo_as_auxinput: inp_aux_mu = Input(shape=(nx_aux),name='inputs_aux_mu0') # mu0
    
    # Target outputs: these are fed as part of the input to avoid problem with 
    # validation data where TF complained about wrong shape
    # target  = Input((None,ny))
    target  = Input((nlay+1,ny))
    
    # other inputs required to compute heating rate
    dpres   = Input((nlay,))
    incflux = Input((nlay+1))
    
            
    # hidden0,last_state = layers.SimpleRNN(nneur,return_sequences=True,return_state=True)(inputs)
    hidden0,last_state = layers.GRU(nneur,return_sequences=True,return_state=True)(inputs)
    
    last_state_plus_albedo =  tf.concat([last_state,inp_aux_albedo],axis=1)
    
    mlp_surface_outp = Dense(nneur, activation=activ0,name='dense_surface')(last_state_plus_albedo)
    
    hidden0_lev = tf.concat([hidden0,tf.reshape(mlp_surface_outp,[-1,1,nneur])],axis=1)
    
    hidden1 = layers.GRU(nneur,return_sequences=True,go_backwards=True)(hidden0_lev)
    # hidden1 = layers.SimpleRNN(nneur,return_sequences=True,go_backwards=True)(hidden0_lev)
    
    # try concatinating hidden0 and hidden 1 instead
    hidden_concat  = tf.concat([hidden0_lev,hidden1],axis=2)
    
    if third_rnn:
        hidden2 = layers.GRU(nneur,return_sequences=True)(hidden_concat)
        # hidden2 = layers.SimpleRNN(nneur,return_sequences=True)(hidden_concat)
    
        outputs = TimeDistributed(layers.Dense(ny, activation=activ_last),name='dense_output')(hidden2)
    else:
        outputs = TimeDistributed(layers.Dense(ny, activation=activ_last),name='dense_output')(hidden_concat)
    
    
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
    
    model.add_metric(rmse_hr(target,outputs,dpres,incflux),'rmse_hr')
    
    # model.add_metric(rmse_hr(target,outputs,inp_aux_albedo,dpres,incflux),'rmse_hr')
    # model.add_metric(rmse_flux(target,outputs,inp_aux,incflux),'rmse_flux')
    
    model.add_loss(CustomLoss(target,outputs,dpres, incflux))
    # model.add_loss(losses.mean_squared_error(target,outputs))
    model.compile(optimizer=optim,loss='mse')
    
    model.summary()
    
    # model.add_metric(heatingrateloss,'heating_rate_mse')
    # model.metrics_names.append("heating_rate_mse")
    
    # # Create earlystopper and possibly other callbacks
    callbacks = [EarlyStopping(monitor='rmse_hr',  patience=patience, verbose=1, \
                                 mode='min',restore_best_weights=True)]
    
    # 16 GRU - BackGRU - GRU
    #     Epoch 10/100000
    # 320/320 [==============================] - 12s 39ms/step - loss: 0.0161 - rmse_hr: 14.5004 - val_loss: 0.0149 - val_rmse_hr: 12.7390
    # stopped at 
    # Epoch 493/100000
    # 320/320 [==============================] - 5s 16ms/step - loss: 5.6712e-04 - rmse_hr: 1.4174 - val_loss: 5.2206e-04 - val_rmse_hr: 1.2316
    
    # 16 GRU - BackGRU - GRU with concatenated albedo layer and predicting all levels
    # Epoch 430/100000
    # 320/320 [==============================] - 5s 17ms/step - loss: 4.5091e-04 - rmse_hr: 1.2175 - val_loss: 4.9115e-04 - val_rmse_hr: 1.1969
    # with concat of the first GRU and BackGRU:
    # Epoch 335/100000
    # 320/320 [==============================] - 6s 17ms/step - loss: 1.9770e-04 - rmse_hr: 1.0664 - val_loss: 2.8051e-04 - val_rmse_hr: 1.0551
    
    # 16 GRU - BackGRU concat
    # Epoch 482/100000
    # 320/320 [==============================] - 4s 11ms/step - loss: 3.3122e-04 - rmse_hr: 0.8857 - val_loss: 4.0671e-04 - val_rmse_hr: 1.0161
    
    # same but tanh activation in surface layer instead of linear
    # Epoch 336/100000
    # 320/320 [==============================] - 4s 13ms/step - loss: 5.5611e-04 - rmse_hr: 1.3147 - val_loss: 5.5451e-04 - val_rmse_hr: 1.1525
    
    # 8 Gru - BackGru - Gru
    # Epoch 424/100000
    # 320/320 [==============================] - 6s 17ms/step - loss: 3.4476e-04 - rmse_hr: 1.2941 - val_loss: 3.3049e-04 - val_rmse_hr: 1.2892
    
    # 16 Gru - Backgru - Gru with coldry
    # Epoch 485/100000
    # 320/320 [==============================] - 5s 16ms/step - loss: 1.5262e-04 - rmse_hr: 0.8288 - val_loss: 1.4025e-04 - val_rmse_hr: 0.8467
    # Restoring model weights from the end of the best epoch.
    
    # 12 Gru - Backgru - Gru with coldry: stuck at RMSE hr 1.01
    
    # 16 Gru - Backgru - Gru with coldry and no flux weights
    # rmse_hr: 0.8889
    
    # 12 Gru BackGru
    # Epoch 756/100000
    # 320/320 [==============================] - 4s 12ms/step - loss: 2.4949e-04 - rmse_hr: 0.7504 - val_loss: 2.7354e-04 - val_rmse_hr: 0.6899
    # 1 core 1.253s (val)
    
    # 24 Gru BackGru : 4.8s,  HR RMSE 0.2-0.3
    # 24 Gru BackGru-16 : 4s
    
    # 16 Gru - Backgru - Gru without coldry and with flux weights, TF 2.50
    # loss 1.13e-04, val rmse 0.23974052; Total params: 5,698
    
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




# INFERENCE USING EXISTING ONNX MODEL
# Test data should NOT be used to compare models, only once for final evaluation!

if evaluate_onnx:
    fpath = 'saved_model/bigru_gru_16_nocoldry_levoutput_FINAL'
    fpath_onnx = fpath+".onnx"
    if final_evaluation:
        xaux = x_test_aux1.reshape(-1,1)
        x_main = x_test_m
        rsd0 = rsd0_test
        rsu = rsu_test; rsd = rsd_test
        pres = pres_test
    else:
        xaux = x_val_aux1.reshape(-1,1)
        x_main = x_val_m
        rsd0 = rsd0_val
        rsu = rsu_val; rsd = rsd_val
        pres = pres_val
        
    # MODEL SAVING; first as TensorFlow SaveModel
    # fpath = 'saved_model/tmp2_bigru_gru_32'
    # fpath = 'saved_model/tmp2_bisimple_simple_32_100epochs'
    # fpath = 'saved_model/tmp'
    # fpath = 'saved_model/bigru_gru_16_nocoldry_levoutput_FINAL'
    # fpath = 'saved_model/levoutput_bigru_gru_24'
    # fpath_onnx = fpath+".onnx"
    # 
    # newmodel.save(fpath, save_format='tf')
    # newmodel.save(fpath, save_format='tf',save_traces=False)
    
    # Now convert to ONNX model
    # os.system("python -m tf2onnx.convert --saved-model {} --output {} --opset 13".format(fpath,fpath_onnx)) 
    
    if not use_gpu:
        # Force one thread on CPUs: Check environmental variables just in case
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
        import onnxruntime as ort
    else:
        import onnxruntime as ort
    
    if not use_gpu:
        # Adjust session options
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess = ort.InferenceSession(fpath_onnx, sess_options=opts)
    else:
        sess = ort.InferenceSession(fpath_onnx, providers=["CUDAExecutionProvider"])
    
    
    # sess.get_providers()
    
    
    start = time.time()
    y_pred = sess.run(["dense_output"], {"inputs_aux_albedo": xaux, "inputs_main": x_main})[0]
    end = time.time()
    print(end - start)
    # BiGRU+GRU 32 : 4.688 s
    # GRU16 : 1.28s
    # GRU16, softsign: 1.28 still
    #...
    # BiSimple+GRU 32 : 2.3s
    # GRU-BackGru-Gru 16 : 1.56737
    # y_pred[:,0,0] = 1.0
    for i in range(2):
           for j in range(61):
               y_pred[:,j,i] = y_pred[:,j,i] * rsd0
               
    rsd_pred = y_pred[:,:,0]
    rsu_pred = y_pred[:,:,1]
    
    
    plot_flux_and_hr_error(rsu, rsd, rsu_pred, rsd_pred, pres)



# from netCDF4 import Dataset

# rootdir = "../fluxes/"
# fname_out = rootdir+'CAMS_2015_rsud_RADSCHEME_RNN.nc'

# dat_out =  Dataset(fname_out,'a')
# var_rsu = dat_out.variables['rsu']
# var_rsd = dat_out.variables['rsd']

# nsite = dat_out.dimensions['site'].size
# ntime = dat_out.dimensions['time'].size
# var_rsu[:] = rsu_pred.reshape(ntime,nsite,nlay+1)
# var_rsd[:] = rsd_pred.reshape(ntime,nsite,nlay+1)

# dat_out.close()


