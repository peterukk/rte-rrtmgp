#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python framework for developing neural networks to replace radiative
transfer computations, either fully or just one component

This code is for tuning the hyperparameters for NN-RTE+RRTMGP 
(emulator of the entire radiation scheme)

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

from ml_loaddata import load_inp_outp_radscheme, preproc_minmax_inputs, \
    preproc_pow_gptnorm, preproc_pow_gptnorm_reverse
from ml_eval_funcs import plot_hist2d
import matplotlib.pyplot as plt

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
    
def plot_flux_and_hr_error(y_true, y_pred, pres):
    fluxup_true = y_true[:,0:61]; fluxdn_true = y_true[:,61:]
    fluxup_pred = y_pred[:,0:61]; fluxdn_pred = y_pred[:,61:]
    dF_true = np.gradient((fluxdn_true - fluxup_true),axis=1)
    dF_pred = np.gradient((fluxdn_pred - fluxup_pred),axis=1)
    dp = np.gradient(pres,axis=1)
    dFdp_true = dF_true/dp; dFdp_pred = dF_pred/dp
    g = 9.81; cp = 1004 
    dTdt_true = -(24*3600)*(g/cp)*(dFdp_true) # K / h
    dTdt_pred = -(24*3600)*(g/cp)*(dFdp_pred) # K / h
    mse_tot = mse(dTdt_true.flatten(), dTdt_pred.flatten())
    rmse_tot = rmse(dTdt_true.flatten(), dTdt_pred.flatten())
    mae_tot = mae(dTdt_true.flatten(), dTdt_pred.flatten())
    str_hre = 'Heating rate error \nMSE: {:0.3f} \nRMSE: {:0.3f} \nMAE: {:0.3f} '.format(mse_tot,rmse_tot, mae_tot)
    mae_rsu = mae(fluxup_true.flatten(), fluxup_pred.flatten())
    mae_rsd = mae(fluxdn_true.flatten(), fluxdn_pred.flatten())
    str_rsu =  'Upwelling flux error \nMAE: {:0.2f}'.format(mae_rsu)
    str_rsd =  'Downwelling flux error \nMAE: {:0.2f}'.format(mae_rsd)
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

# ----------------------------------------------------------------------------
# ----------------- RTE+RRTMGP EMULATION  ------------------------
# ----------------------------------------------------------------------------

#  ----------------- File paths -----------------
datadir     = "/media/peter/samsung/data/CAMS/ml_training/"
# datadir     = "/home/puk/data/"
fpath       = datadir + "/RADSCHEME_data_g224_CAMS_2009-2018_sans_2014-2015.nc"
fpath_val   = datadir + "/RADSCHEME_data_g224_CAMS_2014.nc"
fpath_test  = datadir +  "/RADSCHEME_data_g224_CAMS_2015.nc"

# ----------- config ------------

scale_inputs    = True
scale_outputs   = True

# Model training: use GPU or CPU?
use_gpu = False

# Tune hyperparameters using KerasTuner?
tune_params = False

# Normalize outputs by inc flux? in this case, no other preproc. needed
norm_by_incflux = True

# ----------- config ------------

# Load data
x_tr_raw, y_tr_raw, pres_tr = load_inp_outp_radscheme(fpath,  \
                        scale_p_h2o_o3 = scale_inputs, return_pressures=True)

if (fpath_val != None and fpath_test != None): # If val and test data exists
    x_val_raw, y_val_raw, pres_val      = load_inp_outp_radscheme(fpath_val,  \
                            scale_p_h2o_o3 = scale_inputs, return_pressures=True)
    x_test_raw,y_test_raw, pres_test    = load_inp_outp_radscheme(fpath_test, \
                            scale_p_h2o_o3 = scale_inputs, return_pressures=True)
else: # if we only have one dataset, split manually
    from sklearn.model_selection import train_test_split
    train_ratio = 0.70
    validation_ratio = 0.15
    test_ratio = 0.15
    testval_ratio = test_ratio/(test_ratio + validation_ratio)
    # first split into two, training and test+val
    x_tr_raw, x_test_raw, y_tr_raw, y_test_raw = \
        train_test_split(x_tr_raw, y_tr_raw, test_size=1-train_ratio)
    # then split the latter into to testing and val
    x_val_raw, x_test_raw, y_val_raw, y_test_raw = \
        train_test_split(x_test_raw, y_test_raw, test_size=testval_ratio) 

# Number of inputs and outputs    
nx = x_tr_raw.shape[1]
ny = y_tr_raw.shape[1]   

if norm_by_incflux: 
    nx = nx + 1
    x_tr_raw = np.hstack((x_tr_raw,     y_tr_raw[:,61].reshape(-1,1)))
    x_val_raw = np.hstack((x_val_raw,   y_val_raw[:,61].reshape(-1,1)))
    x_test_raw = np.hstack((x_test_raw, y_test_raw[:,61].reshape(-1,1)))


if scale_inputs:
    x_tr        = np.copy(x_tr_raw)
    x_val       = np.copy(x_val_raw)
    x_test      = np.copy(x_test_raw)
    
    fpath_xcoeffs = "../../../neural/data/nn_radscheme_xmin_xmax.txt"
    xcoeffs = np.loadtxt(fpath_xcoeffs, delimiter=',')
    xmax = xcoeffs[0:nx]
    xmin = np.repeat(0.0, nx)

    x_tr            = preproc_minmax_inputs(x_tr_raw, (xmin, xmax))
    # x_tr, xmin,xmax = preproc_minmax_inputs(x_tr_raw)
    x_val           = preproc_minmax_inputs(x_val_raw,  (xmin,xmax)) 
    x_test          = preproc_minmax_inputs(x_test_raw, (xmin,xmax)) 
else:
    x_tr    = x_tr_raw
    x_val   = x_val_raw
    x_test  = x_test_raw
    
    
if norm_by_incflux: 
    y_tr    = y_tr_raw / np.repeat(y_tr_raw[:,61].reshape(-1,1),y_tr_raw.shape[1], axis=1)
    y_val   = y_val_raw / np.repeat(y_val_raw[:,61].reshape(-1,1),y_val_raw.shape[1], axis=1)
    y_test  = y_test_raw / np.repeat(y_test_raw[:,61].reshape(-1,1),y_test_raw.shape[1], axis=1)
else:
    y_tr    = y_tr_raw    
    y_val   = y_val_raw
    y_test  = y_test_raw
    
    
pres_tr_grad = np.gradient(pres_tr,axis=1)
pres_val_grad = np.gradient(pres_val,axis=1)
pres_test_grad = np.gradient(pres_test,axis=1)


gc.collect()
# Ready for training

# TENSORFLOW-KERAS TRAINING
import optuna
import tensorflow as tf
from tensorflow.keras import losses, optimizers, layers, Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from ml_trainfuncs_keras import create_model_mlp, savemodel
import tensorflow.keras.backend as K

mymetrics   = ['mean_absolute_error']
valfunc     = 'val_mean_absolute_error'

# Model architecture
# First hidden layer (input layer) activation
activ0      = 'softsign'
# activ0       = 'relu'
# Activation in other hidden layers
activ       =  activ0    
# Activation for last layer
activ_last   = 'linear'
# activ_last   = 'relu'
activ_last   = 'sigmoid'

epochs      = 100000
patience    = 25
lossfunc    = losses.mean_squared_error
ninputs     = x_tr.shape[1]
lr          = 0.0001 
batch_size  = 1024

               
num_cpu_threads = 12
devstr = '/cpu:0'
# Maximum number of threads to use for OpenMP parallel regions.
os.environ["OMP_NUM_THREADS"] = str(num_cpu_threads)
# Without setting below 2 environment variables, it didn't work for me. Thanks to @cjw85 
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_cpu_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = str(1)
os.environ['KMP_BLOCKTIME'] = '1' 

tf.config.threading.set_intra_op_parallelism_threads(
    num_cpu_threads
)
tf.config.threading.set_inter_op_parallelism_threads(
    1
)
tf.config.set_soft_device_placement(True)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


batch_size  = 2048
lr          = 0.001 

optim = optimizers.Adam(learning_rate=lr)

    
def my_gradient_tf(a):
    rght = tf.concat((a[..., 1:], tf.expand_dims(a[..., -1], -1)), -1)
    left = tf.concat((tf.expand_dims(a[...,0], -1), a[..., :-1]), -1)
    ones = tf.ones_like(rght[..., 2:], tf.float32)
    one = tf.expand_dims(ones[...,0], -1)
    divi = tf.concat((one, ones*2, one), -1)
    return (rght-left) / divi

def calc_heatingrates_tf_dp(y, dp):
    #  flux_net =   flux_up   - flux_dn
    F = tf.subtract(y[:,0:61], y[:,61:])
    dF = my_gradient_tf(F)
    dFdp = tf.divide(dF, dp)
    coeff = -8842568.807339448 #  -24*3600 / (9.81/1004)  
    dTdt_day = tf.multiply(coeff, dFdp)
    return dTdt_day

def CustomLoss(y_true, y_pred, input_tensor, dpres):
    err_flux = K.sqrt(K.mean(K.square(y_true - y_pred)))
    
    # need to reshape incflux from (batchsize,)  to (batchsize, 122)            
    fluxbig= tf.repeat(tf.expand_dims(input_tensor[:,542], axis=1), 122, axis=1)
    
    flux_true = tf.math.multiply(fluxbig, y_true)
    flux_pred = tf.math.multiply(fluxbig, y_pred)
    
    HR_true = calc_heatingrates_tf_dp(flux_true, dpres)
    HR_pred = calc_heatingrates_tf_dp(flux_pred, dpres)
    err_hr = K.sqrt(K.mean(K.square(HR_true - HR_pred)))
    
    # alpha = 1
    # alpha = 0.001
    # alpha = 0.0005 # best so far, hybridloss.h5
    # alpha = 0.0003 # hybridloss2
    alpha   = 0.0007
    # alpha = 0.0008 # best so far, hybridloss3
    err = (alpha) * err_hr + (1 - alpha)*err_flux
        
    return err
    
batch_size  = 128
lr          = 0.001 
optim = optimizers.Adam(learning_rate=lr)
# optim = COCOB()
# neurons = [128] # super bad
neurons     = [128,128,128]
neurons     = [192, 128]

dense   = layers.Dense(neurons[0], activation=activ0)
inp     = Input(shape=(nx,))
x       = dense(inp)
# more hidden layers
for i in range(1,np.size(neurons)):
    x       = layers.Dense(neurons[i], activation=activ)(x)
out     = layers.Dense(ny, activation=activ_last)(x)
target  = Input((ny,))
dpres   = Input((61,))
model   = Model(inputs=[inp,target, dpres], outputs=out)
model.add_loss(CustomLoss(target,out,inp,dpres))
model.compile(loss=None, optimizer=optim)

callbacks = [EarlyStopping(monitor='val_loss',  patience=21, \
                verbose=1, mode='min',restore_best_weights=True)]
with tf.device(devstr):
    history = model.fit(x=[x_tr,y_tr, pres_tr_grad], y=None,    \
    epochs= epochs, batch_size=batch_size, shuffle = True,      \
    validation_data=[x_val,y_val,pres_val_grad], callbacks=callbacks)
    
        
y_pred      = model.predict([x_test,y_test,pres_test_grad]);

y_pred = y_pred * np.repeat(y_test_raw[:,61].reshape(-1,1),y_test_raw.shape[1], axis=1)


cc = np.corrcoef(y_test_raw.flatten(), y_pred.flatten())
diff = np.abs(y_test_raw-y_pred)
rmse_err = np.sqrt(((y_pred - y_test_raw) ** 2).mean())
print("r {} max diff {} RMSE {}".format(cc[0,1],np.max(diff), rmse_err))

# plot_hist2d(y_test_raw,y_pred,20,True)      # 

plot_flux_and_hr_error(y_test_raw, y_pred, pres_test)


# SAVE MODEL
w1 = model.layers[1].get_weights()
w2 = model.layers[2].get_weights()
# w3 = model.layers[5].get_weights()
w3 = model.layers[3].get_weights()
w4 = model.layers[6].get_weights()
model = create_model_mlp(nx=nx,ny=ny,neurons=neurons,activ0=activ0,activ=activ,
                      activ_last = activ_last, kernel_init='lecun_uniform')
model.layers[0].set_weights(w1)
model.layers[1].set_weights(w2)
model.layers[2].set_weights(w3)
model.layers[3].set_weights(w4)
kerasfile = "../../../neural/data/radscheme-128-128-128-hybridloss2.h5"
savemodel(kerasfile, model)

from tensorflow.keras.models import load_model
kerasfile = "../../../neural/data/neural/data/radscheme-128.h5"
model = tf.lite.TFLiteConverter.from_keras_model(kerasfile)
model = load_model(kerasfile,compile=False)

   