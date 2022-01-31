"""
Python code for developing neural networks to replace RRTMGP 
gas optics look up table

This program takes existing input-output data generated with RRTMGP and
user-specified hyperparameters such as the number of neurons, optionally
scales the data, and trains a neural network

Contributions welcome!

@author: Peter Ukkonen
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import losses, optimizers
from tensorflow.keras.callbacks import EarlyStopping

from ml_load_save_preproc import save_model_netcdf, \
    load_rrtmgp, scale_outputs, \
    preproc_pow_standardization_reverse,\
    preproc_tau_to_crossection, preproc_minmax_inputs_rrtmgp
from ml_scaling_coefficients import * #  ymeans_sw_absorption, ysigma_sw_absorption, \
   # ymeans_sw_ray, ysigma_sw_ray, ymeans_lw_absorption, ysigma_lw_absorption

from ml_eval_funcs import plot_hist2d, plot_hist2d_T
from ml_trainfuncs_keras import create_model_mlp, savemodel


# ----------------------------------------------------------------------------
# ----------------- Provide data with inputs and outputs ---------------------
# ----------------------------------------------------------------------------

datadir     = "/media/peter/samsung/data/CAMS/ml_training/"
# datadir     = "/home/peter/data/"

# fpath       = datadir+"ml_data_g224_CAMS_2012-2016_noclouds.nc"
# fpath_val   = datadir+"ml_data_g224_CAMS_2017_noclouds.nc"
# fpath_test  = datadir+"ml_data_g224_CAMS_2018_noclouds.nc"

fpath       = datadir+"RRTMGP_data_g224_CAMS_2009-2018_sans_2014-2015_RND.nc"
fpath_val   = datadir+"RRTMGP_data_g224_CAMS_2014_RND.nc"
fpath_test  = datadir+"RRTMGP_data_g224_CAMS_2015_RND.nc"


# Just one dataset
datadir = "/media/peter/samlinux/gdrive/data/ml_training/reduced-k/"
fpath   = datadir+"ml_training_lw_g128_CKDMIP_MMM_big.nc"
fpath_val = None
fpath_test = None

# ------------------------------------------------------
# --- Configure predictand, choice of scaling etc. -----
# ------------------------------------------------------


# Choose one of the following predictands (target output)
# 'lw_absorption', 'lw_planck_frac', 'sw_absorption', 'sw_rayleigh'
# predictand = 'sw_absorption'
# predictand = 'sw_rayleigh'

predictand = 'lw_absorption'
# predictand = 'lw_planck_frac'

# Scaling method: currently fixed so that inputs are min-max scaled to (0..1) and 
# outputs are standardized to ~zero mean, ~unit variance
# However, the devil is in the details, and power transformations
# are first used to reduce dynamical range and make distributions more Gaussian 
# (this can speed up training and improve results)
#
# Input preprocessing:
#   x(i) = log(x(i)) for input feature i = pressure
#   x(i) = x(i)**(1/4) for i = H2O and O3
#   x(i) = x(i) - max(i) / (max(x(i) - min(x(i))))
# 
# Output preprocessing:_
#   1. Normalize optical depths (g-point vectors) by layer number of molecules
#      y(ig,isample) = y_raw(ig,isample) / N (isample) 
#   2. y = y**(1/8)
#   3. ynorm = (y - ymean) / ystd, where ymeans are means for individual
#   g-points, but ystd is the standard deviation across all g-points 
#   (preserves relationships between outputs)
scaling_method = 'Ukkonen2020'

# scale_input = True
# scale_output = True

use_existing_input_scaling_coefficients = True 
# ^True is generally a safe choice, min max coefficients have been computed
# using a large dataset spanning both LGM (Last Glacial Maximum) and high 
# future emissions scenarios. However, check that your scaled inputs 
# fall somewhere in the 0-1 range

use_existing_output_scaling_coefficients = False

# Model training: use CPU or GPU?
use_gpu = False
num_cpu_threads = 12

retrain_mae = False
# -----------------------------------------------------
# --------- Load data ---------------------------------
# -----------------------------------------------------
# Load data given three separate datasets for training - validation - testing
# Training data
x_raw, y_raw, col_dry, input_names, kdist     = load_rrtmgp(fpath, predictand) 

# Validation and testing: separate datasets may already exist
if (fpath_val != None and fpath_test != None): 
    x_tr_raw = x_raw; y_tr_raw = y_raw; col_dry_tr = col_dry
    x_val_raw, y_val_raw, col_dry_val   = load_rrtmgp(fpath_val, predictand)
    x_test_raw,y_test_raw,col_dry_test  = load_rrtmgp(fpath_test, predictand)
else: # if we only have one dataset, split manually
    train_ratio = 0.70
    validation_ratio = 0.15

    ntot = x_raw.shape[0]
    indices = np.random.permutation(ntot)
    train_val_ratio = train_ratio + validation_ratio
    
    train_idx, val_idx, test_idx = indices[:int(ntot*train_ratio)],\
    indices[int(ntot*train_ratio):int(ntot*train_val_ratio)],\
    indices[int(ntot*train_val_ratio):]
    
    x_tr_raw, y_tr_raw, col_dry_tr    = x_raw[train_idx,:], y_raw[train_idx,:], col_dry[train_idx]
    x_val_raw, y_val_raw, col_dry_val = x_raw[val_idx,:], y_raw[val_idx,:], col_dry[val_idx]
    x_test_raw, y_test_raw, col_dry_test = x_raw[test_idx,:], y_raw[test_idx,:], col_dry[test_idx]
    
    
nx = x_tr_raw.shape[1] #  temperature + pressure + gases
ny = y_tr_raw.shape[1] #  number of g-points

# -----------------------------------------------------
# -------- Input and output scaling ------------------
# -----------------------------------------------------
if scaling_method != 'Ukkonen2020':
    print ("Only one type of pre-processing currently supported!")
else:
    # Input scaling - min-max
    if use_existing_input_scaling_coefficients:
        if xcoeffs_all == None:
            sys.exit("Input scaling coefficients (xcoeffs) missing!")
        (xmin_all,xmax_all) = xcoeffs_all
        
        # input_names loaded from file, describes the inputs in their order
        # in the data (x_tr_raw)
        # input_names_all corresponds to xmin_all and xmax_all
        # input_names = ['tlay','play','n2o','co2']
        # Order of inputs may be different than in the existing coefficients:
        a = np.array(input_names_all)
        b = np.array(input_names)
        indices = np.where(b[:, None] == a[None, :])[1]
        xmin = xmin_all[indices]
        xmax = xmax_all[indices]
        
        # input_names = input_names_all[0:nx]
        # xmin = xmin_all[0:nx]
        # xmax = xmax_all[0:nx]
        x_tr = preproc_minmax_inputs_rrtmgp(x_tr_raw, (xmin,xmax))
    else:
        x_tr,xmin,xmax  = preproc_minmax_inputs_rrtmgp(x_tr_raw)
        
    x_val           = preproc_minmax_inputs_rrtmgp(x_val_raw, (xmin,xmax))
    x_test          = preproc_minmax_inputs_rrtmgp(x_test_raw, (xmin,xmax))
    
    # Output scaling
    # first, transform y: y=y**(1/nfac); cheaper and weaker version of 
    # log scaling. nfac = 8 for cross-sections, 2 for Planck fraction
    # After this, use standard-scaling
    if (predictand == 'lw_planck_frac'):
        nfac = 2
        ymean = None; ystd = None
    else: 
        nfac = 8
        if use_existing_output_scaling_coefficients:
            if (predictand == 'sw_absorption'):
                ymean  = ymeans_sw_absorption_224
                ystd   = ysigma_sw_absorption_224
            elif (predictand == 'sw_rayleigh'):
                ymean  = ymeans_sw_ray_224 #
                ystd   = ysigma_sw_ray_224
            elif (predictand == 'lw_absorption'):
                ymean = ymeans_lw_absorption_256
                ystd = ysigma_lw_absorption_256
            else: 
                print("invalid predictand")
        else:
            ymean = np.zeros(ny); ystd = np.zeros(ny)
            y_tr   = preproc_tau_to_crossection(y_tr_raw, col_dry_tr)
    
            for i in range(ny):
                ymean[i] = np.mean(y_tr[:,i]**(1/nfac))
                # ystd[i]  = np.std(y_tr[:,i]**(1/nfac))
            ystd = np.repeat(np.std(y_tr**(1/nfac)),ny)
                
    # Scale data
    y_tr    = scale_outputs(y_tr_raw, col_dry_tr, nfac, ymean, ystd)
    y_val   = scale_outputs(y_val_raw, col_dry_val, nfac,  ymean, ystd)
    y_test  = scale_outputs(y_test_raw, col_dry_test, nfac,  ymean, ystd)
    

# x_tr,y_tr       = scale_gasopt(x_tr_raw, y_tr_raw, col_dry_tr, scale_inputs, 
#         scale_outputs, nfac=nfac, y_mean=ymean, y_sigma=ystd, xcoeffs=xcoeffs)
# # val
# x_val,y_val     = scale_gasopt(x_val_raw, y_val_raw, col_dry_val, scale_inputs, 
#         scale_outputs, nfac=nfac, y_mean=ymean, y_sigma=ystd, xcoeffs=xcoeffs)
# # test
# x_test,y_test   = scale_gasopt(x_test_raw, y_test_raw, col_dry_test, scale_inputs, 
#         scale_outputs, nfac=nfac, y_mean=ymean, y_sigma=ystd, xcoeffs=xcoeffs)

# x_tr[x_tr<0.0] = 0.0
# x_val[x_val<0.0] = 0.0
# x_test[x_test<0.0] = 0.0

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TENSORFLOW-KERAS TRAINING
# ------------------------------------------------------
# --- Model architecture -------------------------------
# ------------------------------------------------------
# Number of neurons in each hidden layer
neurons     = [16,16]
# Activation functions after input layer and hidden layers respectively
activ = ['softsign', 'softsign','linear']
initializer = 'lecun_uniform'

# ------------------------------------------------------
# --- Loss function, batch size and learning rate ------
# ------------------------------------------------------
lossfunc    = losses.mean_squared_error
# lr          = 0.001 
# batch_size  = 1024
batch_size  = 4096
lr          = 0.01

# ------------------------------------------------------
# --- Early stopping : patience and what to monitor ----
# ------------------------------------------------------
patience    = 15
valfunc     = 'val_mean_absolute_error'
epochs      = 800  # set a high number with early stopping
# ------------------

# ------------------------------------------------------
# --- Custom metrics: would be great if we could monitor  
# --- flux errors, but that would require radiation code 
# ------------------------------------------------------
mymetrics   = ['mean_absolute_error']
# batch_size  = 3*batch_size
# lr          = 2 * lr

if use_gpu:
    devstr = '/gpu:0'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

else:
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


# optim = optimizers.Adam(lr=lr,rescale_grad=1/batch_size) 
optim = optimizers.Adam(lr=lr)

# Create and compile model
model = create_model_mlp(nx=nx,ny=ny,neurons=neurons,activ=activ,
                         kernel_init=initializer)
model.compile(loss=lossfunc, optimizer=optim, metrics=mymetrics)
model.summary()

# Create earlystopper and possibly other callbacks
earlystopper = EarlyStopping(monitor=valfunc,  patience=patience, verbose=1, mode='min',restore_best_weights=True)
callbacks = [earlystopper]

# ------------------------------------------------------
# --- Start training -----------------------------------
# ------------------------------------------------------
with tf.device(devstr):
    history = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
                        validation_data=(x_val,y_val), callbacks=callbacks)    
    
if retrain_mae:
    model.compile(loss=losses.mean_absolute_error, optimizer=optim,metrics=['mean_squared_error'])
    callbacks = [EarlyStopping(monitor='val_loss',  patience=patience, verbose=1, mode='min',restore_best_weights=True)]
    with tf.device(devstr):
        history2 = model.fit(x_tr, y_tr, epochs= epochs, batch_size=batch_size, shuffle=True,  verbose=1, 
                            validation_data=(x_val,y_val), callbacks=callbacks)
        
# ------------------------------------------------------
# --- Evaluate on validation data  ---------------------
# ------------------------------------------------------
def eval_valdata():
    y_pred       = model.predict(x_val);  
    y_pred       = preproc_pow_standardization_reverse(y_pred, nfac, ymean, ystd)
    
    if predictand not in ['planck_frac', 'ssa_sw']:
        y_pred = y_pred * (np.repeat(col_dry_val[:,np.newaxis],ny,axis=1))
        
    plot_hist2d(y_val_raw,  y_pred,20,True)   # Optical depth 
    plot_hist2d_T(y_val_raw,y_pred,20,True)   # Transmittance  
    
eval_valdata()

# ------------------------------------------------------
# --- Save model?  -------------------------------------
# ------------------------------------------------------
model.summary()

# fpath_keras = "/media/peter/samlinux/gdrive/phd/soft/rte-rrtmgp-nn/neural/data/tau-sw-ray-7-16-16-CAMS-NEW-mae.h5"
# fpath_keras = "/home/peter/soft/rte-rrtmgp-nn/neural/data/tau-sw-abs-tmp2.h5"
fpath_keras = "../../neural/data/tmp.h5"
model.save(fpath_keras)

# from keras.models import load_model
# fpath_keras = "../../neural/data/BEST_tau-sw-ray-7-16-16_2.h5"
# fpath_keras = '../../neural/data/BEST_tau-sw-abs-7-16-16-mae_2.h5'
# fpath_keras = "../../neural/data/BEST_pfrac-18-16-16.h5"
# fpath_keras = '../../neural/data/BEST_tau-lw-18-58-58.h5'

fpath_netcdf = fpath_keras[:-3]+".nc"

# model = load_model(fpath_keras,compile=False)

# kdist = 'rrtmgp-data-sw-g224-2018-12-04.nc'
# kdist = 'rrtmgp-data-lw-g256-2018-12-04.nc'
# kdist = 'rrtmgp-data-lw-g128-210809.nc'

x_scaling_str = "To get the required NN inputs, do the following: "\
        "x(i) = log(x(i)) for i=pressure; "\
        "x(i) = x(i)**(1/4) for i=H2O and O3; "\
        "x(i) = (x(i) - xmin(i)) / (xmax(i) - xmin(i))"
y_scaling_str = "Model predicts scaled cross-sections. Given the raw NN output y,"\
        " do the following to obtain optical depth: "\
        "y(igpt,j) = ystd(igpt)*y(igpt,j) + ymean(igpt); y(igpt,j) "\
        "= y(igpt,j)**8; y(igpt,j) = y(igpt,j) * layer_dry_air_molecules(j)"
        
data_str = "Extensive training data set comprising of reanalysis, climate model,"\
    " and idealized profiles, which has then been augmented using statistical"\
    " methods (Hypercube sampling). See https://doi.org/10.1029/2020MS002226"

if (predictand == 'sw_absorption'):
    model_str = "Shortwave model predicting ABSORPTION CROSS-SECTION"
elif (predictand == 'sw_rayleigh'):
    model_str = "Shortwave model predicting RAYLEIGH CROSS-SECTION"
elif (predictand == 'lw_absorption'):
    model_str = "Longwave model predicting ABSORPTION CROSS-SECTION"
elif (predictand == 'lw_planck_frac'):
    model_str = "Longwave model predicting PLANCK FRACTION"
    y_scaling_str = "y_pfrac = y_nn * y_nn"
else: 
    model_str = ""

save_model_netcdf(fpath_netcdf, model, activ, input_names, kdist,
                       xmin, xmax, ymean, ystd, y_scaling_comment=y_scaling_str, 
                       x_scaling_comment=x_scaling_str,
                       data_comment=data_str, model_comment=model_str)
    

